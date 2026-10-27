"""Past-only tuning and paired forecast evaluation for the ODI lineup pilot."""

from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np

from player_value.io import sha256, staged_directory, write_json
from player_value.lineup import MODELS, fit_lineup, lineup_diagnostics, load_prepared
from player_value.preparation import write_csv

DEFAULT_C = (0.01, 0.1, 1.0, 10.0)


def losses(outcomes, probabilities):
    outcomes = np.asarray(outcomes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if outcomes.shape != probabilities.shape or outcomes.size == 0:
        raise ValueError("Outcomes and predictions must have the same nonempty shape")
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("Outcomes must be binary")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Predictions must be finite probabilities")
    p = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "log_loss": -(outcomes * np.log(p) + (1 - outcomes) * np.log1p(-p)),
        "brier": (outcomes - probabilities) ** 2,
    }


def split_matches(matches, gender, year, window):
    start, validation_start = date(year - window, 1, 1), date(year - 1, 1, 1)
    test_start, test_end = date(year, 1, 1), date(year + 1, 1, 1)
    selected = [m for m in matches if m.gender == gender]
    training = [
        m for m in selected if m.start_date >= start and m.end_date < validation_start
    ]
    validation = [
        m
        for m in selected
        if m.start_date >= validation_start and m.end_date < test_start
    ]
    testing = [m for m in selected if test_start <= m.start_date < test_end]
    if not all([training, validation, testing]):
        raise ValueError(f"Empty split: {gender}, {year}, {window}-year window")
    return training, validation, testing


def tune(training, validation, model, c_grid):
    scores = []
    for c in sorted(set(c_grid)):
        fit = fit_lineup(training, model, c)
        score = losses([m.outcome for m in validation], fit.predict(validation))
        scores.append(
            {
                "regularization_c": c,
                "validation_log_loss": float(score["log_loss"].mean()),
            }
        )
    best = min(
        scores, key=lambda row: (row["validation_log_loss"], row["regularization_c"])
    )
    return best["regularization_c"], scores


def paired_interval(differences, blocks, replicates=2000, seed=20261027):
    differences = np.asarray(differences, dtype=float)
    labels, assignments = np.unique(blocks, return_inverse=True)
    if len(labels) < 2:
        raise ValueError("Block intervals require at least two nonempty quarters")
    counts = np.bincount(assignments)
    sums = np.bincount(assignments, weights=differences)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(labels), size=(replicates, len(labels)))
    estimates = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "difference": float(differences.mean()),
        "lower": float(low),
        "upper": float(high),
        "quarter_blocks": len(labels),
    }


def summarize(predictions, replicates, seed):
    metrics, calibration, comparisons = [], [], []
    paired_losses = []
    groups = sorted({(r["gender"], r["window_years"]) for r in predictions})
    for gender, window in groups:
        sample = [
            r
            for r in predictions
            if (r["gender"], r["window_years"]) == (gender, window)
        ]
        periods = ["pooled"] + sorted({str(r["evaluation_year"]) for r in sample})
        for model in ("coin",) + MODELS:
            for period in periods:
                rows = [
                    r
                    for r in sample
                    if r["model"] == model
                    and (period == "pooled" or str(r["evaluation_year"]) == period)
                ]
                y, p = [r["outcome"] for r in rows], [r["probability_a"] for r in rows]
                scores = losses(y, p)
                metrics.append(
                    {
                        "gender": gender,
                        "window_years": window,
                        "model": model,
                        "period": period,
                        "matches": len(rows),
                        **{key: float(value.mean()) for key, value in scores.items()},
                    }
                )
                bins = np.minimum((np.array(p) * 5).astype(int), 4)
                for index in range(5):
                    mask = bins == index
                    if mask.any():
                        calibration.append(
                            {
                                "gender": gender,
                                "window_years": window,
                                "model": model,
                                "period": period,
                                "bin_lower": index / 5,
                                "bin_upper": (index + 1) / 5,
                                "matches": int(mask.sum()),
                                "mean_probability": float(np.array(p)[mask].mean()),
                                "observed_fraction": float(np.array(y)[mask].mean()),
                            }
                        )
        baseline = {r["match_id"]: r for r in sample if r["model"] == "team"}
        for model in ("player", "combined"):
            rows = [r for r in sample if r["model"] == model]
            y = [r["outcome"] for r in rows]
            candidate_loss = losses(y, [r["probability_a"] for r in rows])
            baseline_loss = losses(
                y, [baseline[r["match_id"]]["probability_a"] for r in rows]
            )
            blocks = [
                f'{r["start_date"][:4]}-Q{(int(r["start_date"][5:7]) - 1) // 3 + 1}'
                for r in rows
            ]
            for index, row in enumerate(rows):
                paired_losses.append(
                    {
                        "gender": gender,
                        "window_years": window,
                        "evaluation_year": row["evaluation_year"],
                        "model": model,
                        "baseline": "team",
                        "match_id": row["match_id"],
                        "quarter_block": blocks[index],
                        "log_loss_difference": float(
                            candidate_loss["log_loss"][index]
                            - baseline_loss["log_loss"][index]
                        ),
                        "brier_difference": float(
                            candidate_loss["brier"][index]
                            - baseline_loss["brier"][index]
                        ),
                    }
                )
            for metric in candidate_loss:
                interval = paired_interval(
                    candidate_loss[metric] - baseline_loss[metric],
                    blocks,
                    replicates,
                    seed,
                )
                comparisons.append(
                    {
                        "gender": gender,
                        "window_years": window,
                        "model": model,
                        "baseline": "team",
                        "metric": metric,
                        "matches": len(rows),
                        **interval,
                    }
                )
    return metrics, calibration, comparisons, paired_losses


def compare(
    input_dir,
    output_dir,
    years=(2023, 2024, 2025),
    windows=(5, 3),
    c_grid=DEFAULT_C,
    replicates=2000,
    seed=20261027,
):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Analysis output must not contain prepared inputs")
    years, windows = sorted(set(years)), sorted(set(windows))
    if not years or not windows or min(windows) < 2 or not c_grid or replicates < 100:
        raise ValueError("Require years, windows >= 2, a C grid, and >= 100 replicates")
    matches, preparation = load_prepared(input_dir)
    if date.fromisoformat(preparation["as_of"]) < date(max(years), 12, 31):
        raise ValueError("Preparation cutoff does not cover all evaluation years")
    predictions, tuning, coefficients, diagnostics = [], [], [], []
    split_records = []
    for gender in ("male", "female"):
        for window in windows:
            for year in years:
                training, validation, testing = split_matches(
                    matches, gender, year, window
                )
                refit = [
                    m
                    for m in matches
                    if m.gender == gender
                    and m.start_date >= date(year - window, 1, 1)
                    and m.end_date < date(year, 1, 1)
                ]
                player_info, summary = lineup_diagnostics(refit)
                key = {
                    "gender": gender,
                    "window_years": window,
                    "evaluation_year": year,
                }
                diagnostics.append(
                    {
                        **key,
                        "training_matches": len(refit),
                        "validation_matches": len(validation),
                        "evaluation_matches": len(testing),
                        **summary,
                    }
                )
                split_records.append(
                    {
                        **key,
                        "training_ids": [m.match_id for m in training],
                        "validation_ids": [m.match_id for m in validation],
                        "refit_ids": [m.match_id for m in refit],
                        "evaluation_ids": [m.match_id for m in testing],
                    }
                )
                for model in MODELS:
                    best_c, scores = tune(training, validation, model, c_grid)
                    tuning.extend({**key, "model": model, **score} for score in scores)
                    fit = fit_lineup(refit, model, best_c)
                    for match, probability in zip(
                        testing, fit.predict(testing), strict=True
                    ):
                        predictions.append(
                            {
                                **key,
                                "model": model,
                                "match_id": match.match_id,
                                "start_date": match.start_date.isoformat(),
                                "end_date": match.end_date.isoformat(),
                                "team_a": match.team_a,
                                "team_b": match.team_b,
                                "outcome": match.outcome,
                                "probability_a": float(probability),
                                "regularization_c": best_c,
                                "unseen_players": sum(
                                    p not in fit.known_players
                                    for p in match.players_a + match.players_b
                                ),
                                "unseen_teams": sum(
                                    t not in fit.known_teams
                                    for t in (match.team_a, match.team_b)
                                ),
                            }
                        )
                    for feature, coefficient in zip(
                        fit.vectorizer.get_feature_names_out(),
                        fit.estimator.coef_[0],
                        strict=True,
                    ):
                        kind, entity = feature.split(":", 1)
                        info = player_info.get(entity, {}) if kind == "player" else {}
                        coefficients.append(
                            {
                                **key,
                                "model": model,
                                "feature_kind": kind,
                                "entity": entity,
                                "coefficient_log_odds": float(coefficient),
                                "regularization_c": best_c,
                                "player_name": info.get("player_name", ""),
                                "training_appearances": info.get(
                                    "training_appearances", ""
                                ),
                                "max_teammate_share": info.get(
                                    "max_teammate_share", ""
                                ),
                                "identical_pattern_group": info.get(
                                    "identical_pattern_group", ""
                                ),
                            }
                        )
                predictions.extend(
                    {
                        **key,
                        "model": "coin",
                        "match_id": m.match_id,
                        "start_date": m.start_date.isoformat(),
                        "end_date": m.end_date.isoformat(),
                        "team_a": m.team_a,
                        "team_b": m.team_b,
                        "outcome": m.outcome,
                        "probability_a": 0.5,
                        "regularization_c": "",
                        "unseen_players": "",
                        "unseen_teams": "",
                    }
                    for m in testing
                )
    metrics, calibration, comparisons, paired_losses = summarize(
        predictions, replicates, seed
    )
    tables = {
        "predictions": predictions,
        "tuning": tuning,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "calibration": calibration,
        "comparisons": comparisons,
        "paired_losses": paired_losses,
    }
    manifest = {
        "schema_version": 1,
        "experiment": "odi_lineup",
        "years": years,
        "windows": windows,
        "c_grid": list(c_grid),
        "bootstrap_replicates": replicates,
        "seed": seed,
        "preparation_manifest_sha256": sha256(input_dir / "manifest.json"),
        "preparation_as_of": preparation["as_of"],
        "specification_sha256": (
            sha256(Path("docs/lineup_experiment.qmd"))
            if Path("docs/lineup_experiment.qmd").exists()
            else None
        ),
        "dependencies": {
            name: version(name) for name in ["numpy", "scipy", "scikit-learn"]
        },
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "splits": split_records,
    }
    with staged_directory(output_dir) as stage:
        (stage / "preparation_manifest.json").write_bytes(
            (input_dir / "manifest.json").read_bytes()
        )
        for name, rows in tables.items():
            write_csv(stage / f"{name}.csv", list(rows[0]), rows)
        manifest["outputs"] = {
            f"{name}.csv": sha256(stage / f"{name}.csv") for name in tables
        }
        manifest["outputs"]["preparation_manifest.json"] = sha256(
            stage / "preparation_manifest.json"
        )
        write_json(stage / "manifest.json", manifest)
    return manifest
