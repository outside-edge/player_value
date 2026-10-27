"""Current five-year player rankings, fitted separately by format and gender."""

import math
import platform
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from player_value.io import sha256, staged_directory, write_json
from player_value.lineup import features, lineup_diagnostics, load_prepared, read_csv
from player_value.preparation import write_csv

FORMATS = {"Test": "tests", "ODI": "odis", "T20I": "t20is"}
CANDIDATES = (0.01, 0.1, 1.0, 10.0)
RANK_COLUMNS = [
    "format",
    "gender",
    "as_of",
    "training_start",
    "comparison_group",
    "rank",
    "player_id",
    "player_name",
    "teams",
    "rating_log_win_loss_odds",
    "training_appearances",
    "last_appearance",
    "fewer_than_20_appearances",
    "regularization_c",
    "tuning_status",
    "max_teammate_share",
    "identical_pattern_group",
]


def years_before(day, years):
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def outcome_class(match):
    if match.result in {"draw", "tie"} and not match.winner:
        return 1
    if match.result == "win" and match.winner in {match.team_a, match.team_b}:
        return 2 if match.winner == match.team_a else 0
    raise ValueError(f"{match.match_id}: unsupported ranking outcome")


@dataclass
class CurrentFit:
    vectorizer: DictVectorizer
    estimator: LogisticRegression

    @property
    def coefficients(self):
        values = self.estimator.coef_
        return values[0] if len(values) == 1 else values[2] - values[0]

    def predict(self, matches):
        matrix = self.vectorizer.transform(features(m, "player") for m in matches)
        raw = self.estimator.predict_proba(matrix)
        result = np.zeros((len(matches), 3))
        result[:, self.estimator.classes_] = raw
        return result


def fit_current(matches, regularization_c):
    if not matches or not any(outcome_class(m) != 1 for m in matches):
        raise ValueError("Ranking requires at least one decisive match")
    if len({m.gender for m in matches}) != 1:
        raise ValueError("Fit men's and women's matches separately")
    if not np.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("Regularization C must be positive and finite")
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(features(m, "player") for m in matches)
    outcomes = np.array([outcome_class(m) for m in matches])
    has_draws = bool(np.any(outcomes == 1))
    estimator = LogisticRegression(
        C=regularization_c / 2 if has_draws else regularization_c,
        l1_ratio=0,
        solver="lbfgs",
        fit_intercept=has_draws,
        tol=1e-8,
        max_iter=2000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(
                vstack([matrix, -matrix]),
                np.concatenate([outcomes, 2 - outcomes]),
                sample_weight=np.full(2 * len(matches), 0.5),
            )
        except ConvergenceWarning as error:
            raise ValueError("Current ranking optimization did not converge") from error
    return CurrentFit(vectorizer, estimator)


def select_regularization(matches, as_of):
    boundary = years_before(as_of, 1)
    training = [m for m in matches if m.end_date <= boundary]
    validation = [m for m in matches if m.start_date > boundary]
    has_draw = any(outcome_class(m) == 1 for m in matches)
    usable = (
        len(training) >= 20
        and len(validation) >= 10
        and any(outcome_class(m) != 1 for m in training)
        and (not has_draw or any(outcome_class(m) == 1 for m in training))
    )
    scores = []
    if usable:
        target = np.array([outcome_class(m) for m in validation])
        for value in CANDIDATES:
            fit = fit_current(training, value)
            probabilities = fit.predict(validation)[np.arange(len(validation)), target]
            scores.append(
                {
                    "regularization_c": value,
                    "estimator_c": fit.estimator.C,
                    "validation_log_loss": float(
                        -np.log(np.clip(probabilities, 1e-15, 1)).mean()
                    ),
                }
            )
        selected = min(
            scores, key=lambda s: (s["validation_log_loss"], s["regularization_c"])
        )["regularization_c"]
    else:
        selected = 0.1
    return selected, {
        "status": "tuned" if usable else "untuned_sparse_data",
        "boundary": boundary.isoformat(),
        "training_ids": [m.match_id for m in training],
        "validation_ids": [m.match_id for m in validation],
        "boundary_crossing_ids": [
            m.match_id for m in matches if m.start_date <= boundary < m.end_date
        ],
        "candidates": scores,
    }


def comparison_groups(matches):
    parent = {}

    def find(player):
        parent.setdefault(player, player)
        if parent[player] != player:
            parent[player] = find(parent[player])
        return parent[player]

    for match in matches:
        players = match.players_a + match.players_b
        root = find(players[0])
        for player in players[1:]:
            parent[find(player)] = root
    members = defaultdict(list)
    for player in parent:
        members[find(player)].append(player)
    return {player: min(group) for group in members.values() for player in group}


def player_rows(matches, fit, match_format, as_of, start, regularization_c, status):
    diagnostics, _ = lineup_diagnostics(matches)
    groups = comparison_groups(matches)
    teams, last = defaultdict(set), {}
    for match in matches:
        for team, players in (
            (match.team_a, match.players_a),
            (match.team_b, match.players_b),
        ):
            for player in players:
                teams[player].add(team)
                last[player] = max(last.get(player, match.end_date), match.end_date)
    rows = []
    for feature, coefficient in zip(
        fit.vectorizer.get_feature_names_out(), fit.coefficients, strict=True
    ):
        if not feature.startswith("player:"):
            continue
        if not math.isfinite(coefficient):
            raise ValueError("Nonfinite player coefficient")
        player = feature.removeprefix("player:")
        info = diagnostics[player]
        rows.append(
            {
                "format": match_format,
                "gender": matches[0].gender,
                "as_of": as_of.isoformat(),
                "training_start": start.isoformat(),
                "comparison_group": groups[player],
                "player_id": player,
                **info,
                "teams": ";".join(sorted(teams[player])),
                "rating_log_win_loss_odds": float(coefficient),
                "last_appearance": last[player].isoformat(),
                "fewer_than_20_appearances": int(info["training_appearances"] < 20),
                "regularization_c": regularization_c,
                "tuning_status": status,
            }
        )
    rows.sort(
        key=lambda row: (
            row["comparison_group"],
            -round(row["rating_log_win_loss_odds"], 12),
            row["player_id"],
        )
    )
    counters = Counter()
    previous = {}
    ranks = {}
    for row in rows:
        group = row["comparison_group"]
        counters[group] += 1
        value = round(row["rating_log_win_loss_odds"], 12)
        if previous.get(group) != value:
            ranks[group] = counters[group]
        row["rank"] = ranks[group]
        previous[group] = value
    return rows


def report(rows, cohorts, as_of):
    lines = [
        f"Rankings through **{as_of}**, using the preceding five years "
        "of internationals.",
        "Men and women, and each format, are fitted separately. All players in the "
        "window are included, including those now retired.",
        "[Download all rankings](../out/release/current/rankings.csv) · "
        "[Fit details and provenance](../out/release/current/manifest.json)",
        "Ratings are additive log win/loss odds. They are model associations, not WAR. "
        "Ranks apply only within a format, gender, and connected comparison group. "
        "An asterisk marks fewer than 20 appearances; exact tied ranks are retained.",
    ]
    for cohort in cohorts:
        fmt, gender = cohort["format"], cohort["gender"]
        label = "Men" if gender == "male" else "Women"
        lines += [
            f"## {fmt} — {label}",
            f"{cohort['matches']} matches; {cohort['draws']} draws/ties; "
            f"{cohort['players']} players.",
        ]
        if cohort["status"] == "unavailable":
            lines.append("No ranking: no eligible decisive matches in this window.")
            continue
        lines.append(
            f"C = {cohort['regularization_c']:g}; "
            + (
                "selected using the final year's validation matches."
                if cohort["tuning"]["status"] == "tuned"
                else "**untuned, sparse-data estimates** using the specified default."
            )
        )
        lines.append(
            f"[Full {label.lower()}'s {fmt} table]"
            f"(../out/release/current/{FORMATS[fmt]}_{gender}.csv)"
        )
        selected = [r for r in rows if r["format"] == fmt and r["gender"] == gender]
        groups = Counter(r["comparison_group"] for r in selected)
        for group, count in groups.most_common():
            if len(groups) > 1:
                lines.append(f"### Comparison group `{group}` ({count} players)")
            table = [
                "| Rank | Player | Teams | Rating | Matches |",
                "| ---: | --- | --- | ---: | ---: |",
            ]
            for row in selected:
                if row["comparison_group"] != group or row["rank"] > 20:
                    continue
                star = "*" if row["fewer_than_20_appearances"] else ""
                name = row["player_name"].replace("|", "\\|")
                table.append(
                    f"| {row['rank']} | {name}{star} | {row['teams']} | "
                    f"{row['rating_log_win_loss_odds']:.3f} | "
                    f"{row['training_appearances']} |"
                )
            lines.append("\n".join(table))
    return "\n\n".join(lines) + "\n"


def rank_formats(input_root, output, as_of, specification="docs/format_rankings.qmd"):
    specification = Path(specification)
    if not specification.is_file():
        raise ValueError("Ranking specification file is missing")
    if Path(input_root).resolve().is_relative_to(Path(output).resolve()):
        raise ValueError("Ranking output must not contain its inputs")
    as_of = date.fromisoformat(str(as_of))
    start = years_before(as_of, 5) + timedelta(days=1)
    rows, cohorts, preparations, models = [], [], {}, {}
    for match_format, folder in FORMATS.items():
        directory = Path(input_root) / folder
        matches, manifest = load_prepared(
            directory, include_draws=match_format == "Test"
        )
        if manifest.get("format") != match_format:
            raise ValueError(f"Expected {match_format} preparation in {directory}")
        if date.fromisoformat(manifest["as_of"]) < as_of:
            raise ValueError("Preparation cutoff precedes requested ranking cutoff")
        if manifest.get("since") and date.fromisoformat(manifest["since"]) > start:
            raise ValueError("Preparation start omits part of the ranking window")
        preparations[folder] = manifest
        records = read_csv(directory / "matches.csv")
        for gender in ("male", "female"):
            cohort = [
                m
                for m in matches
                if m.gender == gender and m.start_date >= start and m.end_date <= as_of
            ]
            window = [
                r
                for r in records
                if r["gender"] == gender
                and start.isoformat() <= r["start_date"]
                and r["end_date"] <= as_of.isoformat()
            ]
            eligible = {m.match_id for m in cohort}
            exclusions = Counter(
                r["lineup_exclusion"] or "other"
                for r in window
                if r["match_id"] not in eligible
            )
            info = {
                "format": match_format,
                "gender": gender,
                "matches": len(cohort),
                "draws": sum(outcome_class(m) == 1 for m in cohort),
                "players": len({p for m in cohort for p in m.players_a + m.players_b}),
                "available_window_matches": len(window),
                "excluded": dict(exclusions),
                "match_ids": [m.match_id for m in cohort],
            }
            if not cohort or not any(outcome_class(m) != 1 for m in cohort):
                info["status"] = "unavailable"
                cohorts.append(info)
                continue
            value, tuning = select_regularization(cohort, as_of)
            fit = fit_current(cohort, value)
            ranked = player_rows(
                cohort, fit, match_format, as_of, start, value, tuning["status"]
            )
            rows.extend(ranked)
            _, diagnostics = lineup_diagnostics(cohort)
            info.update(
                status="fitted",
                regularization_c=value,
                tuning=tuning,
                diagnostics=diagnostics,
                comparison_groups=len({r["comparison_group"] for r in ranked}),
            )
            cohorts.append(info)
            models[f"{folder}_{gender}"] = {
                "features": fit.vectorizer.get_feature_names_out().tolist(),
                "classes": fit.estimator.classes_.tolist(),
                "coefficients": fit.estimator.coef_.tolist(),
                "intercepts": fit.estimator.intercept_.tolist(),
                "regularization_c": value,
                "estimator_c": fit.estimator.C,
            }
    with staged_directory(output) as stage:
        write_csv(stage / "rankings.csv", RANK_COLUMNS, rows)
        for fmt, folder in FORMATS.items():
            write_json(stage / f"{folder}_preparation.json", preparations[folder])
            for gender in ("male", "female"):
                write_csv(
                    stage / f"{folder}_{gender}.csv",
                    RANK_COLUMNS,
                    [r for r in rows if r["format"] == fmt and r["gender"] == gender],
                )
        write_json(stage / "models.json", models)
        (stage / "report.md").write_text(report(rows, cohorts, as_of), encoding="utf-8")
        (stage / "specification.qmd").write_bytes(specification.read_bytes())
        manifest = {
            "schema_version": 1,
            "as_of": as_of.isoformat(),
            "training_start": start.isoformat(),
            "model": "player_only",
            "cohorts": cohorts,
            "specification_sha256": sha256(specification),
            "code_sha256": {
                p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "versions": {
                "python": platform.python_version(),
                **{p: version(p) for p in ("numpy", "scipy", "scikit-learn", "PyYAML")},
            },
            "outputs": {p.name: sha256(p) for p in sorted(stage.iterdir())},
        }
        write_json(stage / "manifest.json", manifest)
    return manifest
