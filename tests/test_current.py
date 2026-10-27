import json
from copy import deepcopy
from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from test_lineup import simulated_matches

from player_value.current import (
    FORMATS,
    comparison_groups,
    fit_current,
    outcome_class,
    player_rows,
    rank_formats,
    select_regularization,
    years_before,
)
from player_value.lineup import features, fit_lineup, load_prepared, read_csv
from player_value.preparation import prepare
from player_value.publication import verify_outputs
from player_value.site import prepare_site_content


def test_binary_orientation_weights_preserve_existing_estimator():
    matches, _ = simulated_matches(200)
    old = fit_lineup(matches, "player", 0.1)
    new = fit_current(matches, 0.1)
    assert np.allclose(old.estimator.coef_[0], new.coefficients, atol=1e-7)
    assert np.allclose(old.predict(matches), new.predict(matches)[:, 2], atol=1e-7)
    assert np.all(new.predict(matches)[:, 1] == 0)


def draw_matches(count=2200):
    matches, truth = simulated_matches(count)
    rng = np.random.default_rng(31)
    output = []
    for m in matches:
        eta = sum(truth[p] for p in m.players_a) - sum(truth[p] for p in m.players_b)
        weights = np.array([np.exp(-eta / 2), 0.8, np.exp(eta / 2)])
        outcome = rng.choice(3, p=weights / weights.sum())
        output.append(
            replace(
                m,
                result="draw" if outcome == 1 else "win",
                winner="" if outcome == 1 else (m.team_a if outcome == 2 else m.team_b),
            )
        )
    return output, truth


def test_draw_fit_recovers_direction_and_is_symmetric():
    matches, truth = draw_matches()
    fit = fit_current(matches, 1)
    coefficients = dict(
        zip(fit.vectorizer.get_feature_names_out(), fit.coefficients, strict=True)
    )
    assert (
        np.corrcoef(list(truth.values()), [coefficients[f"player:{p}"] for p in truth])[
            0, 1
        ]
        > 0.9
    )
    assert fit.estimator.C == 0.5
    assert np.allclose(fit.estimator.coef_[1], 0, atol=1e-7)
    assert np.allclose(fit.estimator.coef_[0], -fit.estimator.coef_[2], atol=1e-7)
    assert fit.estimator.intercept_[0] == pytest.approx(
        fit.estimator.intercept_[2], abs=1e-7
    )
    swapped = [
        replace(
            m,
            team_a=m.team_b,
            team_b=m.team_a,
            players_a=m.players_b,
            players_b=m.players_a,
        )
        for m in matches[:50]
    ]
    assert np.allclose(
        fit.predict(matches[:50]), fit.predict(swapped)[:, ::-1], atol=1e-7
    )
    matrix = fit.vectorizer.transform(features(m, "player") for m in matches[:50])
    probabilities = fit.predict(matches[:50])
    assert np.allclose(
        np.log(probabilities[:, 2] / probabilities[:, 0]),
        matrix @ fit.coefficients,
        atol=1e-7,
    )


def test_no_result_and_all_draws_do_not_get_directional_rankings():
    matches, _ = simulated_matches(2)
    with pytest.raises(ValueError, match="decisive"):
        fit_current([replace(m, winner="", result="draw") for m in matches], 0.1)
    with pytest.raises(ValueError, match="unsupported"):
        fit_current([replace(matches[0], winner="", result="no result")], 0.1)
    assert outcome_class(replace(matches[0], winner="", result="tie")) == 1
    with pytest.raises(ValueError, match="separately"):
        fit_current([matches[0], replace(matches[1], gender="female")], 0.1)


def test_tuning_respects_completion_and_start_boundaries():
    matches, _ = simulated_matches(61)
    matches = [
        replace(m, start_date=date(2023, 2, 1), end_date=date(2023, 2, 1))
        for m in matches
    ]
    for i in range(40, 60):
        matches[i] = replace(
            matches[i], start_date=date(2024, 2, 1), end_date=date(2024, 2, 1)
        )
    matches[-1] = replace(
        matches[-1], start_date=date(2023, 12, 31), end_date=date(2024, 1, 2)
    )
    value, tuning = select_regularization(matches, date(2024, 12, 31))
    assert len(tuning["training_ids"]) == 40
    assert len(tuning["validation_ids"]) == 20
    assert tuning["boundary_crossing_ids"] == ["60"]
    assert (
        value
        == min(tuning["candidates"], key=lambda r: r["validation_log_loss"])[
            "regularization_c"
        ]
    )
    assert tuning["status"] == "tuned"
    value, tuning = select_regularization(matches[:5], date(2024, 12, 31))
    assert (value, tuning["status"]) == (0.1, "untuned_sparse_data")
    assert years_before(date(2024, 2, 29), 1) == date(2023, 2, 28)


def test_disconnected_groups_and_shared_player_bridge():
    matches, _ = simulated_matches(2)
    first = matches[0]
    second = replace(
        matches[1],
        players_a=tuple("x" + p for p in matches[1].players_a),
        players_b=tuple("x" + p for p in matches[1].players_b),
    )
    assert len(set(comparison_groups([first, second]).values())) == 2
    second = replace(second, players_a=(first.players_a[0],) + second.players_a[1:])
    assert len(set(comparison_groups([first, second]).values())) == 1


def test_identical_teammates_receive_tied_ranks():
    matches, _ = simulated_matches(10)
    fixed = [
        replace(m, players_a=matches[0].players_a, players_b=matches[0].players_b)
        for m in matches
    ]
    fit = fit_current(fixed, 0.1)
    rows = player_rows(
        fixed,
        fit,
        "ODI",
        date(2024, 1, 1),
        date(2019, 1, 2),
        0.1,
        "untuned_sparse_data",
    )
    assert len({r["rank"] for r in rows}) <= 2
    assert all(r["identical_pattern_group"] for r in rows)
    assert all(r["training_appearances"] == 10 for r in rows)


def test_preparation_separates_formats_clubs_and_test_draws(match, inputs, tmp_path):
    for index, (fmt, team_type, result) in enumerate(
        [
            ("T20", "club", "win"),
            ("T20", "international", "win"),
            ("Test", "international", "draw"),
            ("Test", "international", "no result"),
            ("IT20", "international", "win"),
            ("T20", None, "win"),
        ],
        1,
    ):
        doc = deepcopy(match)
        doc["info"].update(match_type=fmt)
        if team_type is None:
            doc["info"].pop("team_type", None)
        else:
            doc["info"]["team_type"] = team_type
        if result != "win":
            doc["info"]["outcome"] = {"result": result}
        (inputs / f"{index}.json").write_text(json.dumps(doc))
    prepare(inputs, tmp_path / "t20", "2024-12-31", "T20I")
    assert {m.match_id for m in load_prepared(tmp_path / "t20")[0]} == {"2", "5"}
    prepare(inputs, tmp_path / "test", "2024-12-31", "Test")
    assert load_prepared(tmp_path / "test")[0] == []
    assert [
        m.match_id for m in load_prepared(tmp_path / "test", include_draws=True)[0]
    ] == ["3"]


def test_rank_publication_is_complete_separate_and_verified(match, tmp_path):
    for fmt, folder in FORMATS.items():
        raw = tmp_path / "raw" / folder
        raw.mkdir(parents=True)
        for i, (gender, result, when) in enumerate(
            [
                ("male", "win", "2024-01-01"),
                ("male", "draw", "2024-02-01"),
                ("female", "win", "2024-03-01"),
                ("male", "win", "2019-12-31"),
            ],
            1,
        ):
            doc = deepcopy(match)
            doc["info"].update(
                match_type="T20" if fmt == "T20I" else fmt,
                team_type="international",
                gender=gender,
                dates=[when],
            )
            if result == "draw":
                doc["info"]["outcome"] = {"result": "draw"}
            (raw / f"{i}.json").write_text(json.dumps(doc))
        prepare(raw, tmp_path / "build" / folder, "2024-12-31", fmt)
    output = tmp_path / "release"
    manifest = rank_formats(tmp_path / "build", output, "2024-12-31")
    assert len(manifest["cohorts"]) == 6
    assert manifest == verify_outputs(output)
    rows = read_csv(output / "rankings.csv")
    assert len(rows) == 22 * 6
    assert {r["format"] for r in rows} == set(FORMATS)
    assert all("4" not in c["match_ids"] for c in manifest["cohorts"])
    assert manifest["cohorts"][0]["draws"] == 1
    assert all(c["draws"] == 0 for c in manifest["cohorts"][2:])
    assert "untuned, sparse-data" in (output / "report.md").read_text()
    content = tmp_path / "site-content"
    assert prepare_site_content(output, content) == 4
    assert "## Test — Men" in (content / "test.md").read_text()
    assert "## ODI" not in (content / "test.md").read_text()
    assert "## T20I — Women" in (content / "t20i.md").read_text()
    assert "odis_female.csv" in (content / "coverage.md").read_text()
    assert manifest == verify_outputs(output)
    with pytest.raises(ValueError, match="contain"):
        prepare_site_content(output, tmp_path)
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="cutoff"):
        rank_formats(tmp_path / "build", output, "2025-01-01")
    assert original == (output / "manifest.json").read_bytes()
    before = (tmp_path / "build" / "tests" / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="contain"):
        rank_formats(tmp_path / "build", tmp_path, "2024-12-31")
    assert before == (tmp_path / "build" / "tests" / "manifest.json").read_bytes()
    (output / "rankings.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_outputs(output)
    old_page = (content / "test.md").read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        prepare_site_content(output, content)
    assert (content / "test.md").read_bytes() == old_page


def test_since_records_old_exclusions_but_rejects_in_window_identity_errors(
    match, inputs, tmp_path
):
    older = deepcopy(match)
    older["info"]["dates"] = ["2020-01-01"]
    older["info"]["players"]["Beta"][0] = "Alpha 0"
    (inputs / "1.json").write_text(json.dumps(older))
    (inputs / "2.json").write_text(json.dumps(match))
    output = tmp_path / "prepared"
    manifest = prepare(inputs, output, "2024-12-31", since="2021-01-01")
    assert manifest["excluded"] == {"before_window": 1}
    assert manifest["sources"][0]["status"] == "before_window"
    assert [m.match_id for m in load_prepared(output)[0]] == ["2"]
    older["info"]["dates"] = ["2023-01-01"]
    (inputs / "1.json").write_text(json.dumps(older))
    with pytest.raises(ValueError, match="Duplicate participant"):
        prepare(inputs, output, "2024-12-31", since="2021-01-01")
