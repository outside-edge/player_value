from copy import deepcopy

import pytest

from player_value.rankings import rank_players, ranking_report


def coefficient(player, score, **changes):
    row = {
        "feature_kind": "player",
        "entity": player,
        "player_name": player,
        "coefficient_log_odds": str(score),
        "gender": "male",
        "model": "combined",
        "window_years": "5",
        "evaluation_year": "2025",
        "training_appearances": "30",
        "regularization_c": "0.1",
        "max_teammate_share": "0.9",
        "identical_pattern_group": "",
    }
    row.update(changes)
    return row


def test_rankings_are_within_fit_and_keep_low_exposure_players():
    source = [
        coefficient("a", 0.1),
        coefficient("b", 0.3, training_appearances="3"),
        coefficient("a", 0.7, model="player"),
        coefficient("c", 0.4, gender="female"),
        coefficient("d", 0.8, evaluation_year="2024"),
        coefficient("Alpha", 5, feature_kind="team"),
    ]
    original = deepcopy(source)
    rows = rank_players(source)
    selected = [
        r
        for r in rows
        if r["gender"] == "male"
        and r["model"] == "combined"
        and r["fit_cutoff"] == "2025-01-01"
    ]
    assert [(r["player_id"], r["rank"]) for r in selected] == [("b", 1), ("a", 2)]
    assert selected[0]["fewer_than_20_appearances"] == 1
    assert selected[0]["training_start"] == "2020-01-01"
    assert selected[0]["training_end"] == "2024-12-31"
    assert len(rows) == 5
    assert source == original


def test_equal_effects_share_competition_rank():
    rows = rank_players(
        [
            coefficient("b", 0.3, identical_pattern_group="a;b"),
            coefficient("a", 0.3 + 1e-14, identical_pattern_group="a;b"),
            coefficient("c", 0.2),
        ]
    )
    assert [(r["player_id"], r["rank"]) for r in rows] == [("a", 1), ("b", 1), ("c", 3)]
    assert rows[0]["identical_pattern_group"] == "a;b"


def test_duplicate_or_nonfinite_effects_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        rank_players([coefficient("a", 0.1), coefficient("a", 0.2)])
    with pytest.raises(ValueError, match="nonfinite"):
        rank_players([coefficient("a", "nan")])


def test_latest_report_uses_combined_five_year_fit():
    rows = rank_players(
        [
            coefficient("Latest", 0.2),
            coefficient("Older", 10, evaluation_year="2024"),
            coefficient("Other model", 10, model="player"),
            coefficient("Other window", 10, window_years="3"),
        ]
    )
    report = ranking_report(rows)
    assert "| Latest |" in report
    assert "| Older |" not in report
    assert "| Other model |" not in report
    assert "| Other window |" not in report
