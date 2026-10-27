import warnings
from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from player_value.evaluation import losses, paired_interval, split_matches, tune
from player_value.lineup import Match, features, fit_lineup, lineup_diagnostics


def simulated_matches(count=800, seed=9):
    rng = np.random.default_rng(seed)
    ids = [f"p{i:02d}" for i in range(36)]
    strength = np.linspace(-0.3, 0.3, len(ids))
    matches = []
    for i in range(count):
        order = rng.permutation(len(ids))
        a, b = order[:11], order[11:22]
        probability = 1 / (1 + np.exp(-(strength[a].sum() - strength[b].sum())))
        matches.append(
            Match(
                match_id=str(i),
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 1),
                gender="male",
                team_a="Alpha",
                team_b="Beta",
                players_a=tuple(ids[j] for j in a),
                players_b=tuple(ids[j] for j in b),
                names={player: player for player in ids},
                winner="Alpha" if rng.random() < probability else "Beta",
                toss_winner="Alpha" if i % 2 else "Beta",
                toss_decision="bat",
            )
        )
    return matches, dict(zip(ids, strength, strict=True))


@pytest.mark.parametrize("model", ["team", "player", "combined"])
def test_swapping_sides_complements_probability(model):
    matches, _ = simulated_matches(80)
    fit = fit_lineup(matches, model, 0.1)
    swapped = [
        replace(
            m,
            team_a=m.team_b,
            team_b=m.team_a,
            players_a=m.players_b,
            players_b=m.players_a,
        )
        for m in matches
    ]
    assert np.allclose(fit.predict(matches) + fit.predict(swapped), 1, atol=1e-12)


def test_simulated_player_effects_recovered_with_lineup_variation():
    matches, truth = simulated_matches(2200)
    fit = fit_lineup(matches, "player", 1)
    estimates = dict(
        zip(fit.vectorizer.get_feature_names_out(), fit.estimator.coef_[0], strict=True)
    )
    fitted = [estimates[f"player:{p}"] for p in truth]
    assert np.corrcoef(list(truth.values()), fitted)[0, 1] > 0.9
    assert estimates["player:p35"] > estimates["player:p00"]


def test_unseen_players_are_zero_and_never_expand_vocabulary():
    matches, _ = simulated_matches(80)
    fit = fit_lineup(matches, "player", 1)
    original = dict(fit.vectorizer.vocabulary_)
    future = replace(matches[0], players_a=("new_player",) + matches[0].players_a[1:])
    prediction = fit.predict([future])[0]
    assert 0 < prediction < 1
    assert fit.vectorizer.vocabulary_ == original
    assert "new_player" not in fit.known_players


def test_identical_teammates_exposed_and_not_individually_recovered():
    matches, _ = simulated_matches(60)
    fixed = [
        replace(
            m,
            players_a=tuple(f"a{i}" for i in range(11)),
            players_b=tuple(f"b{i}" for i in range(11)),
            names={f"{t}{i}": f"{t}{i}" for t in "ab" for i in range(11)},
        )
        for m in matches
    ]
    players, summary = lineup_diagnostics(fixed)
    assert summary["identical_groups"] == 2
    assert summary["players_in_identical_groups"] == 22
    assert summary["opponent_components"] == 1
    assert players["a0"]["max_teammate_share"] == 1
    fit = fit_lineup(fixed, "player", 1)
    coefs = dict(
        zip(fit.vectorizer.get_feature_names_out(), fit.estimator.coef_[0], strict=True)
    )
    assert coefs["player:a0"] == pytest.approx(coefs["player:a1"])


def test_toss_decision_uses_only_toss_fields():
    matches, _ = simulated_matches(2)
    match = replace(matches[0], toss_winner="Alpha", toss_decision="field")
    assert features(match, "team")["context:batting_first"] == -1
    assert (
        features(replace(match, toss_winner="", toss_decision=""), "team")[
            "context:batting_first"
        ]
        == 0
    )


def test_completed_outcomes_and_start_dates_control_boundaries():
    matches, _ = simulated_matches(6)
    dates = [
        ("2020-06-01", "2020-06-01"),
        ("2021-12-31", "2022-01-01"),
        ("2022-06-01", "2022-06-01"),
        ("2022-12-31", "2023-01-01"),
        ("2023-06-01", "2023-06-01"),
        ("2024-01-01", "2024-01-01"),
    ]
    matches = [
        replace(
            m, start_date=date.fromisoformat(start), end_date=date.fromisoformat(end)
        )
        for m, (start, end) in zip(matches, dates, strict=True)
    ]
    training, validation, testing = split_matches(matches, "male", 2023, 3)
    assert [m.match_id for m in training] == ["0"]
    assert [m.match_id for m in validation] == ["2"]
    assert [m.match_id for m in testing] == ["4"]


def test_tuning_uses_validation_and_reports_all_candidates():
    matches, _ = simulated_matches(120)
    best, scores = tune(matches[:80], matches[80:], "player", [0.01, 1])
    assert len(scores) == 2
    assert (
        best == min(scores, key=lambda s: s["validation_log_loss"])["regularization_c"]
    )


def test_loss_values_and_probability_validation():
    score = losses([0, 1], [0.5, 0.5])
    assert np.allclose(score["log_loss"], np.log(2))
    assert np.allclose(score["brier"], 0.25)
    with pytest.raises(ValueError, match="finite probabilities"):
        losses([1], [np.nan])


def test_cluster_bootstrap_weights_by_matches_and_is_deterministic():
    differences, blocks = [0.1, 0.1, -0.2], ["Q1", "Q1", "Q2"]
    first = paired_interval(differences, blocks, replicates=500, seed=7)
    assert first == paired_interval(differences, blocks, replicates=500, seed=7)
    assert first["difference"] == pytest.approx(0)
    assert first["quarter_blocks"] == 2
    assert first["lower"] == pytest.approx(-0.2)
    assert first["upper"] == pytest.approx(0.1)


def test_gender_pooling_and_single_class_training_fail():
    matches, _ = simulated_matches(40)
    with pytest.raises(ValueError, match="separately"):
        fit_lineup(matches + [replace(matches[0], gender="female")], "team", 1)
    with pytest.raises(ValueError, match="both outcome classes"):
        fit_lineup([replace(m, winner="Alpha") for m in matches], "team", 1)


def test_nonconvergence_fails_instead_of_emitting_coefficients(monkeypatch):
    matches, _ = simulated_matches(40)

    def fail(*args, **kwargs):
        warnings.warn("not converged", ConvergenceWarning)

    monkeypatch.setattr("player_value.lineup.LogisticRegression.fit", fail)
    with pytest.raises(ValueError, match="did not converge"):
        fit_lineup(matches, "team", 1)
