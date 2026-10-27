"""Symmetric, regularized team and lineup win-probability models."""

import csv
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from player_value.io import sha256

MODELS = ("team", "player", "combined")


@dataclass(frozen=True)
class Match:
    match_id: str
    start_date: date
    end_date: date
    gender: str
    team_a: str
    team_b: str
    players_a: tuple
    players_b: tuple
    names: dict
    winner: str
    toss_winner: str = ""
    toss_decision: str = ""
    result: str = "win"

    @property
    def outcome(self):
        return int(self.winner == self.team_a)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_prepared(directory, include_draws=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("matches.csv", "participants.csv", "players.csv"):
        if manifest["outputs"].get(name) != sha256(directory / name):
            raise ValueError(f"Prepared output checksum mismatch: {name}")
    records = read_csv(directory / "matches.csv")
    by_id = {row["match_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate match IDs in prepared matches")
    rosters = defaultdict(list)
    seen = set()
    for player in read_csv(directory / "participants.csv"):
        match_id, team = player["match_id"], player["team"]
        if match_id not in by_id or team not in {
            by_id[match_id]["team_a"],
            by_id[match_id]["team_b"],
        }:
            raise ValueError("Participant does not join to a match and team")
        key = (match_id, player["player_id"])
        if key in seen or not player["player_id"] or not player["player_name"]:
            raise ValueError("Missing or duplicated participant identity")
        seen.add(key)
        rosters[match_id, team].append(player)
    matches = []
    for row in records:
        if row["lineup_eligible"] not in {"0", "1"}:
            raise ValueError("Invalid lineup eligibility flag")
        drawn = (
            include_draws
            and row["outcome"] in {"draw", "tie"}
            and row["has_fixed_xi"] == "1"
        )
        if row["lineup_eligible"] != "1" and not drawn:
            continue
        match_id = row["match_id"]
        sides = [rosters[match_id, row[key]] for key in ("team_a", "team_b")]
        if any(len(side) != 11 for side in sides):
            raise ValueError(f"{match_id}: eligible match does not have two XIs")
        if (
            (not drawn and row["winner"] not in {row["team_a"], row["team_b"]})
            or (drawn and row["winner"] != "")
            or row["team_a"] == row["team_b"]
            or (not drawn and row["outcome"] != "win")
            or row["has_match_replacements"] != "0"
        ):
            raise ValueError(f"{match_id}: inconsistent eligibility or outcome")
        end_date = date.fromisoformat(row["end_date"])
        start_date = date.fromisoformat(row["start_date"])
        if start_date > end_date:
            raise ValueError(f"{match_id}: start date exceeds completion date")
        if end_date > date.fromisoformat(manifest["as_of"]):
            raise ValueError(f"{match_id}: match exceeds preparation cutoff")
        if row["gender"] not in {"male", "female"}:
            raise ValueError(f"{match_id}: unknown gender")
        if row["toss_winner"] not in {"", row["team_a"], row["team_b"]}:
            raise ValueError(f"{match_id}: unknown toss winner")
        if row["toss_decision"] not in {"", "bat", "field"}:
            raise ValueError(f"{match_id}: unknown toss decision")
        matches.append(
            Match(
                match_id=match_id,
                start_date=start_date,
                end_date=end_date,
                gender=row["gender"],
                team_a=row["team_a"],
                team_b=row["team_b"],
                players_a=tuple(p["player_id"] for p in sides[0]),
                players_b=tuple(p["player_id"] for p in sides[1]),
                names={
                    p["player_id"]: p["player_name"] for side in sides for p in side
                },
                winner=row["winner"],
                toss_winner=row["toss_winner"],
                toss_decision=row["toss_decision"],
                result=row["outcome"],
            )
        )
    matches.sort(key=lambda m: (m.end_date, m.match_id))
    return matches, manifest


def features(match, model):
    if model not in MODELS:
        raise ValueError(f"Unknown lineup model: {model}")
    toss = (1 if match.toss_winner == match.team_a else -1) if match.toss_winner else 0
    batting = toss * {"bat": 1, "field": -1, "": 0}[match.toss_decision]
    result = {"context:toss_winner": toss, "context:batting_first": batting}
    if model in {"team", "combined"}:
        result.update({f"team:{match.team_a}": 1, f"team:{match.team_b}": -1})
    if model in {"player", "combined"}:
        result.update({f"player:{p}": 1 for p in match.players_a})
        result.update({f"player:{p}": -1 for p in match.players_b})
    return result


@dataclass
class LineupFit:
    model: str
    regularization_c: float
    vectorizer: DictVectorizer
    estimator: LogisticRegression
    known_players: set
    known_teams: set

    def predict(self, matches):
        matrix = self.vectorizer.transform(features(m, self.model) for m in matches)
        return self.estimator.predict_proba(matrix)[:, 1]


def fit_lineup(matches, model, regularization_c):
    if not matches or len({m.outcome for m in matches}) != 2:
        raise ValueError("Training requires matches with both outcome classes")
    if len({m.gender for m in matches}) != 1:
        raise ValueError("Fit men's and women's matches separately")
    if not np.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("Regularization C must be positive and finite")
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(features(m, model) for m in matches)
    estimator = LogisticRegression(
        C=regularization_c,
        l1_ratio=0,
        solver="lbfgs",
        fit_intercept=False,
        tol=1e-8,
        max_iter=2000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(matrix, [m.outcome for m in matches])
        except ConvergenceWarning as error:
            raise ValueError(f"{model}: optimization did not converge") from error
    return LineupFit(
        model,
        regularization_c,
        vectorizer,
        estimator,
        {p for m in matches for p in m.players_a + m.players_b},
        {t for m in matches for t in (m.team_a, m.team_b)},
    )


def lineup_diagnostics(matches):
    appearances = Counter()
    shared = Counter()
    patterns = defaultdict(list)
    names = {}
    parent = {}

    def find(team):
        parent.setdefault(team, team)
        while parent[team] != team:
            team = parent[team]
        return team

    for index, match in enumerate(matches):
        parent[find(match.team_a)] = find(match.team_b)
        names.update(match.names)
        for sign, side in ((1, match.players_a), (-1, match.players_b)):
            appearances.update(side)
            shared.update(combinations(sorted(side), 2))
            for player in side:
                patterns[player].append((index, sign))
    twins = defaultdict(list)
    for player, pattern in patterns.items():
        twins[tuple(pattern)].append(player)
    group_for = {
        player: ";".join(sorted(group))
        for group in twins.values()
        if len(group) > 1
        for player in group
    }
    max_shared = Counter()
    for (first, second), count in shared.items():
        max_shared[first] = max(max_shared[first], count)
        max_shared[second] = max(max_shared[second], count)
    players = {
        player: {
            "player_name": names[player],
            "training_appearances": count,
            "max_teammate_share": max_shared[player] / count,
            "identical_pattern_group": group_for.get(player, ""),
        }
        for player, count in appearances.items()
    }
    summary = {
        "training_players": len(players),
        "players_in_identical_groups": len(group_for),
        "identical_groups": sum(len(group) > 1 for group in twins.values()),
        "opponent_components": len({find(team) for team in parent}),
    }
    return players, summary
