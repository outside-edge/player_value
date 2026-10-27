"""Build ODI match and participant tables without treating substitutes as starters."""

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from player_value.io import sha256, staged_directory, write_json

MATCH_COLUMNS = [
    "match_id",
    "start_date",
    "end_date",
    "gender",
    "match_type",
    "team_type",
    "team_a",
    "team_b",
    "venue",
    "city",
    "toss_winner",
    "toss_decision",
    "outcome",
    "winner",
    "outcome_method",
    "eliminator",
    "bowl_out",
    "balls_per_over",
    "scheduled_overs",
    "delivery_count",
    "has_match_replacements",
    "has_fixed_xi",
    "lineup_eligible",
    "lineup_exclusion",
    "source_file",
]
PARTICIPANT_COLUMNS = ["match_id", "team", "player_id", "player_name"]
PLAYER_COLUMNS = ["player_id", "player_name", "first_date", "last_date"]


def load_match(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def iter_deliveries(match):
    for innings in match.get("innings", []):
        if "team" in innings:
            for over in innings["overs"]:
                yield from over["deliveries"]
        else:
            for details in innings.values():
                for delivery in details.get("deliveries", []):
                    yield from delivery.values()


def parse_match(match, match_id, source_file):
    info = match["info"]
    gender = info["gender"]
    if gender not in {"male", "female"}:
        raise ValueError(f"Unknown competition gender: {gender}")
    teams = info["teams"]
    if len(teams) != 2 or len(set(teams)) != 2:
        raise ValueError("Expected two distinct teams")
    teams = sorted(teams)
    dates = sorted(date.fromisoformat(str(value)) for value in info["dates"])
    if not dates:
        raise ValueError("Missing match dates")
    outcome = info.get("outcome", {})
    winner = outcome.get("winner", "")
    result = outcome.get("result", "")
    if winner and winner not in teams:
        raise ValueError("Winner is not a participating team")
    if result and result not in {"tie", "no result", "draw"}:
        raise ValueError(f"Unknown outcome: {result}")
    if winner and result:
        raise ValueError("Outcome has both a winner and a non-winning result")
    if not winner and not result:
        raise ValueError("Outcome has neither a winner nor an explicit result")
    toss = info.get("toss", {})
    if toss.get("winner") and toss["winner"] not in teams:
        raise ValueError("Toss winner is not a participating team")
    people = info.get("registry", {}).get("people", {})
    participants = []
    seen = set()
    rosters = info.get("players", {})
    if set(rosters) - set(teams):
        raise ValueError("Participant list contains an unknown team")
    for team in teams:
        for name in rosters.get(team, []):
            player_id = people.get(name)
            if not isinstance(player_id, str) or not player_id:
                raise ValueError(f"Missing stable player ID: {name}")
            if player_id in seen:
                raise ValueError(f"Duplicate participant ID: {player_id}")
            seen.add(player_id)
            participants.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "player_id": player_id,
                    "player_name": name,
                }
            )
    deliveries = list(iter_deliveries(match))
    replacements = any(d.get("replacements", {}).get("match") for d in deliveries)
    replacements = replacements or bool(info.get("supersubs"))
    fixed_xi = all(len(rosters.get(team, [])) == 11 for team in teams)
    fixed_xi = fixed_xi and not replacements
    exclusions = []
    if not winner:
        exclusions.append("non_decisive")
    if not all(len(rosters.get(team, [])) == 11 for team in teams):
        exclusions.append("participant_count")
    if replacements:
        exclusions.append("match_replacements")
    row = {
        "match_id": match_id,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "gender": gender,
        "match_type": info["match_type"],
        "team_type": info.get("team_type", ""),
        "team_a": teams[0],
        "team_b": teams[1],
        "venue": info.get("venue", ""),
        "city": info.get("city", ""),
        "toss_winner": toss.get("winner", ""),
        "toss_decision": toss.get("decision", ""),
        "outcome": "win" if winner else result,
        "winner": winner,
        "outcome_method": outcome.get("method", ""),
        "eliminator": outcome.get("eliminator", ""),
        "bowl_out": outcome.get("bowl_out", ""),
        "balls_per_over": info.get("balls_per_over", ""),
        "scheduled_overs": info.get("overs", ""),
        "delivery_count": len(deliveries),
        "has_match_replacements": int(replacements),
        "has_fixed_xi": int(fixed_xi),
        "lineup_eligible": int(not exclusions),
        "lineup_exclusion": ";".join(exclusions),
        "source_file": source_file,
    }
    return row, participants


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def prepare(input_dir, output_dir, as_of, match_format="ODI", since=None):
    formats = {"Test": {"Test"}, "ODI": {"ODI"}, "T20I": {"T20", "IT20"}}
    if match_format not in formats:
        raise ValueError(f"Unsupported format: {match_format}")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    as_of = date.fromisoformat(str(as_of))
    since = date.fromisoformat(str(since)) if since is not None else None
    if since is not None and since > as_of:
        raise ValueError("Preparation start follows cutoff")
    source_root, target_root = input_dir.resolve(), output_dir.resolve()
    if source_root == target_root or source_root.is_relative_to(target_root):
        raise ValueError("Output directory must not contain the input directory")
    files = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix in {".json", ".yaml", ".yml"} and p.stem.isdecimal()
    )
    if not files:
        raise ValueError(f"No numeric Cricsheet match files in {input_dir}")
    matches, participants, sources = [], [], []
    seen = set()
    excluded = Counter()
    for path in files:
        if path.stem in seen:
            raise ValueError(f"Duplicate match ID: {path.stem}")
        seen.add(path.stem)
        try:
            document = load_match(path)
            checksum = sha256(path)
            if document["info"]["match_type"] not in formats[match_format]:
                reason = "other_format"
            elif document["info"].get("team_type", "international") != "international":
                reason = "not_international"
            elif (
                match_format == "T20I"
                and document["info"].get("team_type") != "international"
            ):
                reason = "unknown_team_type"
            elif (
                since is not None
                and min(date.fromisoformat(str(d)) for d in document["info"]["dates"])
                < since
            ):
                reason = "before_window"
            elif (
                max(date.fromisoformat(str(d)) for d in document["info"]["dates"])
                > as_of
            ):
                reason = "after_cutoff"
            else:
                reason = "included"
                row, roster = parse_match(document, path.stem, path.name)
                matches.append(row)
                participants.extend(roster)
            sources.append({"file": path.name, "sha256": checksum, "status": reason})
            if reason != "included":
                excluded[reason] += 1
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"{path.name}: {error}") from error
    if not matches:
        raise ValueError(f"No {match_format} matches at or before the requested cutoff")
    matches.sort(key=lambda row: (row["end_date"], int(row["match_id"])))
    by_match = {row["match_id"]: row for row in matches}
    participants.sort(
        key=lambda row: (
            by_match[row["match_id"]]["end_date"],
            int(row["match_id"]),
            row["team"],
            row["player_id"],
        )
    )
    players = {}
    for row in participants:
        match_date = by_match[row["match_id"]]["end_date"]
        player = players.setdefault(
            row["player_id"],
            {
                "player_id": row["player_id"],
                "first_date": match_date,
            },
        )
        player.update(player_name=row["player_name"], last_date=match_date)
    summary = {
        gender: {
            "matches": sum(m["gender"] == gender for m in matches),
            "lineup_eligible": sum(
                m["lineup_eligible"] for m in matches if m["gender"] == gender
            ),
            "players": len(
                {
                    p["player_id"]
                    for p in participants
                    if by_match[p["match_id"]]["gender"] == gender
                }
            ),
        }
        for gender in ["male", "female"]
    }
    manifest = {
        "schema_version": 1,
        "format": match_format,
        "as_of": as_of.isoformat(),
        "since": since.isoformat() if since is not None else None,
        "source_directory": str(input_dir),
        "sources": sources,
        "excluded": dict(excluded),
        "summary": summary,
        "code_sha256": sha256(Path(__file__)),
        "source_receipt": (
            json.loads((input_dir / "receipt.json").read_text(encoding="utf-8"))
            if (input_dir / "receipt.json").exists()
            else None
        ),
    }
    with staged_directory(output_dir) as stage:
        write_csv(stage / "matches.csv", MATCH_COLUMNS, matches)
        write_csv(stage / "participants.csv", PARTICIPANT_COLUMNS, participants)
        write_csv(
            stage / "players.csv",
            PLAYER_COLUMNS,
            sorted(players.values(), key=lambda row: row["player_id"]),
        )
        manifest["outputs"] = {
            name: sha256(stage / name)
            for name in ["matches.csv", "participants.csv", "players.csv"]
        }
        write_json(stage / "manifest.json", manifest)
    return manifest
