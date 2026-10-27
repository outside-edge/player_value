import csv
import json
from copy import deepcopy

import pytest
import yaml

from player_value.io import sha256
from player_value.preparation import parse_match, prepare


def save(inputs, match, match_id=1):
    path = inputs / f"{match_id}.json"
    path.write_text(json.dumps(match))
    return path


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_tables_preserve_ids_outcomes_and_gender(match, inputs, tmp_path):
    save(inputs, match)
    women = deepcopy(match)
    women["info"].update(gender="female", dates=["2024-02-01"])
    save(inputs, women, 2)
    future = deepcopy(match)
    future["info"]["dates"] = ["2026-01-01"]
    save(inputs, future, 3)
    t20 = deepcopy(match)
    t20["info"]["match_type"] = "T20"
    save(inputs, t20, 4)
    output = tmp_path / "build"
    manifest = prepare(inputs, output, "2024-12-31")
    rows = read_rows(output / "matches.csv")
    assert [row["match_id"] for row in rows] == ["1", "2"]
    assert rows[0]["winner"] == "Alpha"
    assert rows[0]["lineup_eligible"] == "1"
    assert rows[0]["delivery_count"] == "2"
    assert len(read_rows(output / "participants.csv")) == 44
    assert manifest["summary"]["female"]["matches"] == 1
    assert manifest["excluded"] == {"after_cutoff": 1, "other_format": 1}
    assert manifest["source_receipt"] is None
    for name, checksum in manifest["outputs"].items():
        assert sha256(output / name) == checksum


def test_name_changes_join_by_id(match, inputs, tmp_path):
    save(inputs, match)
    match["info"]["dates"] = ["2024-02-01"]
    match["info"]["players"]["Alpha"][0] = "Renamed Player"
    match["info"]["registry"]["people"]["Renamed Player"] = "alpha00"
    save(inputs, match, 2)
    prepare(inputs, tmp_path / "build", "2024-12-31")
    rows = read_rows(tmp_path / "build" / "players.csv")
    player = next(row for row in rows if row["player_id"] == "alpha00")
    assert len(rows) == 22
    assert player == {
        "player_id": "alpha00",
        "player_name": "Renamed Player",
        "first_date": "2024-01-01",
        "last_date": "2024-02-01",
    }


@pytest.mark.parametrize("result", ["tie", "no result", "draw"])
def test_non_decisive_outcomes_are_retained(match, result):
    match["info"]["outcome"] = {"result": result}
    if result == "tie":
        match["info"]["outcome"]["eliminator"] = "Beta"
    row, _ = parse_match(match, "1", "1.json")
    assert row["outcome"] == result
    assert row["winner"] == ""
    assert row["lineup_eligible"] == 0
    assert row["has_fixed_xi"] == 1
    assert row["eliminator"] == ("Beta" if result == "tie" else "")


def test_substitutes_are_not_treated_as_starters(match):
    match["info"]["players"]["Alpha"].append("Replacement")
    match["info"]["registry"]["people"]["Replacement"] = "replace1"
    delivery = match["innings"][0]["overs"][0]["deliveries"][0]
    delivery["replacements"] = {
        "match": [
            {
                "in": "Replacement",
                "out": "Alpha 0",
                "team": "Alpha",
                "reason": "concussion_substitute",
            }
        ]
    }
    row, participants = parse_match(match, "1", "1.json")
    assert len(participants) == 23
    assert row["has_match_replacements"] == 1
    assert row["has_fixed_xi"] == row["lineup_eligible"] == 0
    assert "match_replacements" in row["lineup_exclusion"]


def test_supersub_flag_excludes_apparently_eleven_person_roster(match):
    match["info"]["supersubs"] = {"Alpha": "Alpha 0"}
    row, _ = parse_match(match, "1", "1.json")
    assert row["has_fixed_xi"] == 0


def test_missing_roster_remains_visible(match):
    del match["info"]["players"]
    row, participants = parse_match(match, "1", "1.json")
    assert row["lineup_eligible"] == 0
    assert row["lineup_exclusion"] == "participant_count"
    assert not participants


@pytest.mark.parametrize("defect", ["id", "duplicate", "winner", "outcome", "gender"])
def test_invalid_matches_fail_with_source_context(match, inputs, tmp_path, defect):
    if defect == "id":
        del match["info"]["registry"]["people"]["Alpha 0"]
    elif defect == "duplicate":
        match["info"]["players"]["Beta"][0] = "Alpha 0"
    elif defect == "winner":
        match["info"]["outcome"]["winner"] = "Unknown"
    elif defect == "outcome":
        match["info"]["outcome"] = {}
    else:
        match["info"]["gender"] = "unknown"
    save(inputs, match)
    with pytest.raises(ValueError, match="1.json"):
        prepare(inputs, tmp_path / "build", "2024-12-31")


def test_yaml_delivery_order_and_ids(match, inputs, tmp_path):
    deliveries = match["innings"][0]["overs"][0]["deliveries"]
    match["innings"] = [
        {
            "1st innings": {
                "team": "Alpha",
                "deliveries": [{0.1: deliveries[0]}, {0.2: deliveries[1]}],
            }
        }
    ]
    (inputs / "1.yaml").write_text(yaml.safe_dump(match))
    prepare(inputs, tmp_path / "build", "2024-12-31")
    rows = read_rows(tmp_path / "build" / "matches.csv")
    assert rows[0]["delivery_count"] == "2"
    assert len(read_rows(tmp_path / "build" / "participants.csv")) == 22


def test_duplicate_files_fail_before_replacing_output(match, inputs, tmp_path):
    save(inputs, match)
    output = tmp_path / "build"
    prepare(inputs, output, "2024-12-31")
    original = (output / "manifest.json").read_bytes()
    (inputs / "1.yaml").write_text(yaml.safe_dump(match))
    with pytest.raises(ValueError, match="Duplicate match ID"):
        prepare(inputs, output, "2024-12-31")
    assert (output / "manifest.json").read_bytes() == original


def test_cutoff_uses_last_match_date(match, inputs, tmp_path):
    match["info"]["dates"] = ["2024-01-01", "2024-01-02"]
    save(inputs, match)
    with pytest.raises(ValueError, match="No ODI matches"):
        prepare(inputs, tmp_path / "build", "2024-01-01")


def test_rebuild_is_deterministic(match, inputs, tmp_path):
    save(inputs, match)
    output = tmp_path / "build"
    prepare(inputs, output, "2024-12-31")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    prepare(inputs, output, "2024-12-31")
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_output_cannot_replace_input_parent(match, inputs, tmp_path):
    path = save(inputs, match)
    with pytest.raises(ValueError, match="must not contain"):
        prepare(inputs, tmp_path, "2024-12-31")
    assert path.exists()
