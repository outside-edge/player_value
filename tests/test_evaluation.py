import csv
import json
from copy import deepcopy

import pytest

from player_value.evaluation import compare
from player_value.io import sha256
from player_value.lineup import load_prepared
from player_value.preparation import prepare
from player_value.publication import publish_comparison, verify_outputs


@pytest.fixture
def prepared_experiment(match, inputs, tmp_path):
    identifier = 1
    for gender in ("male", "female"):
        for year in range(2020, 2025):
            for month in (2, 5, 8, 11):
                row = deepcopy(match)
                row["info"].update(gender=gender, dates=[f"{year}-{month:02d}-01"])
                row["info"]["outcome"]["winner"] = "Alpha" if month % 2 else "Beta"
                (inputs / f"{identifier}.json").write_text(json.dumps(row))
                identifier += 1
    output = tmp_path / "prepared"
    prepare(inputs, output, "2024-12-31")
    return output


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_experiment_has_paired_predictions_cutoffs_and_provenance(
    prepared_experiment, tmp_path
):
    output = tmp_path / "analysis"
    manifest = compare(
        prepared_experiment,
        output,
        years=[2023, 2024],
        windows=[3],
        c_grid=[0.01, 1],
        replicates=100,
    )
    predictions = rows(output / "predictions.csv")
    assert len(predictions) == 2 * 2 * 4 * 4
    for split in manifest["splits"]:
        assert not set(split["training_ids"]) & set(split["validation_ids"])
        assert not set(split["refit_ids"]) & set(split["evaluation_ids"])
    assert manifest["preparation_manifest_sha256"] == sha256(
        prepared_experiment / "manifest.json"
    )
    for name, checksum in manifest["outputs"].items():
        assert sha256(output / name) == checksum
    assert {row["quarter_blocks"] for row in rows(output / "comparisons.csv")} == {"8"}
    paired = rows(output / "paired_losses.csv")
    assert len(paired) == 2 * 2 * 4 * 2


def test_future_names_and_debuts_do_not_enter_training(
    prepared_experiment, inputs, tmp_path
):
    for path in inputs.glob("*.json"):
        match = json.loads(path.read_text())
        if match["info"]["dates"][0].startswith("2024"):
            match["info"]["players"]["Alpha"][0] = "Future Label"
            match["info"]["registry"]["people"]["Future Label"] = "alpha00"
            match["info"]["players"]["Alpha"][1] = "Debut"
            match["info"]["registry"]["people"]["Debut"] = "future00"
            path.write_text(json.dumps(match))
    prepare(inputs, prepared_experiment, "2024-12-31")
    output = tmp_path / "analysis"
    compare(
        prepared_experiment,
        output,
        years=[2024],
        windows=[3],
        c_grid=[0.1],
        replicates=100,
    )
    coefficients = rows(output / "coefficients.csv")
    assert all(
        row["player_name"] not in {"Future Label", "Debut"} for row in coefficients
    )
    assert all(row["entity"] != "future00" for row in coefficients)
    forecasts = rows(output / "predictions.csv")
    assert all(
        row["unseen_players"] == "1" for row in forecasts if row["model"] != "coin"
    )


def test_modified_prepared_tables_fail_checksums(prepared_experiment):
    with (prepared_experiment / "matches.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_prepared(prepared_experiment)


def test_incomplete_year_and_output_overlap_fail(prepared_experiment, tmp_path):
    with pytest.raises(ValueError, match="does not cover"):
        compare(prepared_experiment, tmp_path / "analysis", years=[2025])
    with pytest.raises(ValueError, match="must not contain"):
        compare(prepared_experiment, tmp_path, years=[2024])


def test_holdout_outcomes_do_not_change_fits_or_tuning(
    prepared_experiment, inputs, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    compare(
        prepared_experiment,
        first,
        years=[2024],
        windows=[3],
        c_grid=[0.01, 1],
        replicates=100,
    )
    for path in inputs.glob("*.json"):
        match = json.loads(path.read_text())
        if match["info"]["dates"][0].startswith("2024"):
            match["info"]["outcome"]["winner"] = "Alpha"
            path.write_text(json.dumps(match))
    prepare(inputs, prepared_experiment, "2024-12-31")
    compare(
        prepared_experiment,
        second,
        years=[2024],
        windows=[3],
        c_grid=[0.01, 1],
        replicates=100,
    )
    for name in ["tuning.csv", "coefficients.csv"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert [r["probability_a"] for r in rows(first / "predictions.csv")] == [
        r["probability_a"] for r in rows(second / "predictions.csv")
    ]


def test_publication_copies_verified_results_and_detects_tampering(
    prepared_experiment, tmp_path
):
    analysis, release = tmp_path / "analysis", tmp_path / "release"
    compare(
        prepared_experiment,
        analysis,
        years=[2024],
        windows=[3],
        c_grid=[0.1],
        replicates=100,
    )
    publish_comparison(analysis, release)
    ranked = rows(release / "rankings.csv")
    assert len(ranked) == sum(
        r["feature_kind"] == "player" for r in rows(analysis / "coefficients.csv")
    )
    assert (analysis / "coefficients.csv").read_bytes() == (
        release / "coefficients.csv"
    ).read_bytes()
    assert (analysis / "predictions.csv").read_bytes() == (
        release / "predictions.csv"
    ).read_bytes()
    assert verify_outputs(release)["analysis_manifest_sha256"] == sha256(
        analysis / "manifest.json"
    )
    before = (release / "report.md").read_bytes()
    (analysis / "metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        publish_comparison(analysis, release)
    assert (release / "report.md").read_bytes() == before
    (release / "report.md").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_outputs(release)


def test_cross_year_match_enters_later_refit_but_not_earlier_forecast(
    prepared_experiment, inputs, match, tmp_path
):
    for gender, identifier in [("male", "1001"), ("female", "1002")]:
        match["info"].update(gender=gender, dates=["2022-12-31", "2023-01-01"])
        (inputs / f"{identifier}.json").write_text(json.dumps(match))
    prepare(inputs, prepared_experiment, "2024-12-31")
    manifest = compare(
        prepared_experiment,
        tmp_path / "analysis",
        years=[2023, 2024],
        windows=[3],
        c_grid=[0.1],
        replicates=100,
    )
    for split in manifest["splits"]:
        identifier = "1001" if split["gender"] == "male" else "1002"
        assert (
            identifier
            not in split["training_ids"]
            + split["validation_ids"]
            + split["evaluation_ids"]
        )
        assert (identifier in split["refit_ids"]) == (split["evaluation_year"] == 2024)
