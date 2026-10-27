"""Publish verified saved comparisons without fitting models during rendering."""

import json
import shutil
from pathlib import Path

from player_value.io import sha256, staged_directory, write_json
from player_value.lineup import read_csv
from player_value.preparation import write_csv
from player_value.rankings import rank_players, ranking_report

MODEL_LABELS = {
    "coin": "50:50 reference",
    "team": "Team only",
    "player": "Players only",
    "combined": "Teams + players",
}
GENDER_LABELS = {"male": "Men", "female": "Women"}


def verify_outputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    outputs = manifest.get("outputs", {})
    if not outputs:
        raise ValueError("Manifest contains no output checksums")
    for name, checksum in outputs.items():
        if Path(name).name != name:
            raise ValueError("Manifest output paths must be local filenames")
        if sha256(directory / name) != checksum:
            raise ValueError(f"Output checksum mismatch: {name}")
    return manifest


def comparison_report(directory, manifest):
    directory = Path(directory)
    metrics = read_csv(directory / "metrics.csv")
    comparisons = read_csv(directory / "comparisons.csv")
    years = ", ".join(str(year) for year in manifest["years"])
    lines = [
        "## Saved ODI forecast comparison",
        "",
        f"Evaluation years: {years}. Models are fitted separately by competition,",
        "with past-only tuning and a fixed forecast model within each evaluation year.",
        "Lower log loss and Brier scores are better. Each match has equal weight.",
        "",
        "This is a retrospective pilot restricted to decisive matches with fixed XIs.",
        "It is not a published ranking of current player ability.",
        "",
    ]
    for window in sorted(manifest["windows"], key=lambda w: (w != 5, w)):
        label = "primary" if window == 5 else "sensitivity"
        lines += [
            f"### {window}-year window ({label})",
            "",
            "| Competition | Model | Matches | Log loss | Brier |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for gender in ("male", "female"):
            for model in MODEL_LABELS:
                row = next(
                    r
                    for r in metrics
                    if r["gender"] == gender
                    and r["model"] == model
                    and r["period"] == "pooled"
                    and int(r["window_years"]) == window
                )
                lines.append(
                    f"| {GENDER_LABELS[gender]} | {MODEL_LABELS[model]} "
                    f'| {row["matches"]} '
                    f'| {float(row["log_loss"]):.3f} | {float(row["brier"]):.3f} |'
                )
        lines += [
            "",
            "Paired differences below are model minus team-only log loss.",
            "Negative differences favor the model containing player identities.",
            "",
            "| Competition | Model | Difference | 95% block interval | Quarters |",
            "| --- | --- | ---: | --- | ---: |",
        ]
        for row in comparisons:
            if int(row["window_years"]) != window or row["metric"] != "log_loss":
                continue
            lines.append(
                f'| {GENDER_LABELS[row["gender"]]} | {MODEL_LABELS[row["model"]]} '
                f'| {float(row["difference"]):+.3f} '
                f'| [{float(row["lower"]):+.3f}, {float(row["upper"]):+.3f}] '
                f'| {row["quarter_blocks"]} |'
            )
        lines.append("")
    lines += [
        "Intervals resample quarters, retaining every match in each sampled quarter.",
        "Few blocks, dependence across quarters, and uncertainty from training",
        "and tuning limit their interpretation. They are conditional, exploratory",
        "summaries, not evidence that individual player coefficients are identified.",
        "",
        "The team baseline omits venue and dynamic strength. Player identities can",
        "proxy those changes; the combined model also changes team shrinkage.",
        "",
        "### Download the saved experiment",
        "",
    ]
    for filename in sorted(manifest["outputs"]):
        label = filename.replace("_", " ")
        lines.append(f"- [{label}](../out/release/lineup/{filename})")
    lines += [
        "- [Analysis provenance](../out/release/lineup/analysis_manifest.json)",
        "- [Publication checksums](../out/release/lineup/manifest.json)",
    ]
    lines += [
        "",
        "The coefficient table contains static associations in log-odds units,",
        "with exposure and co-occurrence diagnostics. It contains no WAR,",
        "player confidence intervals, or current-strength ranks. The saved analysis",
        "manifest records splits, dependencies, code hashes, and the input snapshot.",
        "",
    ]
    return "\n".join(lines)


def publish_comparison(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Publication output must not contain analysis inputs")
    manifest = verify_outputs(input_dir)
    report = comparison_report(input_dir, manifest)
    rankings = rank_players(read_csv(input_dir / "coefficients.csv"))
    with staged_directory(output_dir) as stage:
        for name in manifest["outputs"]:
            shutil.copyfile(input_dir / name, stage / name)
        shutil.copyfile(input_dir / "manifest.json", stage / "analysis_manifest.json")
        (stage / "report.md").write_text(report, encoding="utf-8")
        write_csv(stage / "rankings.csv", list(rankings[0]), rankings)
        (stage / "rankings.md").write_text(ranking_report(rankings), encoding="utf-8")
        publication = {
            "schema_version": 1,
            "experiment": "odi_lineup",
            "analysis_manifest_sha256": sha256(input_dir / "manifest.json"),
            "publication_code_sha256": {
                name: sha256(Path(__file__).parent / name)
                for name in ["publication.py", "rankings.py"]
            },
            "outputs": {p.name: sha256(p) for p in sorted(stage.iterdir())},
        }
        write_json(stage / "manifest.json", publication)
    return publication
