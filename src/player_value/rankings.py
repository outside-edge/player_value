"""Rank saved player coefficients within each fitted model and competition."""

import math
from collections import defaultdict
from datetime import date


def rank_players(coefficients):
    groups = defaultdict(list)
    for row in coefficients:
        if row["feature_kind"] != "player":
            continue
        score = float(row["coefficient_log_odds"])
        if not math.isfinite(score):
            raise ValueError("Cannot rank nonfinite player coefficients")
        key = (
            row["gender"],
            row["model"],
            int(row["window_years"]),
            int(row["evaluation_year"]),
        )
        groups[key].append(row)
    if not groups:
        raise ValueError("No player coefficients to rank")
    rankings = []
    for (gender, model, window, year), players in sorted(groups.items()):
        if len({r["entity"] for r in players}) != len(players):
            raise ValueError("Duplicate player coefficient within a fitted model")
        players.sort(
            key=lambda r: (-round(float(r["coefficient_log_odds"]), 12), r["entity"])
        )
        previous, rank = None, 0
        for position, row in enumerate(players, start=1):
            score = round(float(row["coefficient_log_odds"]), 12)
            if score != previous:
                rank = position
            previous = score
            appearances = int(row["training_appearances"])
            rankings.append(
                {
                    "gender": gender,
                    "model": model,
                    "window_years": window,
                    "fit_cutoff": date(year, 1, 1).isoformat(),
                    "training_start": date(year - window, 1, 1).isoformat(),
                    "training_end": date(year - 1, 12, 31).isoformat(),
                    "rank": rank,
                    "player_id": row["entity"],
                    "player_name": row["player_name"],
                    "rating_log_odds": float(row["coefficient_log_odds"]),
                    "training_appearances": appearances,
                    "fewer_than_20_appearances": int(appearances < 20),
                    "regularization_c": float(row["regularization_c"]),
                    "max_teammate_share": float(row["max_teammate_share"]),
                    "identical_pattern_group": row["identical_pattern_group"],
                }
            )
    return rankings


def ranking_report(rankings):
    cutoff = max(row["fit_cutoff"] for row in rankings)
    windows = {row["window_years"] for row in rankings if row["fit_cutoff"] == cutoff}
    window = 5 if 5 in windows else max(windows)
    sample = [
        r
        for r in rankings
        if r["fit_cutoff"] == cutoff
        and r["window_years"] == window
        and r["model"] == "combined"
    ]
    if not sample:
        raise ValueError("No combined-model rankings for the latest saved cutoff")
    lines = [
        "## Rankings from the latest saved fit",
        "",
        f'**Training period: {sample[0]["training_start"]} through '
        f'{sample[0]["training_end"]}. Fit cutoff: {cutoff}.**',
        "",
        "These tables rank player coefficients from the combined team-plus-player",
        "model. Higher values indicate a larger estimated contribution to the team's",
        "log odds of winning, conditional on its team effect and other predictors.",
        "They describe the training window, not current form or WAR.",
        "All players are eligible for display, including those no longer active.",
        "",
        "The tables show the first 20 rows per competition. The full download includes",
        "both player-only and combined models, every fitted cutoff, and both windows.",
        "Ranks are assigned separately within each fit. Equal coefficients share rank;",
        "rounding to twelve decimal places resolves numerical ties only.",
        "",
    ]
    for gender, label in [("male", "Men"), ("female", "Women")]:
        lines += [
            f"### {label}'s ODIs",
            "",
            "| Rank | Player | Rating (log odds) | Appearances | Flags |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
        for row in [r for r in sample if r["gender"] == gender][:20]:
            flags = []
            if row["fewer_than_20_appearances"]:
                flags.append("Fewer than 20 appearances")
            if row["identical_pattern_group"]:
                flags.append("Identical lineup pattern")
            name = row["player_name"].replace("|", "\\|")
            lines.append(
                f'| {row["rank"]} | {name} | {row["rating_log_odds"]:+.3f} '
                f'| {row["training_appearances"]} | {"; ".join(flags)} |'
            )
        lines.append("")
    lines += [
        "The 20-appearance flag is a display convention, not a validation threshold.",
        "Identical lineup patterns mean the match outcomes cannot distinguish those",
        "players individually. No rank confidence intervals have been estimated.",
        "",
        "[Download all player rankings](../out/release/lineup/rankings.csv).",
        "",
    ]
    return "\n".join(lines)
