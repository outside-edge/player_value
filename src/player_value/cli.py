"""Command-line entry points for acquisition, preparation, and website checks."""

import argparse
import json
from datetime import date
from pathlib import Path

from player_value.acquisition import ARCHIVES, fetch_matches
from player_value.preparation import prepare
from player_value.site import check_site


def main(argv=None):
    parser = argparse.ArgumentParser(prog="player-value")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch", help="Download an official international JSON archive"
    )
    fetch.add_argument("--format", choices=ARCHIVES, default="ODI")
    fetch.add_argument("--output", type=Path)
    build = commands.add_parser(
        "prepare", help="Validate and prepare international participants"
    )
    build.add_argument("--format", choices=ARCHIVES, default="ODI")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path)
    build.add_argument("--as-of", type=date.fromisoformat, required=True)
    build.add_argument("--since", type=date.fromisoformat)
    rankings = commands.add_parser(
        "rank", help="Fit player rankings by international format"
    )
    rankings.add_argument("--input", type=Path, default=Path("data/build"))
    rankings.add_argument("--output", type=Path, default=Path("out/release/current"))
    rankings.add_argument(
        "--specification", type=Path, default=Path("docs/format_rankings.qmd")
    )
    rankings.add_argument("--as-of", type=date.fromisoformat, required=True)
    compare_parser = commands.add_parser("compare", help="Evaluate ODI lineup models")
    compare_parser.add_argument("--input", type=Path, default=Path("data/build/odis"))
    compare_parser.add_argument(
        "--output", type=Path, default=Path("out/analysis/lineup")
    )
    compare_parser.add_argument(
        "--years", type=int, nargs="+", default=[2023, 2024, 2025]
    )
    compare_parser.add_argument("--windows", type=int, nargs="+", default=[5, 3])
    publish = commands.add_parser("publish", help="Publish a saved lineup comparison")
    publish.add_argument("--input", type=Path, default=Path("out/analysis/lineup"))
    publish.add_argument("--output", type=Path, default=Path("out/release/lineup"))
    release = commands.add_parser(
        "check-release", help="Verify saved publication files"
    )
    release.add_argument("--input", type=Path, default=Path("out/release/lineup"))
    site = commands.add_parser("check-site", help="Check rendered links and assets")
    site.add_argument("--input", type=Path, default=Path("out/site"))
    content = commands.add_parser("site-content", help="Prepare saved format tables")
    content.add_argument("--input", type=Path, default=Path("out/release/current"))
    content.add_argument("--output", type=Path, default=Path("out/site-content"))
    args = parser.parse_args(argv)
    if args.command in {"fetch", "prepare"} and args.output is None:
        folder = {"Test": "tests", "ODI": "odis", "T20I": "t20is"}[args.format]
        root = "data/raw/cricsheet" if args.command == "fetch" else "data/build"
        args.output = Path(root) / folder
    try:
        if args.command == "fetch":
            result = fetch_matches(args.output, args.format)
        elif args.command == "prepare":
            manifest = prepare(
                args.input, args.output, args.as_of, args.format, args.since
            )
            result = {key: manifest[key] for key in ["as_of", "summary", "excluded"]}
        elif args.command == "rank":
            from player_value.current import rank_formats

            manifest = rank_formats(
                args.input, args.output, args.as_of, args.specification
            )
            result = {
                "output": str(args.output),
                "as_of": manifest["as_of"],
                "cohorts": [
                    {
                        key: c[key]
                        for key in ("format", "gender", "matches", "players", "status")
                    }
                    for c in manifest["cohorts"]
                ],
            }
        elif args.command == "compare":
            from player_value.evaluation import compare

            manifest = compare(args.input, args.output, args.years, args.windows)
            result = {
                "output": str(args.output),
                "years": manifest["years"],
                "windows": manifest["windows"],
                "splits": len(manifest["splits"]),
            }
        elif args.command == "publish":
            from player_value.publication import publish_comparison

            manifest = publish_comparison(args.input, args.output)
            result = {"output": str(args.output), "files": len(manifest["outputs"])}
        elif args.command == "check-release":
            from player_value.publication import verify_outputs

            manifest = verify_outputs(args.input)
            result = {"files_verified": len(manifest["outputs"])}
        elif args.command == "site-content":
            from player_value.site import prepare_site_content

            result = {"files_written": prepare_site_content(args.input, args.output)}
        else:
            result = {"html_pages_checked": check_site(args.input)}
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))
