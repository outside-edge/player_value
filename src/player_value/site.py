"""Validate local links and assets in the rendered publication."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from player_value.io import staged_directory


def prepare_site_content(source, destination):
    """Generate format pages from a verified publication without fitting models."""
    from player_value.current import FORMATS, report
    from player_value.lineup import read_csv
    from player_value.publication import verify_outputs

    source, destination = Path(source), Path(destination)
    if source.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Site content output must not contain its source")
    manifest = verify_outputs(source)
    rows = read_csv(source / "rankings.csv")
    for row in rows:
        for key in ("rank", "training_appearances", "fewer_than_20_appearances"):
            row[key] = int(row[key])
        row["rating_log_win_loss_odds"] = float(row["rating_log_win_loss_odds"])
    coverage = [
        f"Publication cutoff: **{manifest['as_of']}**. "
        f"Window starts **{manifest['training_start']}**.",
        "",
        "| Format | Competition | Available matches | Used | Players | Download |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for cohort in manifest["cohorts"]:
        fmt, gender = cohort["format"], cohort["gender"]
        label = "Men" if gender == "male" else "Women"
        coverage.append(
            f"| {fmt} | {label} | {cohort['available_window_matches']} | "
            f"{cohort['matches']} | {cohort['players']} | "
            f"[CSV](../out/release/current/{FORMATS[fmt]}_{gender}.csv) |"
        )
    with staged_directory(destination) as stage:
        for fmt in FORMATS:
            text = report(
                [row for row in rows if row["format"] == fmt],
                [c for c in manifest["cohorts"] if c["format"] == fmt],
                manifest["as_of"],
            )
            (stage / f"{fmt.lower()}.md").write_text(text, encoding="utf-8")
        (stage / "coverage.md").write_text("\n".join(coverage) + "\n", encoding="utf-8")
    return len(FORMATS) + 1


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.targets.append(value)


def check_site(directory):
    root = Path(directory).resolve()
    pages = sorted(root.rglob("*.html"))
    if not pages or not (root / "index.html").is_file():
        raise ValueError(f"No rendered website at {root}")
    missing = []
    for page in pages:
        links = Links()
        links.feed(page.read_text(encoding="utf-8"))
        for value in links.targets:
            url = urlsplit(value)
            if url.scheme or url.netloc or not url.path:
                continue
            path = unquote(url.path)
            target = (
                (root / path.lstrip("/"))
                if path.startswith("/")
                else page.parent / path
            )
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                missing.append(f"{page.relative_to(root)}: {value}")
    if missing:
        raise ValueError("Broken local links:\n" + "\n".join(missing))
    return len(pages)
