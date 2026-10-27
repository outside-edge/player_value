"""Acquire the official Cricsheet ODI archive with a source receipt."""

import shutil
import zipfile
from datetime import datetime, timezone
from urllib.request import urlopen

from player_value.io import sha256, staged_directory, write_json

ARCHIVES = {"Test": "tests", "ODI": "odis", "T20I": "t20s"}


def fetch_matches(destination, match_format="ODI"):
    source_url = f"https://cricsheet.org/downloads/{ARCHIVES[match_format]}_json.zip"
    with staged_directory(destination) as stage:
        archive = stage / source_url.rsplit("/", 1)[-1]
        with urlopen(source_url, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        count = 0
        with zipfile.ZipFile(archive) as source:
            names = set()
            for entry in source.infolist():
                name = entry.filename
                if not name.endswith(".json"):
                    continue
                if not name.removesuffix(".json").isdecimal() or name in names:
                    raise ValueError(f"Unexpected or duplicate archive member: {name}")
                names.add(name)
                with source.open(entry) as incoming, (stage / name).open("wb") as out:
                    shutil.copyfileobj(incoming, out)
                count += 1
        if not count:
            raise ValueError("The Cricsheet archive contains no match JSON files")
        receipt = {
            "source_url": source_url,
            "format": match_format,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "archive": archive.name,
            "sha256": sha256(archive),
            "match_files": count,
        }
        write_json(stage / "receipt.json", receipt)
    return receipt
