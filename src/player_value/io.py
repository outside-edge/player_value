"""Checksums and staged output replacement."""

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@contextmanager
def staged_directory(destination):
    """Keep the previous output intact if building or installing a stage fails."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = destination.with_name(destination.name + ".previous")
    if previous.exists():
        raise ValueError(f"Resolve interrupted output before continuing: {previous}")
    staging = Path(tempfile.mkdtemp(prefix=".stage_", dir=destination.parent))
    try:
        yield staging
        if destination.exists():
            destination.rename(previous)
        try:
            staging.rename(destination)
        except OSError:
            if previous.exists():
                previous.rename(destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
