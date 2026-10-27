import json
from pathlib import Path

import pytest


@pytest.fixture
def match():
    return json.loads((Path(__file__).parent / "fixtures" / "1.json").read_text())


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "input"
    path.mkdir()
    return path
