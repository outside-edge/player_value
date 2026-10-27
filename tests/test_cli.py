import json
import subprocess
import sys
from pathlib import Path


def test_installed_cli_prepares_fixture(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from player_value.cli import main; main()",
            "prepare",
            "--input",
            str(Path(__file__).parent / "fixtures"),
            "--output",
            str(tmp_path / "build"),
            "--as-of",
            "2024-12-31",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["summary"]["male"]["matches"] == 1
    assert (tmp_path / "build" / "participants.csv").is_file()
