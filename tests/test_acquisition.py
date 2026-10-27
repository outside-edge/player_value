import io
import json
import zipfile

import pytest

from player_value.acquisition import fetch_matches
from player_value.io import sha256, staged_directory

ODI_URL = "https://cricsheet.org/downloads/odis_json.zip"


def archive(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as source:
        for name, value in members.items():
            source.writestr(name, value)
    return stream.getvalue()


def test_fetch_records_source_and_preserves_archive(monkeypatch, tmp_path):
    data = archive({"1.json": '{"info": {}}', "README.txt": "source notes"})

    def open_archive(url, timeout):
        assert url == ODI_URL
        assert timeout == 60
        return io.BytesIO(data)

    monkeypatch.setattr("player_value.acquisition.urlopen", open_archive)
    output = tmp_path / "raw"
    receipt = fetch_matches(output)
    assert json.loads((output / "1.json").read_text()) == {"info": {}}
    assert receipt["sha256"] == sha256(output / "odis_json.zip")
    assert receipt["match_files"] == 1
    assert receipt["source_url"] == ODI_URL
    assert (output / "odis_json.zip").read_bytes() == data


@pytest.mark.parametrize("member", ["../1.json", "/1.json", "notes.txt"])
def test_invalid_archives_preserve_existing_data(monkeypatch, tmp_path, member):
    data = archive({member: "{}"})
    monkeypatch.setattr(
        "player_value.acquisition.urlopen", lambda *a, **kw: io.BytesIO(data)
    )
    output = tmp_path / "raw"
    output.mkdir()
    (output / "original").write_text("keep")
    with pytest.raises(ValueError):
        fetch_matches(output)
    assert (output / "original").read_text() == "keep"
    assert not (tmp_path / "1.json").exists()


def test_stage_failure_preserves_previous_output(tmp_path):
    output = tmp_path / "build"
    output.mkdir()
    (output / "old").write_text("keep")
    with pytest.raises(RuntimeError):
        with staged_directory(output) as stage:
            (stage / "partial").write_text("incomplete")
            raise RuntimeError("stage failed")
    assert (output / "old").read_text() == "keep"
    assert not list(tmp_path.glob(".stage_*"))


def test_interrupted_install_requires_recovery(tmp_path):
    (tmp_path / "build.previous").mkdir()
    with pytest.raises(ValueError, match="interrupted"):
        with staged_directory(tmp_path / "build"):
            pytest.fail("Should stop before work begins")
