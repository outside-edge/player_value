import pytest

from player_value.site import check_site


def test_links_resolve_relative_paths_and_encoded_names(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="docs/page.html#heading">Page</a>'
        '<a href="https://example.org/">External</a>'
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/page.html").write_text(
        '<img src="../a%20b.svg"><a href="../index.html">Home</a>'
    )
    (tmp_path / "a b.svg").write_text("<svg/>")
    assert check_site(tmp_path) == 2


def test_broken_assets_fail(tmp_path):
    (tmp_path / "index.html").write_text('<script src="missing.js"></script>')
    with pytest.raises(ValueError, match="missing.js"):
        check_site(tmp_path)


def test_empty_site_fails(tmp_path):
    with pytest.raises(ValueError, match="No rendered website"):
        check_site(tmp_path)
