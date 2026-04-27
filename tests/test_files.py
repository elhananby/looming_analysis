from __future__ import annotations

from looming_analysis.files import find_braidz


def test_find_braidz_returns_sorted_paths(tmp_path):
    (tmp_path / "b.braidz").write_text("", encoding="utf-8")
    (tmp_path / "a.braidz").write_text("", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")

    paths = find_braidz(tmp_path)

    assert paths == [
        str(tmp_path / "a.braidz"),
        str(tmp_path / "b.braidz"),
    ]


def test_find_braidz_can_search_recursively(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.braidz").write_text("", encoding="utf-8")

    assert find_braidz(tmp_path, recursive=False) == []
    assert find_braidz(tmp_path, recursive=True) == [str(nested / "a.braidz")]
