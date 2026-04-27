from __future__ import annotations

from looming_analysis.io import _braidz_cache_path


def test_braidz_cache_path_includes_stem_and_hash(tmp_path):
    first = tmp_path / "one" / "recording.braidz"
    second = tmp_path / "two" / "recording.braidz"

    first_cache = _braidz_cache_path(first, tmp_path / "cache")
    second_cache = _braidz_cache_path(second, tmp_path / "cache")

    assert first_cache.parent == tmp_path / "cache"
    assert second_cache.parent == tmp_path / "cache"
    assert first_cache.name.startswith("recording-")
    assert second_cache.name.startswith("recording-")
    assert first_cache != second_cache
