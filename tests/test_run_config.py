from __future__ import annotations

from pathlib import Path

import pytest

from looming_analysis.run_config import (
    build_parser,
    build_output_dir,
    load_analysis_config,
    load_files_config,
    load_files_configs,
    run_from_config,
)


def test_load_files_config_json(tmp_path):
    config_path = tmp_path / "files.json"
    config_path.write_text(
        '{"groups": {"CS": ["a.braidz"], "DNp03": ["b.braidz"]}}',
        encoding="utf-8",
    )

    assert load_files_config(config_path) == {
        "CS": ["a.braidz"],
        "DNp03": ["b.braidz"],
    }


def test_load_files_config_accepts_explicit_single_group_file(tmp_path):
    config_path = tmp_path / "cs.json"
    config_path.write_text(
        '{"group": "CS", "files": ["a.braidz", "b.braidz"]}',
        encoding="utf-8",
    )

    assert load_files_config(config_path) == {
        "CS": ["a.braidz", "b.braidz"],
    }


def test_load_files_configs_merges_repeated_file_configs(tmp_path):
    cs_config = tmp_path / "cs.json"
    cs_config.write_text('{"group": "CS", "files": ["a.braidz"]}', encoding="utf-8")
    empty_split_config = tmp_path / "empty-split.json"
    empty_split_config.write_text(
        '{"group": "Empty-Split", "files": ["b.braidz"]}',
        encoding="utf-8",
    )

    assert load_files_configs([cs_config, empty_split_config]) == {
        "CS": ["a.braidz"],
        "Empty-Split": ["b.braidz"],
    }


def test_load_files_configs_rejects_duplicate_group_names(tmp_path):
    first_config = tmp_path / "first.json"
    first_config.write_text('{"group": "CS", "files": ["a.braidz"]}', encoding="utf-8")
    second_config = tmp_path / "second.json"
    second_config.write_text('{"group": "CS", "files": ["b.braidz"]}', encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate group name"):
        load_files_configs([first_config, second_config])


def test_parser_accepts_repeated_files_args():
    args = build_parser().parse_args(
        [
            "--files",
            "CS.json",
            "--files",
            "Empty-Split.json",
            "--analysis",
            "analysis.toml",
        ]
    )

    assert args.files == ["CS.json", "Empty-Split.json"]


def test_load_analysis_config_toml(tmp_path):
    config_path = tmp_path / "analysis.toml"
    config_path.write_text(
        """
[analysis]
pre_ms = 100
post_ms = 500
include_sham = true
cache_dir = ".braidz_cache"

[responsiveness]
method = "combined"
threshold_deg_s = 500
heading_threshold_deg = 45
baseline_window_ms = [-90, -10]
window_ms = [100, 200]

[plots]
col_by = "stimulus_offset_deg"
hue_by = "group"
""",
        encoding="utf-8",
    )

    config = load_analysis_config(config_path)

    assert config["analysis"]["pre_ms"] == 100
    assert config["responsiveness"]["method"] == "combined"
    assert config["responsiveness"]["window_ms"] == [100, 200]
    assert config["plots"]["col_by"] == "stimulus_offset_deg"


def test_build_output_dir_uses_timestamp_and_groups(tmp_path):
    out = build_output_dir(
        tmp_path,
        {"CS": ["a.braidz"], "DNp03": ["b.braidz"]},
        timestamp="20260427_153000",
    )

    assert out == tmp_path / "20260427_153000-CS_DNp03"


def test_run_from_config_saves_data_plots_and_config_snapshots(monkeypatch, tmp_path):
    cs_config = tmp_path / "cs.json"
    cs_config.write_text('{"group": "CS", "files": ["a.braidz"]}', encoding="utf-8")
    empty_split_config = tmp_path / "empty-split.json"
    empty_split_config.write_text(
        '{"group": "Empty-Split", "files": ["b.braidz"]}',
        encoding="utf-8",
    )
    analysis_config = tmp_path / "analysis.toml"
    analysis_config.write_text(
        """
[analysis]
pre_ms = 100
post_ms = 500

[responsiveness]
method = "combined"

[plots]
col_by = "stimulus_offset_deg"
hue_by = "group"
""",
        encoding="utf-8",
    )

    class FakeFrame:
        def __init__(self, name):
            self.name = name

        def write_csv(self, path):
            Path(path).write_text(self.name, encoding="utf-8")

        def write_parquet(self, path):
            Path(path).write_bytes(self.name.encode("utf-8"))

    class FakeFigure:
        def __init__(self, name):
            self.name = name

        def savefig(self, path, **kwargs):
            Path(path).write_text(self.name, encoding="utf-8")

    class FakeResult:
        responses = []

        def to_dataframe(self, kind, backend="polars"):
            return FakeFrame(kind)

        def plot_traces(self, **kwargs):
            return FakeFigure("average-angular-velocity")

        def plot_heading_traces(self, **kwargs):
            return FakeFigure("average-heading")

        def plot_heading_changes(self, **kwargs):
            return FakeFigure("heading-change-distribution")

        def plot_responsiveness_rates(self, **kwargs):
            return FakeFigure("responsiveness-rates")

        def plot_peak_velocity(self, **kwargs):
            return FakeFigure("peak-angular-velocity")

        def plot_turn_proportions(self, **kwargs):
            return FakeFigure("turn-proportions")

        def plot_heading_change_comparison(self, **kwargs):
            return FakeFigure("heading-change-comparison")

        def plot_inter_trigger_interval(self, **kwargs):
            return FakeFigure("inter-trigger-interval")

        def plot_response_latency(self, **kwargs):
            return FakeFigure("response-latency")

        def plot_latency_by_direction(self, **kwargs):
            return FakeFigure("latency-by-direction")

        def plot_peak_aligned_traces(self, **kwargs):
            return FakeFigure("peak-aligned-angular-velocity")

        def plot_screen_position_effect(self, **kwargs):
            return FakeFigure("screen-position-effect")

        def plot_heading_changes_polar(self, **kwargs):
            return FakeFigure("heading-change-polar")

    calls = {}

    def fake_run_analysis(file_groups, **kwargs):
        calls["file_groups"] = file_groups
        return FakeResult()

    monkeypatch.setattr(
        "looming_analysis.run_config.run_analysis",
        fake_run_analysis,
    )
    monkeypatch.setattr(
        "looming_analysis.run_config.build_output_dir",
        lambda output_root, file_groups, **kwargs: Path(output_root)
        / "20260427_153000-CS_Empty-Split",
    )

    output_dir = run_from_config(
        [cs_config, empty_split_config],
        analysis_config,
        output_root=tmp_path / "outputs",
    )

    assert calls["file_groups"] == {
        "CS": ["a.braidz"],
        "Empty-Split": ["b.braidz"],
    }
    assert (output_dir / "trials.csv").read_text(encoding="utf-8") == "scalar"
    assert (output_dir / "traces.parquet").read_bytes() == b"long"
    assert (output_dir / "average-angular-velocity.png").exists()
    assert (output_dir / "average-heading.png").exists()
    assert (output_dir / "heading-change-distribution.png").exists()
    assert (output_dir / "responsiveness-rates.png").exists()
    assert (output_dir / "peak-angular-velocity.png").exists()
    assert (output_dir / "turn-proportions.png").exists()
    assert (output_dir / "inter-trigger-interval.png").exists()
    assert (output_dir / "response-latency.png").exists()
    assert (output_dir / "latency-by-direction.png").exists()
    assert (output_dir / "peak-aligned-angular-velocity.png").exists()
    assert (output_dir / "screen-position-effect.png").exists()
    assert (output_dir / "cs.json").exists()
    assert (output_dir / "empty-split.json").exists()
    assert (output_dir / "analysis.toml").exists()
