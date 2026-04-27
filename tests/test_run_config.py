from __future__ import annotations

from pathlib import Path

from looming_analysis.run_config import (
    build_output_dir,
    load_analysis_config,
    load_files_config,
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
    files_config = tmp_path / "files.json"
    files_config.write_text('{"groups": {"CS": ["a.braidz"]}}', encoding="utf-8")
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

    monkeypatch.setattr(
        "looming_analysis.run_config.run_analysis", lambda *args, **kwargs: FakeResult()
    )
    monkeypatch.setattr(
        "looming_analysis.run_config.build_output_dir",
        lambda output_root, file_groups: Path(output_root) / "20260427_153000-CS",
    )

    output_dir = run_from_config(
        files_config,
        analysis_config,
        output_root=tmp_path / "outputs",
    )

    assert (output_dir / "trials.csv").read_text(encoding="utf-8") == "scalar"
    assert (output_dir / "traces.parquet").read_bytes() == b"long"
    assert (output_dir / "average-angular-velocity.png").exists()
    assert (output_dir / "average-heading.png").exists()
    assert (output_dir / "heading-change-distribution.png").exists()
    assert (output_dir / "responsiveness-rates.png").exists()
    assert (output_dir / "peak-angular-velocity.png").exists()
    assert (output_dir / "turn-proportions.png").exists()
    assert (output_dir / "files.json").exists()
    assert (output_dir / "analysis.toml").exists()
