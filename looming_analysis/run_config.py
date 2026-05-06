"""Config-driven batch runner for looming analysis."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .config import AnalysisConfig, ResponsivenessConfig, RunConfig
from .pipeline import AnalysisResult, filter_by_iti, run_analysis
from .plots import (
    plot_heading_changes,
    plot_heading_changes_polar,
    plot_heading_traces,
    plot_inter_trigger_interval,
    plot_peak_velocity,
    plot_responses,
    plot_responsiveness_rates,
    plot_screen_position_effect,
    plot_sham_vs_real,
    plot_turn_proportions,
)
from .plots.peak_aligned import plot_latency_by_direction, plot_response_latency


FilesConfigPath = str | Path


def load_files_config(path: FilesConfigPath) -> dict[str, list[str]]:
    """Load a JSON files config with `groups`, `group` + `files`, or `files`."""
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if "groups" in data:
        return {
            str(group): [str(p) for p in paths]
            for group, paths in data["groups"].items()
        }
    if "group" in data and "files" in data:
        return {str(data["group"]): [str(p) for p in data["files"]]}
    if "files" in data:
        return {"experiment": [str(p) for p in data["files"]]}
    raise ValueError(
        "Files config must contain 'groups', 'group' with 'files', or 'files'."
    )


def load_files_configs(paths: Sequence[FilesConfigPath]) -> dict[str, list[str]]:
    """Load and merge one or more files configs."""
    if not paths:
        raise ValueError("At least one files config is required.")

    merged: dict[str, list[str]] = {}
    for path in paths:
        file_groups = load_files_config(path)
        for group, files in file_groups.items():
            if group in merged:
                raise ValueError(f"Duplicate group name in files configs: {group}")
            merged[group] = files
    return merged


def _normalize_files_config_paths(
    files_config: FilesConfigPath | Sequence[FilesConfigPath],
) -> list[FilesConfigPath]:
    if isinstance(files_config, (str, Path)):
        return [files_config]
    return list(files_config)


def build_output_dir(
    output_root: str | Path,
    file_groups: dict[str, list[str]],
    *,
    timestamp: str | None = None,
    suffix: str | None = None,
) -> Path:
    """Build `YYYYMMDD_HHMMSS-G1_G2...[_suffix]` output directory path."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    groups = "_".join(str(group) for group in file_groups)
    name = f"{timestamp}-{groups}"
    if suffix:
        name = f"{name}_{suffix}"
    return Path(output_root) / name


def run_from_config(
    files_config: FilesConfigPath | Sequence[FilesConfigPath],
    analysis_config: str | Path,
    *,
    output_root: str | Path = "outputs",
    suffix: str | None = None,
) -> Path:
    """Run analysis from config files and save outputs."""
    files_config_paths = _normalize_files_config_paths(files_config)
    file_groups = load_files_configs(files_config_paths)
    config = RunConfig.from_toml(analysis_config)
    output_dir = build_output_dir(output_root, file_groups, suffix=suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_kwargs = {k: v for k, v in vars(config.analysis).items()}
    min_iti_s = analysis_kwargs.pop("min_iti_s", None)

    result = run_analysis(
        file_groups,
        analysis=config.analysis,
        responsiveness=config.responsiveness,
    )

    if min_iti_s is not None:
        result = result.filter_by_iti(float(min_iti_s), verbose=True)

    result.to_dataframe(kind="scalar").write_csv(output_dir / "trials.csv")
    result.to_dataframe(kind="long").write_parquet(output_dir / "traces.parquet")

    plots = config.plots
    col_by = plots.get("col_by", "stimulus_offset_deg")
    hue_by = plots.get("hue_by", "group")
    row_by = plots.get("row_by", "is_responsive")
    responsive_only = plots.get("responsive_only", False)

    responses = result.responses
    if responsive_only:
        responses = [r for r in responses if r.get("is_responsive")]
        result = AnalysisResult(responses, config)

    figures = {
        "average-angular-velocity.png": plot_responses(
            responses, col_by=col_by, hue_by=hue_by, row_by=row_by
        ),
        "average-heading.png": plot_heading_traces(
            responses, col_by=col_by, hue_by=hue_by, row_by=row_by
        ),
        "heading-change-distribution.png": plot_heading_changes(
            responses, col_by=col_by, hue_by=hue_by, row_by=row_by
        ),
        "responsiveness-rates.png": plot_responsiveness_rates(
            responses, col_by=col_by, hue_by=hue_by
        ),
        "peak-angular-velocity.png": plot_peak_velocity(
            responses, col_by=col_by, hue_by=hue_by, row_by=row_by
        ),
        "turn-proportions.png": plot_turn_proportions(
            responses, col_by=col_by, group_by=hue_by
        ),
        "inter-trigger-interval.png": plot_inter_trigger_interval(
            responses, hue_by=hue_by,
            percentile_cutoff=plots.get("iti_percentile_cutoff", None),
        ),
        "response-latency.png": plot_response_latency(responses, hue_by=hue_by),
        "latency-by-direction.png": plot_latency_by_direction(responses, hue_by=hue_by),
        "peak-aligned-angular-velocity.png": plot_peak_velocity(
            responses, col_by=col_by, hue_by=hue_by, row_by=row_by
        ),
        "screen-position-effect.png": plot_screen_position_effect(responses, hue_by=hue_by),
        "heading-change-polar.png": plot_heading_changes_polar(
            responses, hue_by=hue_by, col_by=col_by
        ),
    }
    if any(r.get("is_sham") for r in responses):
        figures["sham-vs-real.png"] = plot_sham_vs_real(
            responses, col_by=col_by, hue_by=hue_by
        )

    for filename, fig in figures.items():
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")

    for path in files_config_paths:
        shutil.copy2(path, output_dir / Path(path).name)
    shutil.copy2(analysis_config, output_dir / Path(analysis_config).name)
    return output_dir


class _VerboseParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage()
        required = {
            "--files": "a JSON file with recording paths; repeat for reusable group files",
            "--analysis": "a TOML file with analysis parameters (thresholds, window sizes, etc.)",
        }
        for flag, desc in required.items():
            if flag in message:
                self.exit(
                    2,
                    f"\nerror: {message}\n"
                    f"  {flag} expects {desc}\n"
                    f"  example: {flag} path/to/config{'.json' if 'files' in flag else '.toml'}\n",
                )
        self.exit(2, f"\nerror: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _VerboseParser(
        description="Run looming analysis from config files.",
        epilog=(
            "examples:\n"
            "  %(prog)s --files experiment.json --analysis params.toml\n"
            "  %(prog)s --files CS.json --files Empty-Split.json --analysis params.toml\n"
            "  %(prog)s --files experiment.json --analysis params.toml --output-root /data/results"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--files",
        required=True,
        action="append",
        metavar="FILES_JSON",
        help=(
            "JSON config with recording file paths. Repeat to compare reusable "
            "single-group files."
        ),
    )
    parser.add_argument(
        "--analysis",
        required=True,
        metavar="ANALYSIS_TOML",
        help="TOML config defining analysis parameters (thresholds, window sizes, etc.).",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        metavar="DIR",
        help="Root directory for timestamped output folders (default: %(default)s).",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        metavar="LABEL",
        help="Optional string appended to the output folder name (e.g. 'responsive_only').",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = run_from_config(
        args.files,
        args.analysis,
        output_root=args.output_root,
        suffix=args.suffix,
    )
    print(output_dir)
    return 0
