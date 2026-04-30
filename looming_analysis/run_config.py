"""Config-driven batch runner for looming analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AnalysisConfig, ResponsivenessConfig
from .pipeline import AnalysisResult, filter_by_iti, run_analysis


def load_files_config(path: str | Path) -> dict[str, list[str]]:
    """Load a JSON files config with either `groups` or `files`."""
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if "groups" in data:
        return {
            str(group): [str(p) for p in paths]
            for group, paths in data["groups"].items()
        }
    if "files" in data:
        return {"experiment": [str(p) for p in data["files"]]}
    raise ValueError("Files config must contain either 'groups' or 'files'.")


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML analysis config."""
    config_path = Path(path)
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    return {
        "analysis": data.get("analysis", {}),
        "responsiveness": data.get("responsiveness", {}),
        "plots": data.get("plots", {}),
    }


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
    files_config: str | Path,
    analysis_config: str | Path,
    *,
    output_root: str | Path = "outputs",
    suffix: str | None = None,
) -> Path:
    """Run analysis from config files and save outputs."""
    file_groups = load_files_config(files_config)
    config = load_analysis_config(analysis_config)
    output_dir = build_output_dir(output_root, file_groups, suffix=suffix)
    output_dir.mkdir(parents=True, exist_ok=False)

    analysis_kwargs = dict(config["analysis"])
    min_iti_s = analysis_kwargs.pop("min_iti_s", None)

    result = run_analysis(
        file_groups,
        analysis=AnalysisConfig(**analysis_kwargs),
        responsiveness=ResponsivenessConfig(**config["responsiveness"]),
    )

    if min_iti_s is not None:
        result = result.filter_by_iti(float(min_iti_s), verbose=True)

    result.to_dataframe(kind="scalar").write_csv(output_dir / "trials.csv")
    result.to_dataframe(kind="long").write_parquet(output_dir / "traces.parquet")

    plots = config["plots"]
    col_by = plots.get("col_by", "stimulus_offset_deg")
    hue_by = plots.get("hue_by", "group")
    row_by = plots.get("row_by", "is_responsive")
    responsive_only = plots.get("responsive_only", False)

    responses = result.responses
    if responsive_only:
        responses = [r for r in responses if r.get("is_responsive")]
        result = AnalysisResult(responses)

    figures = {
        "average-angular-velocity.png": result.plot_traces(
            col_by=col_by, hue_by=hue_by, row_by=row_by
        ),
        "average-heading.png": result.plot_heading_traces(
            col_by=col_by, hue_by=hue_by, row_by=row_by
        ),
        "heading-change-distribution.png": result.plot_heading_changes(
            col_by=col_by,
            hue_by=hue_by,
            row_by=row_by,
        ),
        "responsiveness-rates.png": result.plot_responsiveness_rates(
            col_by=col_by,
            hue_by=hue_by,
        ),
        "peak-angular-velocity.png": result.plot_peak_velocity(
            col_by=col_by,
            hue_by=hue_by,
            row_by=row_by,
        ),
        "turn-proportions.png": result.plot_turn_proportions(
            x_by=col_by,
            col_by=hue_by,
        ),
        "inter-trigger-interval.png": result.plot_inter_trigger_interval(
            hue_by=hue_by,
            percentile_cutoff=plots.get("iti_percentile_cutoff", None),
        ),
        "response-latency.png": result.plot_response_latency(hue_by=hue_by),
        "latency-by-direction.png": result.plot_latency_by_direction(hue_by=hue_by),
        "peak-aligned-angular-velocity.png": result.plot_peak_aligned_traces(
            col_by=col_by,
            hue_by=hue_by,
            row_by=row_by,
            half_window_ms=plots.get("peak_aligned_half_window_ms", 100),
            fallback_window_ms=plots.get("peak_aligned_fallback_window_ms", 200),
        ),
        "screen-position-effect.png": result.plot_screen_position_effect(
            hue_by=hue_by,
        ),
    }
    for filename, fig in figures.items():
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")

    shutil.copy2(files_config, output_dir / Path(files_config).name)
    shutil.copy2(analysis_config, output_dir / Path(analysis_config).name)
    return output_dir


class _VerboseParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage()
        required = {
            "--files": "a JSON file mapping group names to lists of recording paths",
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
            "  %(prog)s --files experiment.json --analysis params.toml --output-root /data/results"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--files",
        required=True,
        metavar="FILES_JSON",
        help="JSON config mapping group names to lists of recording file paths.",
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
