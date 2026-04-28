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
from .pipeline import run_analysis


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
) -> Path:
    """Build `YYYYMMDD_HHMMSS-G1_G2...` output directory path."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    groups = "_".join(str(group) for group in file_groups)
    return Path(output_root) / f"{timestamp}-{groups}"


def run_from_config(
    files_config: str | Path,
    analysis_config: str | Path,
    *,
    output_root: str | Path = "outputs",
) -> Path:
    """Run analysis from config files and save outputs."""
    file_groups = load_files_config(files_config)
    config = load_analysis_config(analysis_config)
    output_dir = build_output_dir(output_root, file_groups)
    output_dir.mkdir(parents=True, exist_ok=False)

    result = run_analysis(
        file_groups,
        analysis=AnalysisConfig(**config["analysis"]),
        responsiveness=ResponsivenessConfig(**config["responsiveness"]),
    )

    result.to_dataframe(kind="scalar").write_csv(output_dir / "trials.csv")
    result.to_dataframe(kind="long").write_parquet(output_dir / "traces.parquet")

    plots = config["plots"]
    col_by = plots.get("col_by", "stimulus_offset_deg")
    hue_by = plots.get("hue_by", "group")

    figures = {
        "average-angular-velocity.png": result.plot_traces(
            col_by=col_by, hue_by=hue_by, row_by="is_responsive"
        ),
        "average-heading.png": result.plot_heading_traces(
            col_by=col_by, hue_by=hue_by, row_by="is_responsive"
        ),
        "heading-change-distribution.png": result.plot_heading_changes(
            col_by=col_by,
            hue_by=hue_by,
            row_by="is_responsive",
        ),
        "responsiveness-rates.png": result.plot_responsiveness_rates(
            col_by=col_by,
            hue_by=hue_by,
        ),
        "peak-angular-velocity.png": result.plot_peak_velocity(
            col_by=col_by,
            hue_by=hue_by,
            row_by="is_responsive",
        ),
        "turn-proportions.png": result.plot_turn_proportions(
            x_by=col_by,
            col_by=hue_by,
        ),
        "heading-change-comparison.png": result.plot_heading_change_comparison(
            group_by=hue_by,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = run_from_config(
        args.files,
        args.analysis,
        output_root=args.output_root,
    )
    print(output_dir)
    return 0
