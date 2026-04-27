"""Minimal looming-analysis workflow.

Run from the repository root after installing the package:

    python examples/quickstart.py /mnt/data/experiments
"""

from __future__ import annotations

import sys
from pathlib import Path

from looming_analysis import (
    AnalysisConfig,
    ResponsivenessConfig,
    find_braidz,
    run_analysis,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/quickstart.py /path/to/experiments")

    root = Path(sys.argv[1])
    files = find_braidz(root)
    if not files:
        raise SystemExit(f"No .braidz files found in {root}")

    result = run_analysis(
        {"experiment": files},
        analysis=AnalysisConfig(pre_ms=100.0, post_ms=500.0, cache_dir=".braidz_cache"),
        responsiveness=ResponsivenessConfig(
            method="combined",
            threshold_deg_s=500.0,
            heading_threshold_deg=45.0,
            baseline_window_ms=(-90.0, -10.0),
            window_ms=(100.0, 200.0),
        ),
    )

    trials = result.to_dataframe(kind="scalar")
    print(trials)

    fig = result.plot_traces(hue_by="stimulus_offset_deg", baseline_subtract=True)
    fig.savefig("looming-traces.png", dpi=150, bbox_inches="tight")

    heading_fig = result.plot_heading_traces(hue_by="stimulus_offset_deg")
    heading_fig.savefig("looming-heading.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
