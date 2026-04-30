"""Inter-trigger interval histogram."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .._types import Response
from ._common import build_hue_colormap, unique_values


def plot_inter_trigger_interval(
    responses: list[Response],
    *,
    hue_by: Optional[str] = "group",
    bins: int = 50,
    percentile_cutoff: Optional[float] = None,
    ax_size: tuple[float, float] = (8, 4),
) -> Figure:
    """Histogram of inter-trigger intervals (time between consecutive triggers).

    Args:
        responses: Response list. Each response must have an
            ``inter_trigger_interval`` field (seconds); None values are skipped.
        hue_by: Field used for color grouping. Overlapping semi-transparent
            histograms are drawn per unique value. Pass ``None`` for a single
            combined histogram.
        bins: Number of histogram bins.
        percentile_cutoff: If set (e.g. ``99``), values above this percentile
            across all groups are dropped before plotting. Useful for removing
            long idle periods that compress the histogram scale.
        ax_size: Figure size ``(width, height)`` in inches.
    """
    fig, ax = plt.subplots(figsize=ax_size)
    color_map = build_hue_colormap(responses, hue_by)
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]

    # Compute global cutoff across all groups so axes are comparable.
    cutoff: Optional[float] = None
    if percentile_cutoff is not None:
        all_itv = [
            float(r["inter_trigger_interval"])
            for r in responses
            if r.get("inter_trigger_interval") is not None
            and not np.isnan(float(r["inter_trigger_interval"]))
        ]
        if all_itv:
            cutoff = float(np.percentile(all_itv, percentile_cutoff))

    legend_handles = []
    for hv in hue_vals:
        subset = [
            r for r in responses
            if (hue_by is None or r.get(hue_by) == hv)
        ]
        itv = [
            float(r["inter_trigger_interval"])
            for r in subset
            if r.get("inter_trigger_interval") is not None
            and not np.isnan(float(r["inter_trigger_interval"]))
        ]
        if not itv:
            continue
        itv = np.array(itv, dtype=float)
        n_total = len(itv)
        if cutoff is not None:
            itv = itv[itv <= cutoff]
        n_kept = len(itv)
        color = color_map[hv]
        ax.hist(itv, bins=bins, color=color, alpha=0.5, density=True)

        # Percentile markers: dashed lines at 5/25/50/75 %
        pct_styles = {5: (":", 0.7), 25: ("-.", 0.8), 50: ("--", 1.0), 75: ("-.", 0.8)}
        for pct, (ls, alpha) in pct_styles.items():
            pv = float(np.percentile(itv, pct))
            ax.axvline(pv, color=color, linestyle=ls, linewidth=1.0, alpha=alpha)

        mean_val = float(np.mean(itv))
        p50 = float(np.percentile(itv, 50))
        label = (
            f"{hv if hv is not None else 'all'}  "
            f"(mean={mean_val:.1f} s, median={p50:.1f} s, n={n_kept}"
        )
        if cutoff is not None and n_kept < n_total:
            label += f", {n_total - n_kept} outliers removed"
        label += ")"
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2, label=label))

    ax.set_xlabel("Inter-trigger interval (s)")
    ax.set_ylabel("Density")
    title = "Inter-trigger interval distribution  (lines: 5/25/50/75th percentile)"
    if cutoff is not None:
        title += f"\n>{percentile_cutoff}th percentile removed (cutoff={cutoff:.1f} s)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    if legend_handles:
        ax.legend(handles=legend_handles, title=hue_by, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    return fig
