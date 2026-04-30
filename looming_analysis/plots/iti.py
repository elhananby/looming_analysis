"""Inter-trigger interval histogram."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

    Each group is drawn as a semi-transparent histogram with:
    - a dashed vertical line at the mean,
    - a solid vertical line at the median (50th percentile),
    - a shaded band spanning the IQR (25th–75th percentile).

    Args:
        responses: Response list. Each response must have an
            ``inter_trigger_interval`` field (seconds); None values are skipped.
        hue_by: Field used for color grouping. Pass ``None`` for a single
            combined histogram.
        bins: Number of histogram bins.
        percentile_cutoff: If set (e.g. ``95``), values above this percentile
            across all groups are dropped before plotting.
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

        ax.hist(itv, bins=bins, color=color, alpha=0.45, density=True)

        mean_val = float(np.mean(itv))
        p25, p50, p75 = float(np.percentile(itv, 25)), float(np.percentile(itv, 50)), float(np.percentile(itv, 75))

        # IQR shaded band
        ax.axvspan(p25, p75, color=color, alpha=0.12)
        # Median — solid
        ax.axvline(p50, color=color, linestyle="-", linewidth=1.5)
        # Mean — dashed
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.5)

        label = f"{hv if hv is not None else 'all'}"
        label += f"\n  mean={mean_val:.0f} s  median={p50:.0f} s  IQR=[{p25:.0f}, {p75:.0f}] s  n={n_kept}"
        if cutoff is not None and n_kept < n_total:
            label += f"  ({n_total - n_kept} clipped)"
        legend_handles.append(Patch(facecolor=color, alpha=0.6, label=label))

    # Shared legend entries for line styles
    legend_handles += [
        Line2D([0], [0], color="grey", linestyle="-", linewidth=1.5, label="median"),
        Line2D([0], [0], color="grey", linestyle="--", linewidth=1.5, label="mean"),
        Patch(facecolor="grey", alpha=0.2, label="IQR (25–75th pct)"),
    ]

    title = "Inter-trigger interval distribution"
    if cutoff is not None:
        title += f"  [>{percentile_cutoff:.0f}th pct clipped at {cutoff:.0f} s]"
    ax.set_title(title)
    ax.set_xlabel("Inter-trigger interval (s)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3, axis="y")

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title=hue_by,
            loc="upper right",
            framealpha=0.9,
            fontsize=8,
        )

    fig.tight_layout()
    return fig
