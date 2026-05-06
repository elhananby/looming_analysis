"""Debug plot overlaying heading-change method variants."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ..._types import Response
from .._common import unique_values


def _lighten(color, factor: float = 0.55) -> tuple:
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * factor)


def plot_heading_change_comparison(
    responses: list[Response],
    group_by: str = "group",
    heading_threshold_deg: float = 30.0,
    n_polar_bins: int = 36,
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """Violin + polar histogram of heading_change, split by group and responsiveness.

    Args:
        responses: Response list (must have heading_change set by classify_responsiveness).
        group_by: Field used to split into groups (default "group").
        heading_threshold_deg: Dashed threshold lines on violin panels.
        n_polar_bins: Number of angular bins in polar histograms (default 36 = 10 deg).
        figsize: Overall figure size. Defaults to (10, 8).

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.colors as mcolors

    if figsize is None:
        figsize = (10, 8)

    groups = unique_values(responses, group_by) or [None]
    base_colors = plt.cm.tab10(np.linspace(0, 0.9, len(groups)))
    color_map = {g: mcolors.to_hex(c) for g, c in zip(groups, base_colors)}

    fig = plt.figure(figsize=figsize)
    ax_violin = fig.add_subplot(1, 2, 1)
    ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

    violin_width = 0.55
    group_spacing = 1.0
    responsive_offset = 0.35

    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for gi, grp in enumerate(groups):
        x_center = gi * group_spacing
        base_col = color_map[grp]
        light_col = _lighten(base_col)

        for responsive, x_off, face_col in (
            (True, +responsive_offset / 2, base_col),
            (False, -responsive_offset / 2, light_col),
        ):
            subset = [
                r for r in responses
                if (group_by is None or r.get(group_by) == grp)
                and r.get("is_responsive") is responsive
            ]
            vals = [
                r["heading_change"]
                for r in subset
                if r.get("heading_change") is not None
                and not np.isnan(r["heading_change"])
            ]
            if not vals:
                continue
            pos = x_center + x_off
            vp = ax_violin.violinplot([vals], positions=[pos], widths=violin_width,
                                      showmeans=True, showmedians=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(face_col)
                pc.set_alpha(0.85)
            for part in ("cbars", "cmins", "cmaxes", "cmeans"):
                if part in vp:
                    vp[part].set_color(face_col)
            ax_violin.text(pos, max(vals), f"n={len(vals)}", ha="center", va="bottom", fontsize=6)

        tick_positions.append(x_center)
        tick_labels.append(str(grp) if grp is not None else "all")

    ax_violin.axhline(heading_threshold_deg, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_violin.axhline(-heading_threshold_deg, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_violin.axhline(0, color="k", linewidth=0.4, alpha=0.3)
    ax_violin.set_xticks(tick_positions)
    ax_violin.set_xticklabels(tick_labels, fontsize=8)
    ax_violin.set_title("heading_change (vector method)", fontsize=9)
    ax_violin.set_ylabel("degrees")
    ax_violin.grid(True, axis="y", alpha=0.3)

    bin_edges = np.linspace(-np.pi, np.pi, n_polar_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)

    for gi, grp in enumerate(groups):
        base_col = color_map[grp]
        light_col = _lighten(base_col)
        for responsive, face_col in ((True, base_col), (False, light_col)):
            subset = [
                r for r in responses
                if (group_by is None or r.get(group_by) == grp)
                and r.get("is_responsive") is responsive
            ]
            vals_deg = [
                r["heading_change"] for r in subset
                if r.get("heading_change") is not None and not np.isnan(r["heading_change"])
            ]
            if not vals_deg:
                continue
            vals_rad = np.deg2rad(vals_deg)
            counts, _ = np.histogram(vals_rad, bins=bin_edges)
            ax_polar.bar(bin_centers, counts, width=bin_width, color=face_col, alpha=0.6)

    ax_polar.set_title("heading_change polar", fontsize=9, pad=12)

    legend_handles = []
    for grp in groups:
        col = color_map[grp]
        lbl = str(grp) if grp is not None else "all"
        legend_handles.append(Patch(facecolor=col, alpha=0.85, label=f"{lbl} (resp.)"))
        legend_handles.append(Patch(facecolor=_lighten(col), alpha=0.85, label=f"{lbl} (non-resp.)"))

    fig.legend(handles=legend_handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    fig.suptitle("Heading change debug", y=1.01)
    fig.tight_layout()
    return fig
