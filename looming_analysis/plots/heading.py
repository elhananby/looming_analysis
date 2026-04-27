"""Heading change distribution (violin) plots, faceted."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .._types import Response
from ._common import plot_violin_facets


def plot_heading_changes(
    responses: list[Response],
    *,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    absolute: bool = False,
    responsive_only: bool = False,
    ax_size: tuple[float, float] = (7, 5),
) -> Figure:
    """Violin plot of heading changes.

    Dimension roles for this plot:
        - `row_by`:  subplot rows (None → single row)
        - `col_by`:  x-axis tick categories within each subplot
        - `hue_by`:  side-by-side violins within each x tick

    Args:
        responses: Response list (each must have `heading_change`).
        row_by: Column for subplot rows.
        col_by: Column for x-axis ticks. If None, all responses form a single tick.
        hue_by: Column for side-by-side violins at each tick.
        absolute: If True, plot |heading_change| instead of the signed value.
            The zero-reference line is hidden in this mode.
        ax_size: (width, height) per subplot.

    Returns:
        The matplotlib Figure.
    """
    value_fn = (
        (lambda r: abs(r["heading_change"]))
        if absolute
        else (lambda r: r["heading_change"])
    )
    return plot_violin_facets(
        responses,
        value_fn=value_fn,
        row_by=row_by,
        col_by=col_by,
        hue_by=hue_by,
        responsive_only=responsive_only,
        ylabel="|Heading change| (deg)" if absolute else "Heading change (deg)",
        title="Heading Change Distribution",
        show_zero_line=not absolute,
        ax_size=ax_size,
    )


_HEADING_METRICS = [
    ("heading_change",          "Net change\n(original)",          True),
    ("heading_change_window_net", "Net change\n(detection window)", True),
    ("heading_change_max_dev",  "Max deviation\nfrom baseline",    False),
    ("heading_change_post_saccade", "Post-saccade\nnet change",    True),
    ("heading_change_path_length",  "Path length\n(total rotation)", False),
]


def plot_heading_change_comparison(
    responses: list[Response],
    heading_threshold_deg: float = 30.0,
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """Compare five heading change metrics side-by-side, split by responsiveness.

    Each panel shows the distribution of one metric for responsive (orange) and
    non-responsive (grey) trials. A dashed line marks `heading_threshold_deg`
    on metrics where the threshold has a direct net-change meaning.

    Requires `classify_responsiveness` to have been called first (needs
    `is_responsive`). The four new heading metrics also require `r["heading"]`
    to be present (set by `extract_responses`); panels with all-NaN values
    are shown empty.

    Args:
        responses: Response list.
        heading_threshold_deg: Threshold line drawn on net-change panels.
        figsize: Overall figure size. Defaults to (18, 4).

    Returns:
        The matplotlib Figure.
    """
    if figsize is None:
        figsize = (18, 4)

    fig, axes = plt.subplots(1, len(_HEADING_METRICS), figsize=figsize, sharey=False)

    colors = {"responsive": "#E07B39", "non-responsive": "#7F9DBF"}

    for ax, (field, label, show_threshold) in zip(axes, _HEADING_METRICS):
        for responsive, color, grp_label in (
            (True,  colors["responsive"],     "responsive"),
            (False, colors["non-responsive"], "non-responsive"),
        ):
            vals = [
                v for r in responses
                if r.get("is_responsive") is responsive
                for v in [r.get(field)]
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            if vals:
                vp = ax.violinplot([vals], positions=[1 if responsive else 0],
                                   widths=0.6, showmeans=True, showmedians=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.75)
                for part in ("cbars", "cmins", "cmaxes", "cmeans"):
                    if part in vp:
                        vp[part].set_color(color)
                ax.text(
                    1 if responsive else 0, max(vals),
                    f"n={len(vals)}", ha="center", va="bottom", fontsize=7,
                )

        if show_threshold:
            ax.axhline(heading_threshold_deg, color="k", linestyle="--",
                       linewidth=0.8, alpha=0.6, label=f"{heading_threshold_deg}°")
            ax.axhline(-heading_threshold_deg, color="k", linestyle="--",
                       linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["non-resp.", "resp."], fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("degrees")
        ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor=colors["responsive"],     alpha=0.75, label="responsive"),
        Patch(facecolor=colors["non-responsive"], alpha=0.75, label="non-responsive"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Heading change metrics comparison", y=1.02)
    fig.tight_layout()
    return fig
