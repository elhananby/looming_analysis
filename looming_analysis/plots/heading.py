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
    ("heading_change", "Net change\n(original)", True),
    ("heading_change_window_net", "Net change\n(detection window)", True),
    ("heading_change_max_dev", "Max deviation\nfrom baseline", False),
    ("heading_change_post_saccade", "Post-saccade\nnet change", True),
    ("heading_change_path_length", "Path length\n(total rotation)", False),
]


def _lighten(color, factor: float = 0.55) -> tuple:
    """Blend *color* toward white by *factor* (0 = original, 1 = white)."""
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * factor)


def plot_heading_change_comparison(
    responses: list[Response],
    group_by: str = "group",
    heading_threshold_deg: float = 30.0,
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """Compare five heading change metrics across experimental groups.

    Each panel shows one metric. Within each panel, groups are arranged
    side-by-side. Responsive flies use the full group color; non-responsive
    flies use a lighter tint of the same color. A dashed line marks
    ``heading_threshold_deg`` on net-change panels.

    Requires ``classify_responsiveness`` to have been called first. The four
    new heading metrics also require ``r["heading"]`` (set by
    ``extract_responses``); panels with all-NaN values are left empty.

    Args:
        responses: Response list.
        group_by: Response field used to split into groups (default ``"group"``).
            Falls back to a single unlabelled group when the field is absent.
        heading_threshold_deg: Threshold line drawn on net-change panels.
        figsize: Overall figure size. Defaults to ``(18, 5)``.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.colors as mcolors

    if figsize is None:
        figsize = (18, 5)

    from ._common import unique_values

    groups = unique_values(responses, group_by) or [None]
    base_colors = plt.cm.tab10(np.linspace(0, 0.9, len(groups)))
    color_map = {g: mcolors.to_hex(c) for g, c in zip(groups, base_colors)}

    violin_width = 0.55
    group_spacing = 1.0  # gap between groups
    responsive_offset = 0.35  # responsive violin offset within a group

    fig, axes = plt.subplots(1, len(_HEADING_METRICS), figsize=figsize, sharey=False)

    for ax, (field, label, show_threshold) in zip(axes, _HEADING_METRICS):
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
                    r
                    for r in responses
                    if (group_by is None or r.get(group_by) == grp)
                    and r.get("is_responsive") is responsive
                ]
                vals = [
                    v
                    for r in subset
                    for v in [r.get(field)]
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                ]
                if not vals:
                    continue
                pos = x_center + x_off
                vp = ax.violinplot(
                    [vals],
                    positions=[pos],
                    widths=violin_width,
                    showmeans=True,
                    showmedians=False,
                )
                for pc in vp["bodies"]:
                    pc.set_facecolor(face_col)
                    pc.set_alpha(0.85)
                for part in ("cbars", "cmins", "cmaxes", "cmeans"):
                    if part in vp:
                        vp[part].set_color(face_col)
                ax.text(
                    pos,
                    max(vals),
                    f"n={len(vals)}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

            tick_positions.append(x_center)
            tick_labels.append(str(grp) if grp is not None else "all")

        if show_threshold:
            ax.axhline(
                heading_threshold_deg,
                color="k",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )
            ax.axhline(
                -heading_threshold_deg,
                color="k",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )
        ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("degrees")
        ax.grid(True, axis="y", alpha=0.3)

    # Legend: one entry per group (full color = responsive, light = non-responsive)
    legend_handles = []
    for grp in groups:
        col = color_map[grp]
        lbl = str(grp) if grp is not None else "all"
        legend_handles.append(Patch(facecolor=col, alpha=0.85, label=f"{lbl} (resp.)"))
        legend_handles.append(
            Patch(facecolor=_lighten(col), alpha=0.85, label=f"{lbl} (non-resp.)")
        )

    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
    )
    fig.suptitle("Heading change metrics comparison", y=1.02)
    fig.tight_layout()
    return fig
