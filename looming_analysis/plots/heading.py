"""Heading change distribution (violin) plots, faceted."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .._types import Response
from ._common import plot_violin_facets, unique_values


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
    ("heading_change", "Stim\n(circmean)", True),
    ("heading_change_peak_aligned", "Peak\n(circmean)", True),
    ("heading_change_stim_vector", "Stim\n(vector)", True),
    ("heading_change_peak_vector", "Peak\n(vector)", True),
    ("heading_change_rdp", "Peak\n(RDP)", True),
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
    n_polar_bins: int = 36,
    figsize: Optional[tuple[float, float]] = None,
) -> Figure:
    """Compare four heading change methods: violin plots (top) + polar histograms (bottom).

    Columns: Stim circmean | Peak circmean | Stim vector | Peak vector.
    Responsive flies use the full group color; non-responsive use a lighter tint.

    Args:
        responses: Response list.
        group_by: Field used to split into groups (default ``"group"``).
        heading_threshold_deg: Dashed threshold lines on violin panels.
        n_polar_bins: Number of angular bins in polar histograms (default 36 = 10 deg).
        figsize: Overall figure size. Defaults to ``(16, 10)``.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.colors as mcolors

    if figsize is None:
        figsize = (16, 10)

    n_metrics = len(_HEADING_METRICS)
    groups = unique_values(responses, group_by) or [None]
    base_colors = plt.cm.tab10(np.linspace(0, 0.9, len(groups)))
    color_map = {g: mcolors.to_hex(c) for g, c in zip(groups, base_colors)}

    fig = plt.figure(figsize=figsize)
    violin_axes = [
        fig.add_subplot(2, n_metrics, col + 1) for col in range(n_metrics)
    ]
    polar_axes = [
        fig.add_subplot(2, n_metrics, n_metrics + col + 1, projection="polar")
        for col in range(n_metrics)
    ]

    violin_width = 0.55
    group_spacing = 1.0
    responsive_offset = 0.35

    # ── Row 1: violin plots ───────────────────────────────────────────────────
    for ax, (field, label, show_threshold) in zip(violin_axes, _HEADING_METRICS):
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
            ax.axhline(heading_threshold_deg, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axhline(-heading_threshold_deg, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("degrees")
        ax.grid(True, axis="y", alpha=0.3)

    # ── Row 2: polar histograms ───────────────────────────────────────────────
    bin_edges = np.linspace(-np.pi, np.pi, n_polar_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ax, (field, label, _) in zip(polar_axes, _HEADING_METRICS):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        for gi, grp in enumerate(groups):
            base_col = color_map[grp]
            light_col = _lighten(base_col)

            for responsive, face_col in ((True, base_col), (False, light_col)):
                subset = [
                    r
                    for r in responses
                    if (group_by is None or r.get(group_by) == grp)
                    and r.get("is_responsive") is responsive
                ]
                vals_deg = [
                    r[field]
                    for r in subset
                    if r.get(field) is not None
                    and not (isinstance(r[field], float) and np.isnan(r[field]))
                ]
                if not vals_deg:
                    continue
                vals_rad = np.deg2rad(vals_deg)
                counts, _ = np.histogram(vals_rad, bins=bin_edges)
                ax.bar(bin_centers, counts, width=bin_width, color=face_col, alpha=0.6)

        ax.set_title(label, fontsize=9, pad=12)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    for grp in groups:
        col = color_map[grp]
        lbl = str(grp) if grp is not None else "all"
        legend_handles.append(Patch(facecolor=col, alpha=0.85, label=f"{lbl} (resp.)"))
        legend_handles.append(Patch(facecolor=_lighten(col), alpha=0.85, label=f"{lbl} (non-resp.)"))

    fig.legend(handles=legend_handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    fig.suptitle("Heading change method comparison", y=1.01)
    fig.tight_layout()
    return fig


def plot_heading_changes_polar(
    responses: list[Response],
    *,
    hue_by: Optional[str] = None,
    col_by: Optional[str] = None,
    field: str = "heading_change",
    n_bins: int = 36,
    responsive_only: bool = False,
    ax_size: tuple[float, float] = (5, 5),
) -> Figure:
    """Polar histogram of heading changes with 0 degrees at north (top).

    Args:
        responses: Response list.
        hue_by: Field for color grouping (each group drawn as overlapping bars).
        col_by: Field for subplot columns (one polar axis per unique value).
        field: Response key to plot (default ``"heading_change"``).
        n_bins: Number of angular bins (default 36 = 10 deg per bin).
        responsive_only: If True, include only trials where ``is_responsive`` is True.
        ax_size: (width, height) per subplot in inches.

    Returns:
        The matplotlib Figure.
    """
    from ._common import build_hue_colormap, unique_values

    if responsive_only:
        responses = [r for r in responses if r.get("is_responsive")]

    col_vals = (unique_values(responses, col_by) if col_by else None) or [None]
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]
    color_map = build_hue_colormap(responses, hue_by)

    n_cols = len(col_vals)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(ax_size[0] * n_cols, ax_size[1]),
        subplot_kw={"projection": "polar"},
    )
    if n_cols == 1:
        axes = [axes]

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ax, col_val in zip(axes, col_vals):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        for hue_val in hue_vals:
            subset = responses
            if col_by and col_val is not None:
                subset = [r for r in subset if r.get(col_by) == col_val]
            if hue_by and hue_val is not None:
                subset = [r for r in subset if r.get(hue_by) == hue_val]

            vals_deg = [
                r[field]
                for r in subset
                if r.get(field) is not None
                and not (isinstance(r[field], float) and np.isnan(r[field]))
            ]
            if not vals_deg:
                continue

            vals_rad = np.deg2rad(vals_deg)
            counts, _ = np.histogram(vals_rad, bins=bin_edges)
            color = color_map.get(hue_val, "steelblue") if hue_by else "steelblue"
            label = str(hue_val) if hue_val is not None else None
            ax.bar(
                bin_centers,
                counts,
                width=bin_width,
                color=color,
                alpha=0.6,
                label=label,
            )

        title = f"{col_val}" if col_val is not None else "All"
        ax.set_title(title, pad=12)

        if hue_by and any(h is not None for h in hue_vals):
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.suptitle(f"Polar distribution: {field}", y=1.02)
    fig.tight_layout()
    return fig
