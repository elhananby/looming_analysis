"""Screen position effect on angular velocity response."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .._types import Response
from ._common import build_hue_colormap, build_legend_patches, unique_values


def plot_screen_position_effect(
    responses: list[Response],
    *,
    hue_by: Optional[str] = "group",
    n_bins: int = 10,
    screen_width_px: int = 1920,
    responsive_only: bool = False,
    ax_size: tuple[float, float] = (10, 4),
) -> Figure:
    """Violin + jitter plot of angular velocity vs within-screen x position.

    The full display is ``screen_width_px * n_screens`` pixels wide. Each
    stimulus pixel_x is mapped to within-screen coordinates via modulo, then
    binned into ``n_bins`` equal-width bins. Two subplots are drawn side by
    side: peak and mean angular velocity in the response window.

    Args:
        responses: Response list. Responses missing ``pixel_x`` are skipped.
        hue_by: Field for color grouping (e.g. ``"group"``).
        n_bins: Number of equal-width bins across the screen width.
        screen_width_px: Width of one physical screen in pixels (default 1920).
        responsive_only: If True, restrict to responsive trials only.
        ax_size: Size of each individual subplot ``(width, height)`` in inches.
    """
    if responsive_only:
        responses = [r for r in responses if r.get("is_responsive")]

    valid = [r for r in responses if r.get("pixel_x") is not None]
    if not valid:
        fig, ax = plt.subplots(figsize=ax_size)
        ax.text(0.5, 0.5, "No pixel_x data available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    color_map = build_hue_colormap(responses, hue_by)
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]

    bin_width = screen_width_px / n_bins
    bin_centres = [int((i + 0.5) * bin_width) for i in range(n_bins)]

    metrics = [
        ("peak_ang_vel_deg_s", "Peak |ω| (deg/s)"),
        ("mean_ang_vel_window_deg_s", "Mean |ω| (deg/s)"),
    ]

    fig, axes = plt.subplots(
        len(metrics), 1,
        figsize=(ax_size[0], ax_size[1] * len(metrics)),
        sharex=True,
        sharey=False,
    )

    rng = np.random.default_rng(42)

    for ax, (field, ylabel) in zip(axes, metrics):
        x_positions = []
        x_labels = []

        for bin_idx in range(n_bins):
            group_xs = []
            for h_idx, hv in enumerate(hue_vals):
                subset = [
                    r for r in valid
                    if (hue_by is None or r.get(hue_by) == hv)
                    and r.get(field) is not None
                    and int(float(r["pixel_x"])) % screen_width_px
                    in range(int(bin_idx * bin_width), int((bin_idx + 1) * bin_width))
                ]
                vals = [float(r[field]) for r in subset
                        if not np.isnan(float(r[field]))]
                if not vals:
                    continue

                # position: bins spaced by (n_hues+0.5), hues offset within bin
                x_center = bin_idx * (len(hue_vals) + 0.8) + h_idx
                group_xs.append(x_center)

                color = color_map[hv]
                if len(vals) >= 4:
                    vp = ax.violinplot([vals], positions=[x_center], widths=0.7,
                                       showmeans=True, showextrema=False)
                    for pc in vp["bodies"]:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.5)
                    vp["cmeans"].set_color(color)

                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(
                    np.full(len(vals), x_center) + jitter,
                    vals,
                    color=color, alpha=0.5, s=10, zorder=3,
                )

            if group_xs:
                x_positions.append(np.mean(group_xs))
                x_labels.append(str(bin_centres[bin_idx]))

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_xlabel(f"Within-screen x position (px,  bin width={int(bin_width)}px)")

    if hue_by is not None:
        axes[0].legend(handles=build_legend_patches(color_map, hue_vals), title=hue_by,
                       bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle(
        f"Angular velocity vs screen position  "
        f"({'responsive only' if responsive_only else 'all trials'})",
        y=1.02,
    )
    fig.tight_layout()
    return fig
