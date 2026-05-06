"""Sham vs real diagnostic trace plot."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._types import Response
from ._common import build_hue_colormap, iter_hue_subsets, prepare_ang_vel, unique_values


def plot_sham_vs_real(
    responses: list[Response],
    *,
    col_by: Optional[str] = "stimulus_offset_deg",
    hue_by: Optional[str] = "group",
    baseline_subtract: bool = True,
    sharey: bool = True,
    ax_size: tuple[float, float] = (5, 4),
) -> Figure:
    """Mean ± SD |ω| for sham vs real trials, overlaid per group and condition.

    Real trials are drawn as solid lines; sham trials as dotted lines with the
    same color per hue group.  The CLI calls this only when sham trials are present.

    Args:
        responses: Response list containing both sham (is_sham=True) and real trials.
        col_by: Field mapped to subplot columns (default "stimulus_offset_deg").
        hue_by: Field mapped to line colors (default "group").
        baseline_subtract: Subtract pre-stim baseline from |ω|.
        sharey: Share y-axis across subplots.
        ax_size: (width, height) per subplot.

    Returns:
        The matplotlib Figure.
    """
    real = [r for r in responses if not r.get("is_sham")]
    sham = [r for r in responses if r.get("is_sham")]

    if not sham:
        raise ValueError("No sham trials found in responses (is_sham not set or all False).")

    col_vals = unique_values(responses, col_by) or [None]
    n_cols = len(col_vals)

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(ax_size[0] * n_cols, ax_size[1]),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    color_map = build_hue_colormap(responses, hue_by)
    ylabel = "Δ Angular velocity (deg/s)" if baseline_subtract else "Angular velocity (deg/s)"

    for ci, col_val in enumerate(col_vals):
        ax = axes[0, ci]

        for dataset, linestyle, suffix in ((real, "-", "real"), (sham, ":", "sham")):
            col_subset = [
                r for r in dataset
                if col_by is None or r.get(col_by) == col_val
            ]
            if not col_subset:
                continue

            for hv, hue_subset in iter_hue_subsets(col_subset, hue_by):
                if not hue_subset:
                    continue
                time_axis = hue_subset[0]["time"]
                data = prepare_ang_vel(hue_subset, time_axis, baseline_subtract)
                with np.errstate(all="ignore"):
                    mean_tr = np.nanmean(data, axis=0)
                    std_tr = np.nanstd(data, axis=0)
                color = color_map.get(hv, "steelblue")
                lbl = (
                    f"{hv} {suffix} (n={len(hue_subset)})"
                    if hv is not None
                    else f"{suffix} (n={len(hue_subset)})"
                )
                ax.plot(time_axis, mean_tr, color=color, linestyle=linestyle, label=lbl)
                ax.fill_between(
                    time_axis,
                    mean_tr - std_tr,
                    mean_tr + std_tr,
                    alpha=0.12,
                    color=color,
                )

        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(0, color="k", linestyle=":", alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        if col_by is not None and col_val is not None:
            ax.set_title(f"{col_by} = {col_val}")
        ax.set_xlabel("Time relative to stimulus start (s)")
        if ci == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle("Sham vs Real response traces", y=1.02)
    fig.tight_layout()
    return fig
