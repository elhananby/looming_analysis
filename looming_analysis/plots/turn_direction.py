"""Turn-direction proportion plots (stacked bar), faceted."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .._types import Response
from ._common import effective_axis, filter_real

_LEFT_COLOR = "#4878D0"
_RIGHT_COLOR = "#EE854A"


def plot_turn_proportions(
    responses: list[Response],
    *,
    col_by: str = "stimulus_offset_deg",
    row_by: Optional[str] = None,
    group_by: Optional[str] = None,
    x_by: Optional[str] = None,
    responsive_only: bool = False,
    exclude_sham: bool = True,
    ax_size: tuple[float, float] = (6, 4),
) -> Figure:
    """100 % stacked bar chart of turn direction (left / right) per condition.

    Each bar represents one value of ``col_by`` and shows the percentage of
    trials where the fly turned left or right. Trials with ``turn_direction = None``
    are excluded from the percentage but counted in the ``n=`` label.

    Call ``compute_turn_direction`` on your responses before using this plot.

    Args:
        responses: Response list — each must have ``turn_direction`` set.
        col_by: Column mapped to x-axis bar groups (default ``stimulus_offset_deg``).
        row_by: Column for subplot rows (None → single row).
        group_by: Column for subplot columns (None → single column).
        x_by: Deprecated alias for ``col_by``.
        responsive_only: If True, only include trials where ``is_responsive`` is True.
        ax_size: (width, height) per subplot in inches.

    Returns:
        The matplotlib Figure.
    """
    if x_by is not None:
        warnings.warn("x_by is deprecated, use col_by instead", DeprecationWarning, stacklevel=2)
        col_by = x_by

    if not responses:
        raise ValueError("responses list is empty.")
    if "turn_direction" not in responses[0]:
        raise ValueError(
            "Responses do not include 'turn_direction'. "
            "Call compute_turn_direction(responses) before plotting turn proportions."
        )

    if exclude_sham:
        responses = filter_real(responses)
    if responsive_only:
        responses = [r for r in responses if r.get("is_responsive")]

    effective_x, _ = effective_axis(responses, col_by)
    effective_rows, n_rows = effective_axis(responses, row_by)
    effective_groups, n_groups = effective_axis(responses, group_by)

    fig, axes = plt.subplots(
        n_rows,
        n_groups,
        figsize=(ax_size[0] * n_groups, ax_size[1] * n_rows),
        squeeze=False,
        sharey=True,
    )

    legend_handles = [
        Patch(facecolor=_LEFT_COLOR, label="Left"),
        Patch(facecolor=_RIGHT_COLOR, label="Right"),
    ]

    for ri, rv in enumerate(effective_rows):
        for gi, gv in enumerate(effective_groups):
            ax = axes[ri, gi]
            subset = [
                r
                for r in responses
                if (row_by is None or r.get(row_by) == rv)
                and (group_by is None or r.get(group_by) == gv)
            ]
            _draw_stacked_bars(ax, subset, col_by, effective_x)

            if ri == 0 and group_by is not None:
                ax.set_title(f"{group_by} = {gv}")
            if gi == 0:
                ylabel = "% trials"
                if row_by is not None:
                    ax.set_ylabel(f"{row_by} = {rv}\n{ylabel}")
                else:
                    ax.set_ylabel(ylabel)
            if ri == n_rows - 1:
                ax.set_xlabel(col_by)

    axes[0, -1].legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    title = "Turn Direction Proportions"
    dims = [
        f"{k}={v}" for k, v in [("x", col_by), ("rows", row_by), ("groups", group_by)] if v
    ]
    if dims:
        title += "  |  " + ", ".join(dims)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def _draw_stacked_bars(ax, subset: list[Response], col_by: str, x_vals: list) -> None:
    positions = np.arange(len(x_vals))
    bar_width = 0.6

    for pos, xv in zip(positions, x_vals):
        trials = [r for r in subset if r.get(col_by) == xv]
        n_total = len(trials)
        n_left = sum(1 for r in trials if r.get("turn_direction") == "left")
        n_right = sum(1 for r in trials if r.get("turn_direction") == "right")
        n_valid = n_left + n_right

        if n_valid == 0:
            ax.bar(pos, 100, width=bar_width, color="lightgrey")
            ax.text(pos, 101, f"n={n_total}", ha="center", va="bottom", fontsize=8)
            continue

        pct_left = 100.0 * n_left / n_valid
        pct_right = 100.0 * n_right / n_valid

        ax.bar(pos, pct_left, width=bar_width, color=_LEFT_COLOR, label="Left")
        ax.bar(pos, pct_right, width=bar_width, bottom=pct_left, color=_RIGHT_COLOR, label="Right")
        ax.text(pos, 101, f"n={n_total}", ha="center", va="bottom", fontsize=8)
        if pct_left >= 10:
            ax.text(pos, pct_left / 2, f"{pct_left:.0f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if pct_right >= 10:
            ax.text(pos, pct_left + pct_right / 2, f"{pct_right:.0f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.axhline(50, color="k", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.set_ylim(0, 115)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(xv) if xv is not None else "all" for xv in x_vals])
    ax.set_yticks([0, 25, 50, 75, 100])
