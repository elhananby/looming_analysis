"""Shared plotting helpers: faceting, hue colors, baseline subtraction."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .._types import Response


def add_stats_box(ax, lines: list[str], *, loc: str = "upper left") -> None:
    """Render compact per-group summary stats without expanding the axes."""
    if not lines:
        return

    positions = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "0.8",
        },
    )


def unique_values(responses: list[Response], key: Optional[str]) -> list[Any]:
    """Return sorted unique values for `key` across responses (skips None/missing)."""
    if key is None:
        return []
    return sorted({r.get(key) for r in responses if r.get(key) is not None})


def effective_axis(responses: list[Response], key: Optional[str]) -> tuple[list, int]:
    """Return (values_list, n) where values_list is always non-empty.

    Returns ([None], 1) when `key` is None or no values are present.
    Otherwise returns (sorted_unique_values, len).
    """
    vals = unique_values(responses, key)
    if not vals:
        return [None], 1
    return vals, len(vals)


def iter_facets(
    responses: list[Response],
    row_by: Optional[str],
    col_by: Optional[str],
) -> Iterator[tuple[Any, Any, list[Response], tuple[int, int, int, int]]]:
    """Yield one tuple per facet cell.

    Each yield is `(row_val, col_val, subset, (row_idx, col_idx, n_rows, n_cols))`.
    When a dimension is `None`, its value is `None` and the grid collapses to 1.

    Args:
        responses: Response list.
        row_by: Column name for subplot rows (or None).
        col_by: Column name for subplot columns (or None).
    """
    row_vals = unique_values(responses, row_by)
    col_vals = unique_values(responses, col_by)
    n_rows = len(row_vals) if row_vals else 1
    n_cols = len(col_vals) if col_vals else 1
    effective_rows = row_vals if row_vals else [None]
    effective_cols = col_vals if col_vals else [None]

    for ri, rv in enumerate(effective_rows):
        for ci, cv in enumerate(effective_cols):
            subset = [
                r
                for r in responses
                if (row_by is None or r.get(row_by) == rv)
                and (col_by is None or r.get(col_by) == cv)
            ]
            yield rv, cv, subset, (ri, ci, n_rows, n_cols)


def iter_hue_subsets(
    responses: list[Response],
    hue_by: Optional[str],
) -> Iterator[tuple[Any, list[Response]]]:
    """Yield ``(hue_val, subset)`` pairs for every distinct value of *hue_by*.

    When *hue_by* is ``None``, yields a single ``(None, responses)`` pair.
    """
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]
    for hv in hue_vals:
        subset = (
            responses
            if hue_by is None
            else [r for r in responses if r.get(hue_by) == hv]
        )
        yield hv, subset


def build_legend_patches(
    color_map: dict,
    hue_vals: list,
    alpha: float = 0.7,
) -> list:
    """Build a list of ``Patch`` handles for a hue legend."""
    from matplotlib.patches import Patch

    return [Patch(facecolor=color_map[hv], alpha=alpha, label=str(hv)) for hv in hue_vals]


def build_hue_colormap(
    responses: list[Response],
    hue_by: Optional[str],
    *,
    light: bool = False,
) -> dict[Any, Any]:
    """Build a {hue_value: rgba-color} mapping. Single-entry when hue_by is None.

    Uses discrete tab10 indices so successive groups get maximally distinct
    colours (blue, orange, green, red, purple, …).  Pass ``light=True`` to
    get washed-out variants suitable for a secondary series (e.g. non-responsive).
    """
    if hue_by is None:
        base = np.array(plt.cm.tab10(0))
        if light:
            base = np.clip(base * 0.4 + 0.6, 0, 1)
        return {None: tuple(base)}
    hue_vals = unique_values(responses, hue_by)
    result = {}
    for i, hv in enumerate(hue_vals):
        color = np.array(plt.cm.tab10(i % 10))
        if light:
            color = np.clip(color * 0.4 + 0.6, 0, 1)
        result[hv] = tuple(color)
    return result


def grouped_offsets(n: int, width: float = 0.8) -> tuple[np.ndarray, float]:
    """Evenly-spaced per-group x-offsets that fit within *width*.

    Returns ``(offsets, bar_width)`` where *offsets* has shape ``(n,)`` and
    ``offsets[i]`` is the x-shift for group *i* relative to the tick centre.
    """
    bar_width = width / max(n, 1)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bar_width
    return offsets, bar_width


def prepare_ang_vel(
    responses: list[Response],
    time_axis: np.ndarray,
    baseline_subtract: bool,
) -> np.ndarray:
    """Stack trials into `|angular velocity|` array (deg/s), optionally baseline-subtracted.

    When `baseline_subtract` is True, each trial's mean `|ω|` during the
    pre-stimulus window (`time < 0`) is subtracted from that trial's trace.
    """
    data_abs_deg = np.abs(np.rad2deg(np.stack([r["ang_vel"] for r in responses])))
    if baseline_subtract:
        pre_mask = time_axis < 0
        if pre_mask.any():
            with np.errstate(invalid="ignore"):
                baseline = np.nanmean(data_abs_deg[:, pre_mask], axis=1, keepdims=True)
            data_abs_deg = data_abs_deg - baseline
    return data_abs_deg


def plot_violin_facets(
    responses: list[Response],
    *,
    value_fn: Callable[[Response], float | None],
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = None,
    responsive_only: bool = False,
    ylabel: str,
    title: str,
    show_zero_line: bool = True,
    ax_size: tuple[float, float] = (7, 5),
) -> "plt.Figure":
    """Generic faceted violin plot.

    Each violin shows the distribution of `value_fn(response)` for trials
    matching one (row_val, col_val, hue_val) cell. NaN values are dropped.
    """
    if responsive_only:
        responses = [r for r in responses if r.get("is_responsive")]

    effective_rows, n_rows = effective_axis(responses, row_by)
    effective_x, _ = effective_axis(responses, col_by)
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]

    width = max(ax_size[0], 1.0 * max(len(effective_x), 1) * max(len(hue_vals), 1))
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(width, ax_size[1] * n_rows), squeeze=False
    )
    color_map = build_hue_colormap(responses, hue_by)

    violin_width = 0.7
    gap = 1.0

    for row_idx, row_val in enumerate(effective_rows):
        ax = axes[row_idx, 0]
        x = 0.0
        tick_positions: list[float] = []
        tick_labels: list[str] = []

        for xv in effective_x:
            group_start = x
            for hv in hue_vals:
                subset = [
                    r
                    for r in responses
                    if (row_by is None or r.get(row_by) == row_val)
                    and (col_by is None or r.get(col_by) == xv)
                    and (hue_by is None or r.get(hue_by) == hv)
                ]
                vals = [
                    v
                    for r in subset
                    for v in [value_fn(r)]
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                ]
                if vals:
                    vp = ax.violinplot(
                        [vals], positions=[x], widths=violin_width, showmeans=True
                    )
                    for pc in vp["bodies"]:
                        pc.set_facecolor(color_map[hv])
                        pc.set_alpha(0.7)
                    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
                        if partname in vp:
                            vp[partname].set_color(color_map[hv])
                    ax.text(
                        x,
                        max(vals),
                        f"n={len(vals)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                x += 1
            tick_positions.append((group_start + x - 1) / 2)
            tick_labels.append(str(xv) if xv is not None else "all")
            x += gap

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        if row_idx == 0 and hue_by is not None:
            ax.legend(
                handles=build_legend_patches(color_map, hue_vals),
                title=hue_by,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )
        if row_by is not None:
            ax.set_ylabel(f"{row_by} = {row_val}\n{ylabel}")
        else:
            ax.set_ylabel(ylabel)
        if row_idx == n_rows - 1 and col_by is not None:
            ax.set_xlabel(col_by)
        if show_zero_line:
            ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")

    dims = [
        f"{k}={v}" for k, v in [("rows", row_by), ("x", col_by), ("hue", hue_by)] if v
    ]
    fig.suptitle(title + ("  |  " + ", ".join(dims) if dims else ""), y=1.02)
    fig.tight_layout()
    return fig


def annotate_facet(
    ax,
    row_val: Any,
    col_val: Any,
    row_by: Optional[str],
    col_by: Optional[str],
    position: tuple[int, int, int, int],
    base_ylabel: str,
    xlabel: str = "Time relative to stimulus start (s)",
) -> None:
    """Apply standard facet-grid axis titles/labels.

    Top row gets column titles, left column gets row-prefixed y-labels,
    bottom row gets x-labels.
    """
    row_idx, col_idx, n_rows, n_cols = position
    is_top = row_idx == 0
    is_left = col_idx == 0
    is_bottom = row_idx == n_rows - 1

    if is_top and col_by is not None:
        ax.set_title(f"{col_by} = {col_val}")
    if is_left:
        if row_by is not None:
            ax.set_ylabel(f"{row_by} = {row_val}\n{base_ylabel}")
        else:
            ax.set_ylabel(base_ylabel)
    if is_bottom:
        ax.set_xlabel(xlabel)


def require_responsiveness(responses: list[Response]) -> None:
    if responses and "is_responsive" not in responses[0]:
        raise ValueError(
            "Responses do not include 'is_responsive'. "
            "Call classify_responsiveness(responses) before this plot."
        )


def suptitle(fig: Figure, row_by, col_by, hue_by, *, prefix: str) -> None:
    parts = [prefix]
    dims = []
    if row_by:
        dims.append(f"rows={row_by}")
    if col_by:
        dims.append(f"cols={col_by}")
    if hue_by:
        dims.append(f"hue={hue_by}")
    if dims:
        parts.append("  |  " + ", ".join(dims))
    fig.suptitle("".join(parts), y=1.02)
