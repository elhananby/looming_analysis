"""Peak-aligned angular velocity traces."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .._types import DT_SECONDS, Response
from ._common import annotate_facet, build_hue_colormap, iter_facets, unique_values


def _get_peak_time_s(
    r: Response,
    fallback_window_ms: float,
) -> tuple[float, bool]:
    """Return (peak_time_s, is_detected) for a single response.

    Priority:
    1. saccade_peak_time_ms — set by classify_responsiveness when a qualifying
       saccade is found.
    2. argmax of |ω| in the detection window around end_expansion_time.
    3. end_expansion_time itself (last resort when window is empty/all-NaN).

    Returns is_detected=True only for case 1.
    """
    spt = r.get("saccade_peak_time_ms")
    if spt is not None and not np.isnan(float(spt)):
        return float(spt) / 1000.0, True

    time = r["time"]
    ang_vel_abs = np.abs(np.rad2deg(r["ang_vel"]))
    end_t = float(r.get("end_expansion_time", 0.0))
    hw_s = fallback_window_ms / 1000.0
    mask = (time >= end_t - hw_s) & (time <= end_t + hw_s)
    vals = ang_vel_abs[mask]
    if vals.size > 0 and not np.all(np.isnan(vals)):
        peak_local = int(np.nanargmax(vals))
        peak_global = int(np.where(mask)[0][peak_local])
        return float(time[peak_global]), False

    return end_t, False


def compute_peak_latency(
    responses: list[Response],
    fallback_window_ms: float = 200.0,
) -> list[Response]:
    """Add ``peak_latency_ms`` to each response dict (in-place).

    For trials where a qualifying saccade was detected by
    ``classify_responsiveness``, the value equals ``saccade_peak_time_ms``
    (time from stimulus onset to saccade peak, in ms).  For all other trials
    the value is the time of the |ω| argmax within the detection window, or
    ``NaN`` when no data are available.

    Calling this function is optional — ``plot_peak_aligned_traces`` runs it
    automatically and stores the results.  Call it explicitly beforehand if you
    want ``peak_latency_ms`` available in ``to_dataframe()`` output.

    Args:
        responses: Response list.  Modified in-place.
        fallback_window_ms: Window half-width (ms) around ``end_expansion_time``
            used to locate a fallback peak for non-responsive trials.

    Returns:
        The same list (for chaining).
    """
    for r in responses:
        peak_t_s, _ = _get_peak_time_s(r, fallback_window_ms)
        r["peak_latency_ms"] = float(peak_t_s * 1000.0)
    return responses


def _extract_aligned(
    r: Response,
    half_window_ms: float,
    fallback_window_ms: float,
    n_samples: int,
    dt_s: float,
) -> np.ndarray:
    """Return |ω| trace of length n_samples centered on the peak (NaN-padded)."""
    time = r["time"]
    ang_vel_abs = np.abs(np.rad2deg(r["ang_vel"]))

    peak_t_s, _ = _get_peak_time_s(r, fallback_window_ms)

    center_idx = int(np.argmin(np.abs(time - peak_t_s)))
    half_n = n_samples // 2

    out = np.full(n_samples, np.nan)
    src_start = max(0, center_idx - half_n)
    src_end = min(len(ang_vel_abs), center_idx + half_n + 1)
    dst_start = src_start - (center_idx - half_n)
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = ang_vel_abs[src_start:src_end]
    return out


def _draw_peak_aligned(
    ax,
    responses: list[Response],
    time_axis: np.ndarray,
    hue_by: Optional[str],
    color_map: dict,
    half_window_ms: float,
    fallback_window_ms: float,
    n_samples: int,
    dt_s: float,
) -> None:
    hue_vals: list = [None] if hue_by is None else unique_values(responses, hue_by)

    for hv in hue_vals:
        subset = (
            responses
            if hue_by is None
            else [r for r in responses if r.get(hue_by) == hv]
        )
        if not subset:
            continue

        traces = np.stack(
            [
                _extract_aligned(r, half_window_ms, fallback_window_ms, n_samples, dt_s)
                for r in subset
            ]
        )
        with np.errstate(all="ignore"):
            mean_tr = np.nanmean(traces, axis=0)
            sem_tr = np.nanstd(traces, axis=0)

        color = color_map.get(hv, "steelblue")
        label = f"{hv} (n={len(subset)})" if hv is not None else f"n={len(subset)}"
        ax.plot(time_axis, mean_tr, color=color, label=label)
        ax.fill_between(
            time_axis,
            mean_tr - sem_tr,
            mean_tr + sem_tr,
            alpha=0.2,
            color=color,
        )

    ax.axvline(0, color="k", linestyle="--", alpha=0.5, label="peak")
    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def plot_peak_aligned_traces(
    responses: list[Response],
    *,
    half_window_ms: float = 100.0,
    fallback_window_ms: float = 200.0,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    hue_by: Optional[str] = "group",
    responsive_only: bool = False,
    sharey: bool = True,
    ax_size: tuple[float, float] = (5, 4),
) -> Figure:
    """Mean ± SEM |ω| traces aligned to each trial's peak response.

    For trials with a detected saccade (``saccade_peak_time_ms`` set by
    ``classify_responsiveness``), traces are centered on that saccade peak.
    For all other trials the peak is taken as the argmax of ``|ω|`` within
    ``±fallback_window_ms`` of ``end_expansion_time``.  This keeps non-
    responsive trials in the average while aligning them to their best-guess
    response location.

    As a side effect, ``peak_latency_ms`` is added to every response dict
    (time from stimulus onset to the peak used for alignment, in ms).  Call
    ``compute_peak_latency()`` explicitly beforehand if you need the field
    before plotting.

    Args:
        responses: Response list.  ``classify_responsiveness`` should have been
            called first so that saccade peaks are available.
        half_window_ms: Half-width of the extracted window (ms).  Default 100 ms
            gives a ±100 ms window (200 ms total) around the peak.
        fallback_window_ms: Window half-width (ms) used to locate the |ω| peak
            for trials without a detected saccade.
        row_by: Field mapped to subplot rows.
        col_by: Field mapped to subplot columns.
        hue_by: Field mapped to line colors.
        responsive_only: If True, restrict to trials where ``is_responsive`` is
            True.
        sharey: Share y-axis across all subplots.
        ax_size: (width, height) per subplot in inches.

    Returns:
        The matplotlib Figure.
    """
    dt_s = DT_SECONDS
    n_samples = int(round(2 * half_window_ms / (dt_s * 1000))) + 1
    time_axis = np.linspace(-half_window_ms, half_window_ms, n_samples) / 1000.0

    if responsive_only:
        responses = [r for r in responses if r.get("is_responsive")]

    # Store peak_latency_ms on every response.
    compute_peak_latency(responses, fallback_window_ms=fallback_window_ms)

    row_vals = unique_values(responses, row_by)
    col_vals = unique_values(responses, col_by)
    n_rows = len(row_vals) if row_vals else 1
    n_cols = len(col_vals) if col_vals else 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(ax_size[0] * n_cols, ax_size[1] * n_rows),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    color_map = build_hue_colormap(responses, hue_by)
    ylabel = "|Angular velocity| (deg/s)"

    for row_val, col_val, subset, position in iter_facets(responses, row_by, col_by):
        ax = axes[position[0], position[1]]
        if subset:
            _draw_peak_aligned(
                ax,
                subset,
                time_axis,
                hue_by,
                color_map,
                half_window_ms,
                fallback_window_ms,
                n_samples,
                dt_s,
            )
        annotate_facet(
            ax, row_val, col_val, row_by, col_by, position, ylabel,
            xlabel="Time relative to peak (s)",
        )

    _suptitle(fig, row_by, col_by, hue_by)
    fig.tight_layout()
    return fig


def _suptitle(fig: Figure, row_by, col_by, hue_by) -> None:
    parts = ["Peak-aligned angular velocity"]
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


def plot_response_latency(
    responses: list[Response],
    *,
    hue_by: Optional[str] = "group",
    bins: int = 40,
    responsive_only: bool = True,
    fallback_window_ms: float = 200.0,
    ax_size: tuple[float, float] = (7, 4),
) -> Figure:
    """Histogram of response latency (time from stimulus onset to response peak).

    Latency is read from ``peak_latency_ms`` if already set on each response
    (e.g. by a prior call to ``plot_peak_aligned_traces`` or
    ``compute_peak_latency``).  Otherwise it is computed on the fly.

    For responsive trials the latency equals ``saccade_peak_time_ms``; for
    non-responsive trials it is the argmax of ``|ω|`` in the fallback window.
    Use ``responsive_only=True`` (default) to restrict the histogram to trials
    where a saccade was actually detected.

    Args:
        responses: Response list.
        hue_by: Field mapped to color grouping.
        bins: Number of histogram bins.
        responsive_only: If True (default), only include trials where
            ``is_responsive`` is True and ``saccade_peak_time_ms`` is finite.
        fallback_window_ms: Passed to ``compute_peak_latency`` when
            ``peak_latency_ms`` is not already set.
        ax_size: Figure size ``(width, height)`` in inches.
    """
    # Ensure peak_latency_ms is present.
    compute_peak_latency(responses, fallback_window_ms=fallback_window_ms)

    fig, ax = plt.subplots(figsize=ax_size)
    color_map = build_hue_colormap(responses, hue_by)
    hue_vals = unique_values(responses, hue_by) if hue_by else [None]

    legend_handles = []
    for hv in hue_vals:
        subset = [
            r for r in responses
            if (hue_by is None or r.get(hue_by) == hv)
        ]
        if responsive_only:
            subset = [
                r for r in subset
                if r.get("is_responsive")
                and r.get("saccade_peak_time_ms") is not None
                and not np.isnan(float(r["saccade_peak_time_ms"]))
            ]
        latencies = np.array(
            [float(r["peak_latency_ms"]) for r in subset
             if r.get("peak_latency_ms") is not None
             and not np.isnan(float(r["peak_latency_ms"]))],
            dtype=float,
        )
        if latencies.size == 0:
            continue

        color = color_map[hv]
        ax.hist(latencies, bins=bins, color=color, alpha=0.45, density=True)

        p25, p50, p75 = (float(np.percentile(latencies, p)) for p in (25, 50, 75))
        mean_val = float(np.mean(latencies))

        ax.axvspan(p25, p75, color=color, alpha=0.12)
        ax.axvline(p50, color=color, linestyle="-", linewidth=1.5)
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.5)

        label = f"{hv if hv is not None else 'all'}"
        label += f"\n  mean={mean_val:.0f} ms  median={p50:.0f} ms  IQR=[{p25:.0f}, {p75:.0f}] ms  n={latencies.size}"
        legend_handles.append(Patch(facecolor=color, alpha=0.6, label=label))

    legend_handles += [
        Line2D([0], [0], color="grey", linestyle="-", linewidth=1.5, label="median"),
        Line2D([0], [0], color="grey", linestyle="--", linewidth=1.5, label="mean"),
        Patch(facecolor="grey", alpha=0.2, label="IQR (25–75th pct)"),
    ]

    qualifier = "responsive trials" if responsive_only else "all trials"
    ax.set_title(f"Response latency distribution  ({qualifier})")
    ax.set_xlabel("Latency: stimulus onset → peak (ms)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3, axis="y")
    if legend_handles:
        ax.legend(handles=legend_handles, title=hue_by, loc="upper right",
                  framealpha=0.9, fontsize=8)

    fig.tight_layout()
    return fig
