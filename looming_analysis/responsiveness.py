"""Classify trials as responsive based on angular velocity metrics."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ._types import DT_SECONDS, Response, _circ_diff_deg
from .extract import _compute_heading_change_vector

_PEAK_KW = dict(height=0.0, prominence=300, width=(3, None), distance=5)
_PEAK_ALIGNED_REF_FRAMES = 10  # half-window around detected peak for peak-aligned heading metrics

RESPONSIVENESS_METHOD_FIELDS = {
    "peak": "is_responsive_peak",
    "zscore": "is_responsive_zscore",
    "heading": "is_responsive_heading",
    "saccade": "is_responsive_saccade",
    "impulse": "is_responsive_impulse",
    "combined": "is_responsive_combined",
}


def _reaction_window_seconds(
    window_ms: float | tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Return (before_s, after_s) for a scalar or asymmetric ms window."""
    if isinstance(window_ms, (tuple, list)):
        if len(window_ms) != 2:
            raise ValueError("window_ms must be a number or a two-item sequence.")
        before_ms, after_ms = window_ms
    else:
        before_ms = after_ms = window_ms

    before = float(before_ms)
    after = float(after_ms)
    if before < 0 or after < 0:
        raise ValueError("window_ms values must be >= 0.")
    return before / 1000.0, after / 1000.0


_EPS = 1e-9


def compute_turn_direction(
    responses: list[Response],
    start_offset_s: float = 0.0,
    end_offset_s: float = 0.0,
) -> list[Response]:
    """Add signed peak angular velocity and turn direction to each response (in-place).

    For responsive trials the sign is taken from ``saccade_peak_ang_vel_signed_deg_s``.
    For non-responsive trials the angular velocity is sampled at the fallback reference
    index set by ``classify_responsiveness`` (population-mean saccade peak time), so
    both ``heading_change`` and ``signed_peak_ang_vel_deg_s`` are coherent.

    The ``_fallback_ref_idx`` key is removed from each response after use.

    Fields added to each response dict:
        signed_peak_ang_vel_deg_s (float): Signed ω at the reference index (deg/s).
        turn_direction (str | None): "left" (ω < 0), "right" (ω > 0), or None (≈0).

    Args:
        responses: List of response dicts from ``extract_responses``.
        start_offset_s: Used only when both classify_responsiveness has not been called
            and _fallback_ref_idx is absent (raw window fallback).
        end_offset_s: Same as above.

    Returns:
        The same list (for chaining).
    """
    for r in responses:
        saccade_signed = r.get("saccade_peak_ang_vel_signed_deg_s")
        if saccade_signed is not None and not np.isnan(saccade_signed):
            r["signed_peak_ang_vel_deg_s"] = saccade_signed
            r["turn_direction"] = "right" if saccade_signed > 0 else "left"
            r.pop("_fallback_ref_idx", None)
            continue

        # Non-responsive: use fallback ref index from classify_responsiveness.
        fallback_ref = r.pop("_fallback_ref_idx", None)
        if fallback_ref is not None:
            ang_vel_deg = np.rad2deg(r["ang_vel"])
            omega = float(ang_vel_deg[fallback_ref])
            r["signed_peak_ang_vel_deg_s"] = omega
            if abs(omega) < _EPS:
                r["turn_direction"] = None
            else:
                r["turn_direction"] = "right" if omega > 0 else "left"
            continue

        # Last resort fallback (classify_responsiveness not called): raw-max over window.
        time = r["time"]
        ang_vel_deg = np.rad2deg(r["ang_vel"])
        ang_vel_abs = np.abs(ang_vel_deg)
        end_t = r["end_expansion_time"]
        mask = (time >= start_offset_s) & (time <= end_t + end_offset_s)
        masked_abs = ang_vel_abs[mask]
        if masked_abs.size == 0 or np.all(np.isnan(masked_abs)):
            r["signed_peak_ang_vel_deg_s"] = float("nan")
            r["turn_direction"] = None
            continue
        peak_local = int(np.nanargmax(masked_abs))
        peak_global = int(np.where(mask)[0][peak_local])
        signed_peak = float(ang_vel_deg[peak_global])
        r["signed_peak_ang_vel_deg_s"] = signed_peak
        if signed_peak > 0:
            r["turn_direction"] = "right"
        elif signed_peak < 0:
            r["turn_direction"] = "left"
        else:
            r["turn_direction"] = None
    return responses


def _detect_peak(
    trace: np.ndarray,
    ang_vel_abs: np.ndarray,
    win_indices: np.ndarray,
    threshold_deg_s: float,
) -> tuple[int, float]:
    """Method 0: find first abs peak above threshold. Returns (peak_global_idx, peak)."""
    kw = {**_PEAK_KW, "height": threshold_deg_s}
    peak_locals, _ = find_peaks(trace[win_indices], **kw)
    if peak_locals.size > 0:
        idx = int(win_indices[peak_locals[0]])
        return idx, float(ang_vel_abs[idx])
    return -1, float("nan")


def _compute_zscore(
    ang_vel_abs: np.ndarray,
    bl_mask: np.ndarray,
    peak: float,
    zscore_k: float,
) -> tuple[float, float, float, bool]:
    """Method 1: baseline z-score. Returns (bl_mean, bl_sd, zscore, is_responsive)."""
    bl_vals = ang_vel_abs[bl_mask]
    bl_mean = float(np.nanmean(bl_vals)) if bl_vals.size > 0 else 0.0
    bl_sd = float(np.nanstd(bl_vals)) if bl_vals.size > 0 else 0.0
    if np.isnan(peak) or bl_sd == 0.0:
        zscore = float("nan")
    else:
        zscore = (peak - bl_mean) / bl_sd
    is_resp = (not np.isnan(zscore)) and (zscore >= zscore_k)
    return bl_mean, bl_sd, zscore, is_resp


def _detect_saccade(
    signed_trace: np.ndarray,
    ang_vel_deg_signed: np.ndarray,
    time: np.ndarray,
    win_indices: np.ndarray,
    threshold_deg_s: float,
) -> tuple[float, float]:
    """Method 3: first signed saccade above threshold. Returns (saccade_peak, saccade_peak_time_ms)."""
    kw = {**_PEAK_KW, "height": threshold_deg_s}
    win_sig = signed_trace[win_indices]
    pos_locals, _ = find_peaks(win_sig, **kw)
    neg_locals, _ = find_peaks(-win_sig, **kw)
    candidates = np.sort(np.concatenate([pos_locals, neg_locals]))
    if candidates.size > 0:
        first_global = win_indices[candidates[0]]
        return float(ang_vel_deg_signed[first_global]), float(time[first_global] * 1000.0)
    return float("nan"), float("nan")


def _fallback_ref_idx(time: np.ndarray, end_t: float, mean_peak_time: float | None) -> int:
    """Index used when a trial has no detected saccade peak."""
    if mean_peak_time is not None:
        return int(np.argmin(np.abs(time - mean_peak_time)))
    return int(np.argmin(np.abs(time - end_t)))


def _apply_canonical_heading_change(
    r: "Response",
    ref_idx: int,
    ref_frames: int = _PEAK_ALIGNED_REF_FRAMES,
) -> None:
    """Write heading_change using the peak-aligned vector method."""
    xvel, yvel = r.get("xvel"), r.get("yvel")
    if xvel is None or yvel is None:
        r["heading_change"] = float("nan")
        return
    r["heading_change"] = _compute_heading_change_vector(xvel, yvel, ref_idx, window=ref_frames)


def classify_responsiveness(
    responses: list[Response],
    threshold_deg_s: float = 300.0,
    window_ms: float | tuple[float, float] | list[float] = 200.0,
    zscore_k: float = 3.0,
    baseline_window_ms: tuple[float, float] = (-400.0, -100.0),
    heading_threshold_deg: float = 30.0,
    impulse_threshold_deg: float = 20.0,
    method: str = "combined",
) -> list[Response]:
    """Tag each response dict with responsiveness metadata.

    Methods:
        0 (peak): peak |ω| ≥ threshold within window_ms of end_expansion_time.
        1 (zscore): same peak normalized by pre-stim baseline SD.
        2 (heading): |heading_change| ≥ heading_threshold_deg (computed in second pass).
        3 (saccade): first signed saccade above threshold in window.
        4 (impulse): ∑|ω|·dt over window ≥ impulse_threshold_deg.
        5 (combined, default): Method 3 AND Method 2.

    heading_change is the canonical peak-aligned vector metric, computed in a
    second pass once saccade_peak_time_ms is known for all trials.

    Args:
        responses: List of response dicts from `extract_responses`.
        threshold_deg_s: |ω| threshold for methods 0, 1, 3, 5.
        window_ms: Reaction window around end_expansion_time in ms.
        zscore_k: Z-score threshold for method 1.
        baseline_window_ms: (start, end) in ms for baseline stats.
        heading_threshold_deg: Heading change threshold for methods 2 and 5.
        impulse_threshold_deg: Angular impulse threshold for method 4.
        method: Which method controls is_responsive.

    Returns:
        The same list (for chaining).
    """
    if method not in RESPONSIVENESS_METHOD_FIELDS:
        valid = ", ".join(sorted(RESPONSIVENESS_METHOD_FIELDS))
        raise ValueError(f"method must be one of: {valid}; got {method!r}")

    before_s, after_s = _reaction_window_seconds(window_ms)
    bl_start = baseline_window_ms[0] / 1000.0
    bl_end = baseline_window_ms[1] / 1000.0

    for r in responses:
        time = r["time"]
        dt = float(time[1] - time[0]) if len(time) > 1 else DT_SECONDS
        ang_vel_deg_signed = np.rad2deg(r["ang_vel"])
        ang_vel_abs = np.abs(ang_vel_deg_signed)
        end_t = r["end_expansion_time"]

        trace = np.where(np.isnan(ang_vel_abs), 0.0, ang_vel_abs)
        signed_trace = np.where(np.isnan(ang_vel_deg_signed), 0.0, ang_vel_deg_signed)

        window_mask = (time >= end_t - before_s) & (time <= end_t + after_s)
        win_indices = np.where(window_mask)[0]
        bl_mask = (time >= bl_start) & (time <= bl_end)

        # Method 0 — peak
        peak_global_idx, peak = _detect_peak(trace, ang_vel_abs, win_indices, threshold_deg_s)
        r["peak_ang_vel_deg_s"] = peak
        r["is_responsive_peak"] = not np.isnan(peak)
        r["_peak_global_idx"] = peak_global_idx

        # Method 1 — z-score
        bl_mean, bl_sd, zscore, is_resp_z = _compute_zscore(ang_vel_abs, bl_mask, peak, zscore_k)
        r["baseline_ang_vel_mean"] = bl_mean
        r["baseline_ang_vel_sd"] = bl_sd
        r["peak_ang_vel_zscore"] = zscore
        r["is_responsive_zscore"] = is_resp_z

        # Method 2 — updated in second pass after heading_change is computed
        r["is_responsive_heading"] = False

        # Method 3 — signed saccade
        saccade_peak, saccade_peak_time = _detect_saccade(
            signed_trace, ang_vel_deg_signed, time, win_indices, threshold_deg_s
        )
        r["saccade_peak_time_ms"] = saccade_peak_time
        r["saccade_peak_ang_vel_signed_deg_s"] = saccade_peak
        r["is_responsive_saccade"] = not np.isnan(saccade_peak)

        # Method 4 — angular impulse
        win_abs = ang_vel_abs[window_mask]
        impulse = float(np.nansum(win_abs) * dt)
        r["angular_impulse_deg"] = impulse
        r["is_responsive_impulse"] = impulse >= impulse_threshold_deg
        r["mean_ang_vel_window_deg_s"] = float(np.nanmean(win_abs)) if win_abs.size > 0 else float("nan")

        # Method 5 — combined (saccade + heading)
        r["peak_ang_vel_signed_deg_s"] = saccade_peak
        r["is_responsive_combined"] = r["is_responsive_saccade"] and r["is_responsive_heading"]
        r["responsiveness_method"] = method
        r["is_responsive"] = bool(r[RESPONSIVENESS_METHOD_FIELDS[method]])

    # Second pass: canonical heading_change.
    # Responsive flies use their own saccade peak time; non-responsive flies use
    # the population-mean saccade peak time (Bug A fix: use saccade_peak_time_ms,
    # not Method-0 _peak_global_idx which can be -1 for combined-method trials).
    responsive_peak_times = [
        r["saccade_peak_time_ms"] / 1000.0
        for r in responses
        if r.get("is_responsive") and not np.isnan(r["saccade_peak_time_ms"])
    ]
    mean_peak_time = float(np.mean(responsive_peak_times)) if responsive_peak_times else None

    for r in responses:
        time = r["time"]
        end_t = r["end_expansion_time"]
        sac_t = r["saccade_peak_time_ms"]
        if not np.isnan(sac_t):
            ref_idx = int(np.argmin(np.abs(time - sac_t / 1000.0)))
        else:
            ref_idx = _fallback_ref_idx(time, end_t, mean_peak_time)
        r.pop("_peak_global_idx", None)
        r["_fallback_ref_idx"] = ref_idx
        _apply_canonical_heading_change(r, ref_idx)
        r["is_responsive_heading"] = (
            not np.isnan(r["heading_change"])
            and abs(r["heading_change"]) >= heading_threshold_deg
        )
        r["is_responsive_combined"] = r["is_responsive_saccade"] and r["is_responsive_heading"]
        r["is_responsive"] = bool(r[RESPONSIVENESS_METHOD_FIELDS[method]])

    return responses
