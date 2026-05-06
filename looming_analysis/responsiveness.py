"""Classify trials as responsive based on angular velocity metrics."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ._types import DT_SECONDS, Response, _circ_diff_deg
from .extract import _compute_heading_change_vector

_PEAK_KW = dict(height=0.0, prominence=300, width=(3, None), distance=5)
_PEAK_ALIGNED_REF_FRAMES = 10

RESPONSIVENESS_METHOD_FIELDS = {
    "peak": "is_responsive_peak",
    "zscore": "is_responsive_zscore",
    "heading": "is_responsive_heading",
    "saccade": "is_responsive_saccade",
    "impulse": "is_responsive_impulse",
    "combined": "is_responsive_combined",
}

_EPS = 1e-9


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
    before, after = float(before_ms), float(after_ms)
    if before < 0 or after < 0:
        raise ValueError("window_ms values must be >= 0.")
    return before / 1000.0, after / 1000.0


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
    xvel, yvel = r.get("xvel"), r.get("yvel")
    if xvel is None or yvel is None:
        r["heading_change"] = float("nan")
        return
    r["heading_change"] = _compute_heading_change_vector(xvel, yvel, ref_idx, window=ref_frames)


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
    """
    for r in responses:
        saccade_signed = r.get("saccade_peak_ang_vel_signed_deg_s")
        if saccade_signed is not None and not np.isnan(saccade_signed):
            r["signed_peak_ang_vel_deg_s"] = saccade_signed
            r["turn_direction"] = "right" if saccade_signed > 0 else "left"
            r.pop("_fallback_ref_idx", None)
            continue

        fallback_ref = r.pop("_fallback_ref_idx", None)
        if fallback_ref is not None:
            ang_vel_deg = np.rad2deg(r["ang_vel"])
            omega = float(ang_vel_deg[fallback_ref])
            r["signed_peak_ang_vel_deg_s"] = omega
            r["turn_direction"] = None if abs(omega) < _EPS else ("right" if omega > 0 else "left")
            continue

        # Last resort (classify_responsiveness not called): raw-max over window.
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
        r["turn_direction"] = "right" if signed_peak > 0 else ("left" if signed_peak < 0 else None)
    return responses


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

    When method="combined" (default), only the saccade (Method 3) and heading-change
    (Method 2) signals are computed — Methods 0/1/4 are skipped.

    Methods:
        peak:     peak |ω| ≥ threshold within window_ms.
        zscore:   same peak normalised by pre-stim baseline SD.
        heading:  |heading_change| ≥ heading_threshold_deg.
        saccade:  first signed saccade above threshold in window.
        impulse:  ∑|ω|·dt over window ≥ impulse_threshold_deg.
        combined: saccade AND heading (default).
    """
    if method not in RESPONSIVENESS_METHOD_FIELDS:
        valid = ", ".join(sorted(RESPONSIVENESS_METHOD_FIELDS))
        raise ValueError(f"method must be one of: {valid}; got {method!r}")

    before_s, after_s = _reaction_window_seconds(window_ms)
    bl_start, bl_end = baseline_window_ms[0] / 1000.0, baseline_window_ms[1] / 1000.0
    needs_saccade = method in ("saccade", "heading", "combined")
    needs_peak = method in ("peak", "zscore")

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
        win_abs = ang_vel_abs[window_mask]

        if needs_peak:
            kw = {**_PEAK_KW, "height": threshold_deg_s}
            peak_locals, _ = find_peaks(trace[win_indices], **kw)
            if peak_locals.size > 0:
                peak_global_idx = int(win_indices[peak_locals[0]])
                peak = float(ang_vel_abs[peak_global_idx])
            else:
                peak = float("nan")
            r["peak_ang_vel_deg_s"] = peak
            r["is_responsive_peak"] = not np.isnan(peak)

        if method == "zscore":
            bl_mask = (time >= bl_start) & (time <= bl_end)
            bl_vals = ang_vel_abs[bl_mask]
            bl_mean = float(np.nanmean(bl_vals)) if bl_vals.size > 0 else 0.0
            bl_sd = float(np.nanstd(bl_vals)) if bl_vals.size > 0 else 0.0
            zscore = (peak - bl_mean) / bl_sd if not np.isnan(peak) and bl_sd != 0.0 else float("nan")
            r["baseline_ang_vel_mean"] = bl_mean
            r["baseline_ang_vel_sd"] = bl_sd
            r["peak_ang_vel_zscore"] = zscore
            r["is_responsive_zscore"] = (not np.isnan(zscore)) and (zscore >= zscore_k)

        if needs_saccade:
            kw = {**_PEAK_KW, "height": threshold_deg_s}
            win_sig = signed_trace[win_indices]
            pos_locals, _ = find_peaks(win_sig, **kw)
            neg_locals, _ = find_peaks(-win_sig, **kw)
            candidates = np.sort(np.concatenate([pos_locals, neg_locals]))
            if candidates.size > 0:
                first_global = win_indices[candidates[0]]
                saccade_peak = float(ang_vel_deg_signed[first_global])
                saccade_peak_time = float(time[first_global] * 1000.0)
            else:
                saccade_peak = float("nan")
                saccade_peak_time = float("nan")
            r["saccade_peak_time_ms"] = saccade_peak_time
            r["saccade_peak_ang_vel_signed_deg_s"] = saccade_peak
            r["is_responsive_saccade"] = not np.isnan(saccade_peak)
            # Derive peak_ang_vel_deg_s from saccade for plot compatibility.
            r["peak_ang_vel_deg_s"] = float("nan") if np.isnan(saccade_peak) else abs(saccade_peak)
            r["mean_ang_vel_window_deg_s"] = float(np.nanmean(win_abs)) if win_abs.size > 0 else float("nan")

        if method == "impulse":
            impulse = float(np.nansum(win_abs) * dt)
            r["angular_impulse_deg"] = impulse
            r["is_responsive_impulse"] = impulse >= impulse_threshold_deg
            r["mean_ang_vel_window_deg_s"] = float(np.nanmean(win_abs)) if win_abs.size > 0 else float("nan")

        r["is_responsive_heading"] = False
        r["is_responsive_combined"] = False
        r["responsiveness_method"] = method
        if method not in ("combined", "heading"):
            r["is_responsive"] = bool(r[RESPONSIVENESS_METHOD_FIELDS[method]])

    if method in ("heading", "combined"):
        # Use is_responsive_saccade (not is_responsive) so the mean is correct even before
        # heading is evaluated — avoids the fallback defaulting to end_expansion_time.
        responsive_peak_times = [
            r["saccade_peak_time_ms"] / 1000.0
            for r in responses
            if r.get("is_responsive_saccade") and not np.isnan(r["saccade_peak_time_ms"])
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
            r["_fallback_ref_idx"] = ref_idx
            _apply_canonical_heading_change(r, ref_idx)
            r["is_responsive_heading"] = (
                not np.isnan(r["heading_change"])
                and abs(r["heading_change"]) >= heading_threshold_deg
            )
            r["is_responsive_combined"] = r.get("is_responsive_saccade", False) and r["is_responsive_heading"]
            r["is_responsive"] = bool(r[RESPONSIVENESS_METHOD_FIELDS[method]])

    return responses
