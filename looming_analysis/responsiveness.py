"""Classify trials as responsive based on angular velocity metrics."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ._types import DT_SECONDS, Response

RESPONSIVENESS_METHOD_FIELDS = {
    "peak": "is_responsive_peak",
    "zscore": "is_responsive_zscore",
    "heading": "is_responsive_heading",
    "saccade": "is_responsive_saccade",
    "impulse": "is_responsive_impulse",
    "combined": "is_responsive_combined",
}


def compute_turn_direction(
    responses: list[Response],
    start_offset_s: float = 0.0,
    end_offset_s: float = 0.0,
) -> list[Response]:
    """Add signed peak angular velocity and turn direction to each response (in-place).

    The peak is defined as the sample with the highest ``|ω|`` within
    ``[start_offset_s, end_expansion_time + end_offset_s]``.  The sign of
    ``ω`` at that sample determines turn direction.

    Fields added to each response dict:
        signed_peak_ang_vel_deg_s (float): Signed ``ω`` at peak |ω| (deg/s).
            ``NaN`` when the window is empty or all-NaN.
        turn_direction (str | None): ``"left"`` (ω < 0), ``"right"`` (ω > 0),
            or ``None`` when the peak is NaN or exactly zero.

    Args:
        responses: List of response dicts from ``extract_responses``.
        start_offset_s: Window start relative to stimulus onset (t=0).
            Negative values extend before onset.
        end_offset_s: Seconds added to ``end_expansion_time`` for window end.
            Positive values extend past the end of expansion.

    Returns:
        The same list (for chaining).
    """
    for r in responses:
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


def classify_responsiveness(
    responses: list[Response],
    threshold_deg_s: float = 300.0,
    window_ms: float = 100.0,
    zscore_k: float = 3.0,
    baseline_window_ms: tuple[float, float] = (-400.0, -100.0),
    min_duration_ms: float = 30.0,
    max_duration_ms: float = 150.0,
    heading_threshold_deg: float = 30.0,
    impulse_threshold_deg: float = 20.0,
    post_expansion_ms: float = 200.0,
    method: str = "combined",
) -> list[Response]:
    """Tag each response dict with responsiveness metadata from multiple methods.

    All methods are computed unconditionally. `method` selects which method controls
    the primary `is_responsive` field. The method-specific fields remain available
    for comparison and plotting.

    **Method 0 — peak:** peak `|ω|` ≥ `threshold_deg_s` within ±`window_ms`
    of `end_expansion_time`. Uses `scipy.signal.find_peaks`, which requires a local
    maximum within the window. A monotonically rising trace without a clear peak will
    fail Method 0 but may still satisfy Methods 3 (saccade) and 4 (impulse), which
    scan the raw trace.

    **Method 1 — z-score:** same peak, normalised by pre-stim baseline SD.
    Responsive if `peak_zscore ≥ zscore_k`.

    **Method 2 — heading change:** `|heading_change| ≥ heading_threshold_deg`.
    Captures the net directional *outcome*, independent of instantaneous rate.

    **Method 3 — min-duration saccade:** `|ω| ≥ threshold_deg_s` for a consecutive
    run of `min_duration_ms`–`max_duration_ms` within the detection window.
    Records onset time and duration of the first qualifying saccade.

    **Method 4 — angular impulse:** `∑|ω| × dt` over the detection window ≥
    `impulse_threshold_deg`. Robust to brief noise spikes.

    **Method 5 — signed peak + heading change (default):** finds the sample with
    the highest `|ω|` on the *signed* trace (no absolute-value baseline inflation)
    within `[t=0, end_expansion_time + post_expansion_ms]`, then requires both
    `|peak| ≥ threshold_deg_s` AND `|heading_change| ≥ heading_threshold_deg`.
    This is the most specific criterion and sets `is_responsive`.

    Fields added to each response dict:
        is_responsive (bool)              — method 5 (default)
        peak_ang_vel_deg_s (float)        — method 0
        is_responsive_peak (bool)         — method 0
        baseline_ang_vel_mean (float)     — method 1
        baseline_ang_vel_sd (float)       — method 1
        peak_ang_vel_zscore (float)       — method 1 (NaN if sd=0 or no peak)
        is_responsive_zscore (bool)       — method 1
        is_responsive_heading (bool)      — method 2
        saccade_onset_ms (float)          — method 3 (NaN if no qualifying saccade)
        saccade_duration_ms (float)       — method 3 (NaN if no qualifying saccade)
        is_responsive_saccade (bool)      — method 3
        angular_impulse_deg (float)       — method 4
        is_responsive_impulse (bool)      — method 4
        peak_ang_vel_signed_deg_s (float) — method 5
        is_responsive_combined (bool)     — method 5
        responsiveness_method (str)       — selected method for is_responsive

    Args:
        responses: List of response dicts from `extract_responses`.
        threshold_deg_s: `|ω|` threshold used by methods 0, 1, 3, and 5.
        window_ms: Half-width (ms) of the detection window around `end_expansion_time`
            (methods 0–4 only).
        zscore_k: Z-score threshold for method 1.
        baseline_window_ms: (start, end) in ms relative to stim onset for baseline stats.
            Default (-400, -100) avoids the 100 ms immediately before stimulus onset.
        min_duration_ms: Minimum saccade duration for method 3.
        max_duration_ms: Maximum saccade duration for method 3.
        heading_threshold_deg: Heading change threshold for methods 2 and 5 (degrees).
        impulse_threshold_deg: Angular impulse threshold for method 4 (degrees).
        post_expansion_ms: ms after `end_expansion_time` included in the method 5
            detection window. Default 200 ms.
        method: Which method controls `is_responsive`. One of "peak", "zscore",
            "heading", "saccade", "impulse", or "combined".

    Returns:
        The same list (for chaining).
    """
    if method not in RESPONSIVENESS_METHOD_FIELDS:
        valid = ", ".join(sorted(RESPONSIVENESS_METHOD_FIELDS))
        raise ValueError(f"method must be one of: {valid}; got {method!r}")

    half_win = window_ms / 1000.0
    bl_start = baseline_window_ms[0] / 1000.0
    bl_end = baseline_window_ms[1] / 1000.0
    min_dur_frames = min_duration_ms / 1000.0
    max_dur_frames = max_duration_ms / 1000.0

    for r in responses:
        time = r["time"]
        dt = float(time[1] - time[0]) if len(time) > 1 else DT_SECONDS
        ang_vel_deg_signed = np.rad2deg(r["ang_vel"])
        ang_vel_abs = np.abs(ang_vel_deg_signed)
        end_t = r["end_expansion_time"]

        # NaN-safe trace for peak/run detection
        trace = np.where(np.isnan(ang_vel_abs), 0.0, ang_vel_abs)

        window_mask = np.abs(time - end_t) <= half_win

        # ------------------------------------------------------------------
        # Method 0 — peak (unchanged logic)
        # ------------------------------------------------------------------
        peak_indices, _ = find_peaks(trace)
        win_peaks = peak_indices[np.abs(time[peak_indices] - end_t) <= half_win]
        if win_peaks.size > 0:
            peak = float(np.max(ang_vel_abs[win_peaks]))
        else:
            peak = float("nan")
        r["peak_ang_vel_deg_s"] = peak
        r["is_responsive_peak"] = (not np.isnan(peak)) and (peak >= threshold_deg_s)

        # ------------------------------------------------------------------
        # Method 1 — z-score relative to pre-stim baseline
        # ------------------------------------------------------------------
        bl_mask = (time >= bl_start) & (time <= bl_end)
        bl_vals = ang_vel_abs[bl_mask]
        bl_mean = float(np.nanmean(bl_vals)) if bl_vals.size > 0 else 0.0
        bl_sd = float(np.nanstd(bl_vals)) if bl_vals.size > 0 else 0.0
        r["baseline_ang_vel_mean"] = bl_mean
        r["baseline_ang_vel_sd"] = bl_sd
        if np.isnan(peak) or bl_sd == 0.0:
            zscore = float("nan")
        else:
            zscore = (peak - bl_mean) / bl_sd
        r["peak_ang_vel_zscore"] = zscore
        r["is_responsive_zscore"] = (not np.isnan(zscore)) and (zscore >= zscore_k)

        # ------------------------------------------------------------------
        # Method 2 — heading change magnitude
        # ------------------------------------------------------------------
        r["is_responsive_heading"] = (
            abs(r.get("heading_change", 0.0)) >= heading_threshold_deg
        )

        # ------------------------------------------------------------------
        # Method 3 — min-duration saccade
        # ------------------------------------------------------------------
        saccade_onset = float("nan")
        saccade_dur = float("nan")
        found_saccade = False

        in_run = False
        run_start_idx = 0
        window_indices = np.where(window_mask)[0]

        for idx in window_indices:
            above = trace[idx] >= threshold_deg_s
            if above and not in_run:
                in_run = True
                run_start_idx = idx
            elif not above and in_run:
                run_dur = (idx - run_start_idx) * dt
                if min_dur_frames <= run_dur <= max_dur_frames:
                    saccade_onset = float(time[run_start_idx] * 1000.0)
                    saccade_dur = float(run_dur * 1000.0)
                    found_saccade = True
                    break
                in_run = False
        # Handle run that extends to end of window
        if in_run and not found_saccade:
            run_dur = (window_indices[-1] + 1 - run_start_idx) * dt
            if min_dur_frames <= run_dur <= max_dur_frames:
                saccade_onset = float(time[run_start_idx] * 1000.0)
                saccade_dur = float(run_dur * 1000.0)
                found_saccade = True

        r["saccade_onset_ms"] = saccade_onset
        r["saccade_duration_ms"] = saccade_dur
        r["is_responsive_saccade"] = found_saccade

        # ------------------------------------------------------------------
        # Method 4 — angular impulse in detection window
        # ------------------------------------------------------------------
        impulse = float(np.sum(trace[window_mask]) * dt)
        r["angular_impulse_deg"] = impulse
        r["is_responsive_impulse"] = impulse >= impulse_threshold_deg

        # ------------------------------------------------------------------
        # Method 5 — signed peak + heading change (default / is_responsive)
        # ------------------------------------------------------------------
        m5_mask = (time >= 0) & (time <= end_t + post_expansion_ms / 1000.0)
        m5_vals = ang_vel_deg_signed[m5_mask]
        if m5_vals.size > 0 and not np.all(np.isnan(m5_vals)):
            m5_peak_idx = int(np.nanargmax(np.abs(m5_vals)))
            signed_peak = float(m5_vals[m5_peak_idx])
        else:
            signed_peak = float("nan")
        r["peak_ang_vel_signed_deg_s"] = signed_peak
        r["is_responsive_combined"] = (
            (not np.isnan(signed_peak))
            and (abs(signed_peak) >= threshold_deg_s)
            and (abs(r.get("heading_change", 0.0)) >= heading_threshold_deg)
        )
        r["responsiveness_method"] = method
        r["is_responsive"] = bool(r[RESPONSIVENESS_METHOD_FIELDS[method]])

    return responses
