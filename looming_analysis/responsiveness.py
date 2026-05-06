"""Classify trials as responsive based on angular velocity metrics."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import circmean

from ._types import DT_SECONDS, Response
from .extract import _compute_heading_change_vector

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


def compute_turn_direction(
    responses: list[Response],
    start_offset_s: float = 0.0,
    end_offset_s: float = 0.0,
) -> list[Response]:
    """Add signed peak angular velocity and turn direction to each response (in-place).

    If ``classify_responsiveness`` has already been called, the sign is taken
    from ``saccade_peak_ang_vel_signed_deg_s`` (the ``find_peaks``-detected
    saccade) so that turn direction is consistent with the responsiveness
    algorithm. Otherwise the peak is defined as the sample with the highest
    ``|ω|`` within ``[start_offset_s, end_expansion_time + end_offset_s]``.

    Fields added to each response dict:
        signed_peak_ang_vel_deg_s (float): Signed ``ω`` at peak |ω| (deg/s).
            ``NaN`` when the window is empty or all-NaN.
        turn_direction (str | None): ``"left"`` (ω < 0), ``"right"`` (ω > 0),
            or ``None`` when the peak is NaN or exactly zero.

    Args:
        responses: List of response dicts from ``extract_responses``.
        start_offset_s: Window start relative to stimulus onset (t=0).
            Negative values extend before onset. Ignored when saccade peak
            is already available.
        end_offset_s: Seconds added to ``end_expansion_time`` for window end.
            Positive values extend past the end of expansion. Ignored when
            saccade peak is already available.

    Returns:
        The same list (for chaining).
    """
    for r in responses:
        # Prefer the saccade-detected signed peak when available.
        saccade_signed = r.get("saccade_peak_ang_vel_signed_deg_s")
        if saccade_signed is not None and not np.isnan(saccade_signed):
            r["signed_peak_ang_vel_deg_s"] = saccade_signed
            r["turn_direction"] = "right" if saccade_signed > 0 else "left"
            continue

        # Fallback: raw-max over the expansion window.
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
    window_ms: float | tuple[float, float] | list[float] = 200.0,
    zscore_k: float = 3.0,
    baseline_window_ms: tuple[float, float] = (-400.0, -100.0),
    heading_threshold_deg: float = 30.0,
    impulse_threshold_deg: float = 20.0,
    method: str = "combined",
) -> list[Response]:
    """Tag each response dict with responsiveness metadata from multiple methods.

    All methods are computed unconditionally. `method` selects which method controls
    the primary `is_responsive` field. The method-specific fields remain available
    for comparison and plotting.

    **Method 0 — peak:** peak `|ω|` ≥ `threshold_deg_s` within `window_ms`
    of `end_expansion_time`. Uses `scipy.signal.find_peaks`, which requires a local
    maximum within the window. A monotonically rising trace without a clear peak will
    fail Method 0 but may still satisfy Methods 3 (saccade) and 4 (impulse), which
    scan the raw trace.

    **Method 1 — z-score:** same peak, normalised by pre-stim baseline SD.
    Responsive if `peak_zscore ≥ zscore_k`.

    **Method 2 — heading change:** `|heading_change| ≥ heading_threshold_deg`.
    Captures the net directional *outcome*, independent of instantaneous rate.

    **Method 3 — signed-peak saccade:** finds positive peaks in `ω` and negative
    peaks in `-ω` within the detection window, then selects the signed peak with
    the largest absolute magnitude. Responsive if `|peak| ≥ threshold_deg_s`.

    **Method 4 — angular impulse:** `∑|ω| × dt` over the detection window ≥
    `impulse_threshold_deg`. Robust to brief noise spikes.

    **Method 5 — saccade + heading change (default):** requires Method 3
    signed-peak saccade detection AND `|heading_change| ≥ heading_threshold_deg`.
    This is the default criterion and sets `is_responsive`.

    Fields added to each response dict:
        is_responsive (bool)              — method 5 (default)
        peak_ang_vel_deg_s (float)        — method 0
        is_responsive_peak (bool)         — method 0
        baseline_ang_vel_mean (float)     — method 1
        baseline_ang_vel_sd (float)       — method 1
        peak_ang_vel_zscore (float)       — method 1 (NaN if sd=0 or no peak)
        is_responsive_zscore (bool)       — method 1
        is_responsive_heading (bool)      — method 2
        saccade_peak_time_ms (float)      — method 3 (NaN if no qualifying saccade)
        saccade_peak_ang_vel_signed_deg_s — method 3 (NaN if no qualifying saccade)
        saccade_onset_ms (float)          — legacy alias for saccade_peak_time_ms
        saccade_duration_ms (float)       — always NaN; duration is not used
        is_responsive_saccade (bool)      — method 3
        angular_impulse_deg (float)              — method 4
        is_responsive_impulse (bool)             — method 4
        heading_change_window_net (float)        — H0: net change, pre-stim → end of detection window
        heading_change_max_dev (float)           — H1: max deviation from pre-stim baseline in window
        heading_change_post_saccade (float)      — H2: net change locked to saccade (NaN if none)
        heading_change_path_length (float)       — H3: total |Δheading| over detection window
        is_responsive_heading_window_net (bool)  — H0 ≥ heading_threshold_deg
        is_responsive_heading_max_dev (bool)     — H1 ≥ heading_threshold_deg
        is_responsive_heading_post_saccade (bool)— H2 ≥ heading_threshold_deg
        is_responsive_heading_path_length (bool) — H3 ≥ heading_threshold_deg (note: path length
            accumulates all rotation, so the effective threshold is higher than for net metrics)
        peak_ang_vel_signed_deg_s (float)        — method 5
        is_responsive_combined (bool)            — method 5
        responsiveness_method (str)              — selected method for is_responsive

    Args:
        responses: List of response dicts from `extract_responses`.
        threshold_deg_s: `|ω|` threshold used by methods 0, 1, 3, and 5.
        window_ms: Reaction window around `end_expansion_time` in ms. A scalar
            gives a symmetric window (`end - window_ms` to `end + window_ms`).
            A two-item sequence gives an asymmetric `(before_ms, after_ms)` window.
        zscore_k: Z-score threshold for method 1.
        baseline_window_ms: (start, end) in ms relative to stim onset for baseline stats.
            Default (-400, -100) avoids the 100 ms immediately before stimulus onset.
        heading_threshold_deg: Heading change threshold for methods 2 and 5 (degrees).
        impulse_threshold_deg: Angular impulse threshold for method 4 (degrees).
        method: Which method controls `is_responsive`. One of "peak", "zscore",
            "heading", "saccade", "impulse", or "combined".

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

        # NaN-safe traces for peak detection
        trace = np.where(np.isnan(ang_vel_abs), 0.0, ang_vel_abs)
        signed_trace = np.where(np.isnan(ang_vel_deg_signed), 0.0, ang_vel_deg_signed)

        window_mask = (time >= end_t - before_s) & (time <= end_t + after_s)
        win_indices = np.where(window_mask)[0]

        # ------------------------------------------------------------------
        # Method 0 — first abs peak above threshold in window
        # ------------------------------------------------------------------
        peak_locals, _ = find_peaks(
            trace[win_indices],
            height=threshold_deg_s,
            prominence=300,
            width=(3, 8),
            distance=5,
        )
        if peak_locals.size > 0:
            peak_global_idx = int(win_indices[peak_locals[0]])
            peak = float(ang_vel_abs[peak_global_idx])
        else:
            peak_global_idx = -1
            peak = float("nan")
        r["peak_ang_vel_deg_s"] = peak
        r["is_responsive_peak"] = not np.isnan(peak)
        r["_peak_global_idx"] = peak_global_idx

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
        # Method 3 — first signed peak above threshold in window
        # ------------------------------------------------------------------
        win_sig = signed_trace[win_indices]
        _peak_kw = dict(
            height=threshold_deg_s, prominence=300, width=(3, 8), distance=5
        )
        pos_locals, _ = find_peaks(win_sig, **_peak_kw)
        neg_locals, _ = find_peaks(-win_sig, **_peak_kw)
        candidates_local = np.sort(np.concatenate([pos_locals, neg_locals]))

        if candidates_local.size > 0:
            first_global = win_indices[candidates_local[0]]
            saccade_peak = float(ang_vel_deg_signed[first_global])
            saccade_peak_time = float(time[first_global] * 1000.0)
        else:
            saccade_peak = float("nan")
            saccade_peak_time = float("nan")

        r["saccade_peak_time_ms"] = saccade_peak_time
        r["saccade_peak_ang_vel_signed_deg_s"] = saccade_peak
        r["saccade_onset_ms"] = saccade_peak_time
        r["saccade_duration_ms"] = float("nan")
        r["is_responsive_saccade"] = not np.isnan(saccade_peak)

        # ------------------------------------------------------------------
        # Method 4 — angular impulse in detection window
        # ------------------------------------------------------------------
        win_abs = ang_vel_abs[window_mask]
        impulse = float(np.sum(trace[window_mask]) * dt)
        r["angular_impulse_deg"] = impulse
        r["is_responsive_impulse"] = impulse >= impulse_threshold_deg
        r["mean_ang_vel_window_deg_s"] = float(np.nanmean(win_abs)) if win_abs.size > 0 else float("nan")

        # ------------------------------------------------------------------
        # Heading change metrics H0–H3 (require r["heading"] in radians)
        # ------------------------------------------------------------------
        heading = r.get("heading")
        if heading is not None and bl_mask.any():
            bl_heading = circmean(heading[bl_mask], low=-np.pi, high=np.pi)

            # H0 — detection-window net change (pre-stim mean vs. end of window)
            post_win_mask = (time >= end_t + after_s - 0.1) & (time <= end_t + after_s)
            if post_win_mask.any():
                post_h = circmean(heading[post_win_mask], low=-np.pi, high=np.pi)
                hc_window_net = float(
                    np.rad2deg(
                        np.arctan2(
                            np.sin(post_h - bl_heading), np.cos(post_h - bl_heading)
                        )
                    )
                )
            else:
                hc_window_net = float("nan")

            # H1 — max circular deviation from pre-stim baseline in window
            if window_mask.any():
                deviations = np.abs(
                    np.rad2deg(
                        np.arctan2(
                            np.sin(heading[window_mask] - bl_heading),
                            np.cos(heading[window_mask] - bl_heading),
                        )
                    )
                )
                hc_max_dev = float(np.nanmax(deviations))
            else:
                hc_max_dev = float("nan")

            # H2 — net heading change locked to detected saccade (NaN if no saccade)
            if not np.isnan(saccade_peak_time):
                t_sac = saccade_peak_time / 1000.0
                pre_sac = (time >= t_sac - 0.05) & (time < t_sac)
                post_sac = (time > t_sac) & (time <= t_sac + 0.05)
                if pre_sac.any() and post_sac.any():
                    h_pre = circmean(heading[pre_sac], low=-np.pi, high=np.pi)
                    h_post = circmean(heading[post_sac], low=-np.pi, high=np.pi)
                    hc_post_saccade = float(
                        np.rad2deg(
                            np.arctan2(np.sin(h_post - h_pre), np.cos(h_post - h_pre))
                        )
                    )
                else:
                    hc_post_saccade = float("nan")
            else:
                hc_post_saccade = float("nan")

            # H3 — heading path length (total rotation) in window
            win_heading = heading[window_mask]
            if win_heading.size > 1:
                diffs = np.abs(
                    np.rad2deg(
                        np.arctan2(
                            np.sin(np.diff(win_heading)),
                            np.cos(np.diff(win_heading)),
                        )
                    )
                )
                hc_path_length = float(np.sum(diffs))
            else:
                hc_path_length = float("nan")
        else:
            hc_window_net = hc_max_dev = hc_post_saccade = hc_path_length = float("nan")

        r["heading_change_window_net"] = hc_window_net
        r["heading_change_max_dev"] = hc_max_dev
        r["heading_change_post_saccade"] = hc_post_saccade
        r["heading_change_path_length"] = hc_path_length

        r["is_responsive_heading_window_net"] = (
            not np.isnan(hc_window_net) and abs(hc_window_net) >= heading_threshold_deg
        )
        r["is_responsive_heading_max_dev"] = (
            not np.isnan(hc_max_dev) and hc_max_dev >= heading_threshold_deg
        )
        r["is_responsive_heading_post_saccade"] = (
            not np.isnan(hc_post_saccade)
            and abs(hc_post_saccade) >= heading_threshold_deg
        )
        r["is_responsive_heading_path_length"] = (
            not np.isnan(hc_path_length) and hc_path_length >= heading_threshold_deg
        )

        # ------------------------------------------------------------------
        # Method 5 — signed-peak saccade + heading change (default / is_responsive)
        # ------------------------------------------------------------------
        r["peak_ang_vel_signed_deg_s"] = saccade_peak
        r["is_responsive_combined"] = (
            r["is_responsive_saccade"] and r["is_responsive_heading"]
        )
        r["responsiveness_method"] = method
        r["is_responsive"] = bool(r[RESPONSIVENESS_METHOD_FIELDS[method]])

    # ------------------------------------------------------------------
    # H4/H5 — peak-aligned heading metrics (second pass)
    # For responsive flies: ref = detected peak angular velocity index.
    # For non-responsive flies: ref = average peak latency of responsive flies,
    #   so non-responsive trials are evaluated at the typical response time.
    # ------------------------------------------------------------------
    responsive_peak_times = [
        float(r["time"][r["_peak_global_idx"]])
        for r in responses
        if r.get("is_responsive") and r["_peak_global_idx"] >= 0
    ]
    mean_peak_time = float(np.mean(responsive_peak_times)) if responsive_peak_times else None

    _ref_frames = 10
    for r in responses:
        time = r["time"]
        end_t = r["end_expansion_time"]
        peak_global_idx = r.pop("_peak_global_idx")

        if peak_global_idx >= 0:
            ref_idx = peak_global_idx
        elif mean_peak_time is not None:
            ref_idx = int(np.argmin(np.abs(time - mean_peak_time)))
        else:
            ref_idx = int(np.argmin(np.abs(time - end_t)))

        heading = r.get("heading")
        if heading is not None:
            pre_pk = heading[max(0, ref_idx - _ref_frames) : ref_idx]
            post_pk = heading[ref_idx : ref_idx + _ref_frames]
            if len(pre_pk) > 0 and len(post_pk) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    h_pre_pk = circmean(pre_pk, low=-np.pi, high=np.pi)
                    h_post_pk = circmean(post_pk, low=-np.pi, high=np.pi)
                hc_peak_aligned = float(
                    np.rad2deg(
                        np.arctan2(
                            np.sin(h_post_pk - h_pre_pk),
                            np.cos(h_post_pk - h_pre_pk),
                        )
                    )
                )
            else:
                hc_peak_aligned = float("nan")

            xvel = r.get("xvel")
            yvel = r.get("yvel")
            hc_peak_vector = (
                _compute_heading_change_vector(xvel, yvel, ref_idx, window=_ref_frames)
                if xvel is not None and yvel is not None
                else float("nan")
            )
        else:
            hc_peak_aligned = float("nan")
            hc_peak_vector = float("nan")

        r["heading_change_peak_aligned"] = hc_peak_aligned
        r["heading_change_peak_vector"] = hc_peak_vector

    return responses
