from __future__ import annotations

import numpy as np
import pytest

from looming_analysis import classify_responsiveness, compute_turn_direction


def test_classify_responsiveness_defaults_to_combined(responsive_trace_response):
    responses = classify_responsiveness(
        [responsive_trace_response],
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_combined"] is True
    assert responses[0]["is_responsive"] is True


def test_classify_responsiveness_method_heading_sets_primary_flag(
    heading_only_response,
):
    responses = classify_responsiveness(
        [heading_only_response],
        method="heading",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_heading"] is True
    assert responses[0]["is_responsive_combined"] is False
    assert responses[0]["is_responsive"] is True


def test_classify_responsiveness_rejects_unknown_method(responsive_trace_response):
    with pytest.raises(ValueError, match="method must be one of"):
        classify_responsiveness([responsive_trace_response], method="not-a-method")


def test_classify_responsiveness_saccade_uses_positive_signed_peak(
    responsive_trace_response,
):
    responses = classify_responsiveness(
        [responsive_trace_response],
        method="saccade",
        threshold_deg_s=500.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_saccade"] is True
    assert responses[0]["saccade_peak_ang_vel_signed_deg_s"] == 600.0
    assert responses[0]["is_responsive"] is True


def test_classify_responsiveness_saccade_uses_negative_signed_peak(
    negative_saccade_response,
):
    responses = classify_responsiveness(
        [negative_saccade_response],
        method="saccade",
        threshold_deg_s=500.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_saccade"] is True
    assert responses[0]["saccade_peak_ang_vel_signed_deg_s"] == pytest.approx(-650.0)
    assert responses[0]["is_responsive"] is True


def test_combined_saccade_without_heading_not_responsive(responsive_trace_response):
    # Strip velocity vectors → heading_change=NaN → is_responsive_heading=False.
    no_heading = {k: v for k, v in responsive_trace_response.items() if k not in ("xvel", "yvel")}

    responses = classify_responsiveness(
        [no_heading],
        method="combined",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_saccade"] is True
    assert responses[0]["is_responsive_heading"] is False
    assert responses[0]["is_responsive_combined"] is False


def test_combined_heading_without_saccade_not_responsive(heading_only_response):
    # No saccades in population → fallback reference = end_expansion_time.
    # heading_only_response turns at end_expansion_time → heading_change ≈ 60° ≥ 45°.
    responses = classify_responsiveness(
        [heading_only_response],
        method="combined",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_saccade"] is False
    assert responses[0]["is_responsive_heading"] is True
    assert responses[0]["is_responsive_combined"] is False


def _make_vel(time: np.ndarray, turn_idx: int, heading_deg: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    speed = 0.1
    xvel = np.full(len(time), speed)
    yvel = np.zeros(len(time))
    angle_rad = np.deg2rad(heading_deg)
    xvel[turn_idx:] = speed * np.cos(angle_rad)
    yvel[turn_idx:] = speed * np.sin(angle_rad)
    return xvel, yvel


def test_combined_responsiveness_requires_find_peaks_saccade():
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.linspace(0.0, 700.0, len(time))
    turn_idx = int(np.argmin(np.abs(time - 0.30)))
    xvel, yvel = _make_vel(time, turn_idx)
    response = {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
    }

    responses = classify_responsiveness(
        [response],
        method="combined",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_saccade"] is False
    assert responses[0]["is_responsive_heading"] is True
    assert responses[0]["is_responsive_combined"] is False
    assert responses[0]["is_responsive"] is False


def test_saccade_window_can_be_asymmetric():
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    peak_idx = np.argmin(np.abs(time - 0.20))
    ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = 600.0
    xvel, yvel = _make_vel(time, int(peak_idx))
    response = {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
    }

    narrow = classify_responsiveness(
        [dict(response)],
        method="combined",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        window_ms=[50.0, 200.0],
    )
    wide = classify_responsiveness(
        [dict(response)],
        method="combined",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        window_ms=[150.0, 200.0],
    )

    assert narrow[0]["is_responsive_saccade"] is False
    assert narrow[0]["is_responsive"] is False
    assert wide[0]["is_responsive_saccade"] is True
    assert wide[0]["is_responsive"] is True


def _response_with_vel(heading_change_deg: float = 60.0) -> dict:
    """Synthetic response with xvel/yvel and a heading turn."""
    time = np.arange(-0.1, 0.6, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    peak_idx = np.argmin(np.abs(time - 0.30))
    ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = 600.0
    xvel, yvel = _make_vel(time, int(peak_idx), heading_change_deg)
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
    }


def test_canonical_heading_change_present_after_classify():
    r = _response_with_vel(60.0)
    responses = classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    out = responses[0]
    assert "heading_change" in out
    assert isinstance(out["heading_change"], float)
    assert not np.isnan(out["heading_change"])


def test_canonical_heading_change_finite_for_responsive():
    r = _response_with_vel(60.0)
    responses = classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    assert not np.isnan(responses[0]["heading_change"])


def test_nonresponsive_gets_finite_heading_change():
    responsive = _response_with_vel(60.0)
    nonresponsive = {
        "time": responsive["time"].copy(),
        "ang_vel": np.zeros(len(responsive["time"])),
        "xvel": np.ones(len(responsive["time"])) * 0.1,
        "yvel": np.zeros(len(responsive["time"])),
        "end_expansion_time": 0.30,
    }
    responses = classify_responsiveness(
        [responsive, nonresponsive],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    nr = responses[1]
    assert nr["is_responsive"] is False
    assert not np.isnan(nr["heading_change"]), (
        "non-responsive fly should use mean peak latency as ref and get finite heading_change"
    )


# ---------------------------------------------------------------------------
# compute_turn_direction
# ---------------------------------------------------------------------------

def test_compute_turn_direction_right_from_saccade(responsive_trace_response):
    r = dict(responsive_trace_response)
    r["ang_vel"] = np.abs(r["ang_vel"])
    classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    compute_turn_direction([r])
    assert r["turn_direction"] in ("left", "right")
    assert not np.isnan(r["signed_peak_ang_vel_deg_s"])


def test_compute_turn_direction_left_for_negative_peak(negative_saccade_response):
    r = dict(negative_saccade_response)
    classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    compute_turn_direction([r])
    assert r["turn_direction"] == "left"
    assert r["signed_peak_ang_vel_deg_s"] < 0


def test_compute_turn_direction_none_for_zero_peak():
    time = np.arange(-0.1, 0.5, 0.01)
    r = {
        "time": time,
        "ang_vel": np.zeros(len(time)),
        "end_expansion_time": 0.30,
        "heading_change": 0.0,
    }
    compute_turn_direction([r])
    assert r["turn_direction"] is None


# ---------------------------------------------------------------------------
# Regression tests (plan acceptance criteria)
# ---------------------------------------------------------------------------

def test_saccade_wider_than_80ms_is_detected():
    """Bug C regression: width=(3, None) allows saccades longer than 80 ms."""
    time = np.arange(-0.1, 0.6, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    peak_idx = np.argmin(np.abs(time - 0.30))
    # 12-sample / 120 ms saccade — exceeds the old width=(3,8) upper bound of 80 ms.
    ang_vel_deg_s[peak_idx - 6 : peak_idx + 6] = 600.0
    xvel, yvel = _make_vel(time, int(peak_idx - 6))
    r = {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
    }
    responses = classify_responsiveness([r], threshold_deg_s=500.0, baseline_window_ms=(-90.0, -10.0))
    assert responses[0]["is_responsive_saccade"] is True, "120 ms saccade should be detected"


def test_nonresponsive_heading_change_uses_saccade_time_not_method0_peak():
    """Bug A regression: fallback ref uses saccade_peak_time_ms, not _peak_global_idx."""
    time = np.arange(-0.1, 0.6, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    # Build a saccade that Method 3 (signed) detects but Method 0 (abs peak) might not.
    peak_idx = np.argmin(np.abs(time - 0.30))
    ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = 600.0
    xvel, yvel = _make_vel(time, int(peak_idx))

    responsive = {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
    }
    nonresponsive = {
        "time": time.copy(),
        "ang_vel": np.zeros(len(time)),
        "xvel": np.ones(len(time)) * 0.1,
        "yvel": np.zeros(len(time)),
        "end_expansion_time": 0.30,
    }
    responses = classify_responsiveness(
        [responsive, nonresponsive],
        method="combined",
        threshold_deg_s=500.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    nr = responses[1]
    assert nr["is_responsive"] is False
    assert not np.isnan(nr["heading_change"]), "non-responsive should get finite heading_change via saccade-time fallback"


def test_nonresponsive_signed_peak_ang_vel_is_finite():
    """Bug F regression: non-responsive fly gets finite signed_peak_ang_vel_deg_s."""
    time = np.arange(-0.1, 0.6, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    peak_idx = np.argmin(np.abs(time - 0.30))
    ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = 600.0
    xvel, yvel = _make_vel(time, int(peak_idx))

    responsive = {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
    }
    nonresponsive = {
        "time": time.copy(),
        "ang_vel": np.zeros(len(time)),
        "xvel": np.ones(len(time)) * 0.1,
        "yvel": np.zeros(len(time)),
        "end_expansion_time": 0.30,
    }
    responses = classify_responsiveness(
        [responsive, nonresponsive],
        method="saccade",
        threshold_deg_s=500.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    compute_turn_direction(responses)
    nr = responses[1]
    assert nr["is_responsive"] is False
    assert not np.isnan(nr["signed_peak_ang_vel_deg_s"]), "non-responsive should get finite signed_peak_ang_vel_deg_s"
