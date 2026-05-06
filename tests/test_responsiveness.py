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


def test_combined_responsiveness_requires_saccade_and_heading(
    responsive_trace_response,
    heading_only_response,
):
    no_heading = {**responsive_trace_response, "heading_change": 10.0}

    responses = classify_responsiveness(
        [no_heading, heading_only_response],
        method="combined",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert responses[0]["is_responsive_saccade"] is True
    assert responses[0]["is_responsive_heading"] is False
    assert responses[0]["is_responsive_combined"] is False
    assert responses[1]["is_responsive_saccade"] is False
    assert responses[1]["is_responsive_heading"] is True
    assert responses[1]["is_responsive_combined"] is False


def test_combined_responsiveness_requires_find_peaks_saccade():
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.linspace(0.0, 700.0, len(time))
    response = {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "end_expansion_time": 0.30,
        "heading_change": 60.0,
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
    ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = (
        600.0  # 5-sample / 50ms rectangular saccade
    )
    response = {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "end_expansion_time": 0.30,
        "heading_change": 60.0,
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


def _response_with_heading(heading_deg_at_peak: float = 60.0) -> dict:
    """Synthetic response with a heading turn and a 5-sample saccade."""
    time = np.arange(-0.1, 0.6, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    headings = np.zeros(len(time))
    peak_idx = np.argmin(np.abs(time - 0.30))
    ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = 600.0
    # Simulate heading rotating during the saccade
    headings[peak_idx:] = np.deg2rad(heading_deg_at_peak)
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "heading": headings,
        "end_expansion_time": 0.30,
        "heading_change": heading_deg_at_peak,
    }


def test_heading_change_fields_present_when_heading_array_available():
    r = _response_with_heading(60.0)
    responses = classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    out = responses[0]
    for field in (
        "heading_change_window_net",
        "heading_change_max_dev",
        "heading_change_post_saccade",
        "heading_change_path_length",
    ):
        assert field in out, f"missing field: {field}"
        assert isinstance(out[field], float), f"{field} should be float"


def test_heading_change_post_saccade_nan_when_no_saccade():
    time = np.arange(-0.1, 0.6, 0.01)
    r = {
        "time": time,
        "ang_vel": np.zeros(len(time)),
        "heading": np.zeros(len(time)),
        "end_expansion_time": 0.30,
        "heading_change": 0.0,
    }
    responses = classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    assert np.isnan(responses[0]["heading_change_post_saccade"])


def test_heading_change_fields_nan_without_heading_array(responsive_trace_response):
    responses = classify_responsiveness(
        [responsive_trace_response],
        threshold_deg_s=500.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    for field in (
        "heading_change_window_net",
        "heading_change_max_dev",
        "heading_change_post_saccade",
        "heading_change_path_length",
    ):
        assert np.isnan(responses[0][field]), (
            f"{field} should be NaN without heading array"
        )


# ---------------------------------------------------------------------------
# Peak-aligned second-pass metrics
# ---------------------------------------------------------------------------

def _response_with_vel(heading_change_deg: float = 60.0) -> dict:
    """Synthetic response with xvel/yvel and a heading turn."""
    time = np.arange(-0.1, 0.6, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    headings = np.zeros(len(time))
    xvel = np.ones(len(time)) * 0.1
    yvel = np.zeros(len(time))
    peak_idx = np.argmin(np.abs(time - 0.30))
    ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = 600.0
    headings[peak_idx:] = np.deg2rad(heading_change_deg)
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "heading": headings,
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
        "heading_change": heading_change_deg,
    }


def test_peak_aligned_metrics_present_after_classify():
    r = _response_with_vel(60.0)
    responses = classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    out = responses[0]
    for field in ("heading_change_peak_aligned", "heading_change_peak_vector", "heading_change_rdp"):
        assert field in out, f"missing field: {field}"
        assert isinstance(out[field], float), f"{field} should be float"


def test_peak_aligned_and_peak_vector_finite_for_responsive():
    r = _response_with_vel(60.0)
    responses = classify_responsiveness(
        [r],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    out = responses[0]
    assert not np.isnan(out["heading_change_peak_aligned"])
    assert not np.isnan(out["heading_change_peak_vector"])


def test_nonresponsive_uses_population_mean_ref():
    # A non-responsive trial (no saccade) should still get finite peak-aligned
    # metrics when there is at least one responsive trial setting a mean latency.
    responsive = _response_with_vel(60.0)
    nonresponsive = {
        "time": responsive["time"].copy(),
        "ang_vel": np.zeros(len(responsive["time"])),
        "heading": np.zeros(len(responsive["time"])),
        "xvel": np.ones(len(responsive["time"])) * 0.1,
        "yvel": np.zeros(len(responsive["time"])),
        "end_expansion_time": 0.30,
        "heading_change": 5.0,
    }
    responses = classify_responsiveness(
        [responsive, nonresponsive],
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )
    nr = responses[1]
    assert nr["is_responsive"] is False
    assert not np.isnan(nr["heading_change_peak_aligned"]), (
        "non-responsive fly should use mean peak latency as ref"
    )


# ---------------------------------------------------------------------------
# compute_turn_direction
# ---------------------------------------------------------------------------

def test_compute_turn_direction_right_from_saccade(responsive_trace_response):
    r = dict(responsive_trace_response)
    r["ang_vel"] = np.abs(r["ang_vel"])  # positive angular velocity
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
