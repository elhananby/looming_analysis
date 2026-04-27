from __future__ import annotations

import numpy as np
import pytest

from looming_analysis import classify_responsiveness


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
    assert responses[0]["saccade_peak_ang_vel_signed_deg_s"] == -650.0
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
