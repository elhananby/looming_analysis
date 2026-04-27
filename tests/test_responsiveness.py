from __future__ import annotations

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
