from __future__ import annotations

import pytest

from looming_analysis.plots import (
    plot_responses_by_responsiveness,
    plot_responsiveness_rates,
    plot_turn_proportions,
)


def test_rates_requires_responsiveness_field(responsive_trace_response):
    response = dict(responsive_trace_response)

    with pytest.raises(ValueError, match="classify_responsiveness"):
        plot_responsiveness_rates([response])


def test_responsive_traces_requires_responsiveness_field(responsive_trace_response):
    response = dict(responsive_trace_response)

    with pytest.raises(ValueError, match="classify_responsiveness"):
        plot_responses_by_responsiveness([response])


def test_turn_plot_requires_turn_direction_field(responsive_trace_response):
    response = dict(responsive_trace_response)

    with pytest.raises(ValueError, match="compute_turn_direction"):
        plot_turn_proportions([response])
