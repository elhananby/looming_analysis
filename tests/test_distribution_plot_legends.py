from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from looming_analysis.plots.iti import plot_inter_trigger_interval
from looming_analysis.plots.peak_aligned import plot_response_latency


def _responses(groups: tuple[str, ...] = ("CS", "J52xKir2.1", "J64xKir2.1")) -> list[dict]:
    responses: list[dict] = []
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel = np.zeros_like(time)
    for group_index, group in enumerate(groups):
        for trial_index in range(4):
            responses.append(
                {
                    "group": group,
                    "inter_trigger_interval": 20.0 + 10.0 * group_index + trial_index,
                    "peak_latency_ms": 200.0 + 20.0 * group_index + trial_index,
                    "saccade_peak_time_ms": 200.0 + 20.0 * group_index + trial_index,
                    "is_responsive": True,
                    "time": time,
                    "ang_vel": ang_vel,
                }
            )
    return responses


def _legend_labels(fig) -> list[str]:
    labels: list[str] = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            labels.extend(text.get_text() for text in legend.get_texts())
    return labels


def test_inter_trigger_interval_legend_keeps_only_group_names():
    fig = plot_inter_trigger_interval(_responses())
    try:
        labels = _legend_labels(fig)
        assert labels == ["CS", "J52xKir2.1", "J64xKir2.1"]
        assert any("mean=" in text.get_text() for text in fig.axes[0].texts)
    finally:
        plt.close(fig)


def test_response_latency_legend_keeps_only_group_names():
    fig = plot_response_latency(_responses())
    try:
        labels = _legend_labels(fig)
        assert labels == ["CS", "J52xKir2.1", "J64xKir2.1"]
        assert any("mean=" in text.get_text() for text in fig.axes[0].texts)
    finally:
        plt.close(fig)


def test_response_latency_uses_precomputed_latency_without_traces():
    responses = [
        {
            "group": "CS",
            "peak_latency_ms": 210.0,
            "saccade_peak_time_ms": 210.0,
            "is_responsive": True,
        },
        {
            "group": "CS",
            "peak_latency_ms": 220.0,
            "saccade_peak_time_ms": 220.0,
            "is_responsive": True,
        },
        {
            "group": "CS",
            "peak_latency_ms": None,
            "saccade_peak_time_ms": None,
            "is_responsive": False,
        },
    ]

    fig = plot_response_latency(responses)
    try:
        assert _legend_labels(fig) == ["CS"]
    finally:
        plt.close(fig)
