"""Smoke tests for plots added in the heading-change / RDP feature."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from looming_analysis import classify_responsiveness
from looming_analysis.plots import (
    plot_heading_change_comparison,
    plot_heading_changes_polar,
    plot_rdp_debug,
)


def _classified_responses(n: int = 8) -> list[dict]:
    rng = np.random.default_rng(0)
    responses = []
    time = np.arange(-0.1, 0.6, 0.01)
    for i in range(n):
        ang_vel_deg_s = np.zeros(len(time))
        headings = np.zeros(len(time))
        peak_idx = np.argmin(np.abs(time - 0.30))
        if i < n // 2:
            ang_vel_deg_s[peak_idx - 2 : peak_idx + 3] = 600.0
            headings[peak_idx:] = np.deg2rad(60.0)
        responses.append({
            "time": time.copy(),
            "ang_vel": np.deg2rad(ang_vel_deg_s),
            "heading": headings,
            "xvel": np.ones(len(time)) * 0.1 + rng.normal(0, 0.01, len(time)),
            "yvel": rng.normal(0, 0.01, len(time)),
            "end_expansion_time": 0.30,
            "heading_change": 60.0 if i < n // 2 else 5.0,
            "group": "control",
        })
    return classify_responsiveness(
        responses,
        threshold_deg_s=500.0,
        heading_threshold_deg=30.0,
        baseline_window_ms=(-90.0, -10.0),
    )


def test_plot_heading_changes_polar_returns_figure():
    responses = _classified_responses()
    fig = plot_heading_changes_polar(responses)
    assert isinstance(fig, Figure)


def test_plot_heading_changes_polar_with_hue_by():
    responses = _classified_responses()
    fig = plot_heading_changes_polar(responses, hue_by="group")
    assert isinstance(fig, Figure)


def test_plot_heading_change_comparison_returns_figure():
    responses = _classified_responses()
    fig = plot_heading_change_comparison(responses)
    assert isinstance(fig, Figure)


def test_plot_rdp_debug_returns_figure():
    responses = _classified_responses(n=8)
    fig = plot_rdp_debug(responses, n_responsive=2, n_nonresponsive=2)
    assert isinstance(fig, Figure)
