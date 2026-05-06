"""Tests for PR 3 sham-as-built-in-control behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from looming_analysis import classify_responsiveness
from looming_analysis.plots import plot_sham_vs_real
from looming_analysis.plots.traces import plot_responses
from looming_analysis.plots.rates import plot_responsiveness_rates
from looming_analysis.plots.turn_direction import plot_turn_proportions


def _make_responses(n_real: int = 4, n_sham: int = 2) -> list[dict]:
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel = np.zeros(len(time))
    responses = []
    for i in range(n_real):
        r = {
            "time": time.copy(),
            "ang_vel": ang_vel.copy(),
            "end_expansion_time": 0.30,
            "heading_change": 0.0,
            "is_responsive": True,
            "turn_direction": "right",
            "is_sham": False,
            "group": "CS",
            "stimulus_offset_deg": 0,
        }
        responses.append(r)
    for i in range(n_sham):
        r = {
            "time": time.copy(),
            "ang_vel": ang_vel.copy(),
            "end_expansion_time": 0.30,
            "heading_change": 0.0,
            "is_responsive": False,
            "turn_direction": None,
            "is_sham": True,
            "group": "CS",
            "stimulus_offset_deg": 0,
        }
        responses.append(r)
    return responses


def test_plot_responses_excludes_sham_by_default():
    responses = _make_responses(n_real=4, n_sham=2)
    fig = plot_responses(responses, hue_by="group")
    assert isinstance(fig, Figure)
    # The legend label should show n=4 (real only), not n=6
    ax = fig.axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("n=4" in t for t in legend_texts), f"Expected n=4 in legend, got: {legend_texts}"


def test_plot_responses_can_include_sham():
    responses = _make_responses(n_real=4, n_sham=2)
    fig = plot_responses(responses, hue_by="group", exclude_sham=False)
    ax = fig.axes[0]
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("n=6" in t for t in legend_texts), f"Expected n=6 in legend, got: {legend_texts}"


def test_plot_responsiveness_rates_excludes_sham_by_default():
    responses = _make_responses(n_real=4, n_sham=2)
    fig = plot_responsiveness_rates(responses)
    assert isinstance(fig, Figure)
    # n= label on bar should show 4, not 6
    ax = fig.axes[0]
    text_values = [t.get_text() for t in ax.texts]
    assert any("n=4" in t for t in text_values), f"Expected n=4 text, got: {text_values}"


def test_plot_turn_proportions_excludes_sham_by_default():
    responses = _make_responses(n_real=4, n_sham=2)
    fig = plot_turn_proportions(responses)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # n= label on stacked bar should show 4
    text_values = [t.get_text() for t in ax.texts]
    assert any("n=4" in t for t in text_values), f"Expected n=4 text, got: {text_values}"


def test_plot_sham_vs_real_returns_figure():
    responses = _make_responses(n_real=4, n_sham=2)
    fig = plot_sham_vs_real(responses)
    assert isinstance(fig, Figure)


def test_plot_sham_vs_real_raises_without_sham():
    responses = _make_responses(n_real=4, n_sham=0)
    with pytest.raises(ValueError, match="No sham trials"):
        plot_sham_vs_real(responses)


def test_is_sham_field_present_in_extract():
    import polars as pl
    from looming_analysis.extract import extract_responses

    frames = np.arange(0, 20, dtype=np.int64)
    df_kalman = pl.DataFrame({
        "obj_id": [1] * 20,
        "frame": frames,
        "xvel": np.ones(20),
        "yvel": np.zeros(20),
    })
    df_stim = pl.DataFrame({
        "obj_id": [1],
        "frame": [10],
        "timestamp": [0.1],
        "stimulus_offset_deg": [0],
        "expansion_duration_ms": [50],
        "sham": [False],
    })
    responses = extract_responses(df_kalman, df_stim, pre_frames=-5, post_frames=10, verbose=False)
    assert len(responses) == 1
    assert "is_sham" in responses[0]
    assert responses[0]["is_sham"] is False
