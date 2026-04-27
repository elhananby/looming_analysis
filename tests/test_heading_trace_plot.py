from __future__ import annotations

from matplotlib.figure import Figure

from looming_analysis.plots import plot_heading_traces


def test_plot_heading_traces_returns_figure(responsive_trace_response):
    response = dict(responsive_trace_response)
    response["heading_deg"] = response["time"] * 10.0

    fig = plot_heading_traces(
        [response],
        col_by="stimulus_offset_deg",
        hue_by="group",
    )

    assert isinstance(fig, Figure)
