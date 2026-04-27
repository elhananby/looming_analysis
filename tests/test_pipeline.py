from __future__ import annotations

import polars as pl

from looming_analysis.config import AnalysisConfig, ResponsivenessConfig
from looming_analysis.pipeline import (
    AnalysisResult,
    normalize_file_selection,
    run_analysis,
)


def test_analysis_result_exposes_scalar_and_long_frames(responsive_trace_response):
    result = AnalysisResult([responsive_trace_response])

    scalar = result.to_dataframe(kind="scalar")
    long = result.to_dataframe(kind="long")

    assert isinstance(scalar, pl.DataFrame)
    assert scalar.height == 1
    assert long.height == len(responsive_trace_response["time"])


def test_normalize_file_selection_accepts_single_group_list():
    groups = normalize_file_selection(["a.braidz", "b.braidz"])

    assert groups == {"experiment": ["a.braidz", "b.braidz"]}


def test_normalize_file_selection_accepts_group_mapping():
    groups = normalize_file_selection({"CS": ["a.braidz"], "DNp03": ["b.braidz"]})

    assert groups == {"CS": ["a.braidz"], "DNp03": ["b.braidz"]}


def test_run_analysis_calls_pipeline_steps(monkeypatch, responsive_trace_response):
    calls = {}

    def fake_process_file_groups(file_groups, **kwargs):
        calls["file_groups"] = file_groups
        calls["extract_kwargs"] = kwargs
        return [responsive_trace_response]

    def fake_classify_responsiveness(responses, **kwargs):
        calls["responsiveness_kwargs"] = kwargs
        responses[0]["is_responsive"] = True
        return responses

    def fake_compute_turn_direction(responses):
        calls["turn_direction"] = True
        responses[0]["turn_direction"] = "right"
        return responses

    monkeypatch.setattr(
        "looming_analysis.pipeline.process_file_groups", fake_process_file_groups
    )
    monkeypatch.setattr(
        "looming_analysis.pipeline.classify_responsiveness",
        fake_classify_responsiveness,
    )
    monkeypatch.setattr(
        "looming_analysis.pipeline.compute_turn_direction", fake_compute_turn_direction
    )

    result = run_analysis(
        {"control": ["a.braidz"]},
        analysis=AnalysisConfig(pre_ms=100.0, post_ms=500.0, cache_dir=None),
        responsiveness=ResponsivenessConfig(method="heading"),
        verbose=False,
    )

    assert isinstance(result, AnalysisResult)
    assert result.responses[0]["turn_direction"] == "right"
    assert calls["extract_kwargs"]["pre_frames"] == -10
    assert calls["extract_kwargs"]["post_frames"] == 50
    assert calls["extract_kwargs"]["cache_dir"] is None
    assert calls["responsiveness_kwargs"]["method"] == "heading"
    assert calls["turn_direction"] is True
