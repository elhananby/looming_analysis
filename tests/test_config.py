from __future__ import annotations

import pytest

from looming_analysis.config import AnalysisConfig, ResponsivenessConfig


def test_analysis_config_converts_ms_to_frames():
    config = AnalysisConfig(pre_ms=100.0, post_ms=500.0, max_gap_ms=50.0)

    assert config.pre_frames == -10
    assert config.post_frames == 50
    assert config.max_gap_frames == 5


def test_analysis_config_rejects_negative_windows():
    with pytest.raises(ValueError, match="pre_ms must be >= 0"):
        AnalysisConfig(pre_ms=-1.0)

    with pytest.raises(ValueError, match="post_ms must be > 0"):
        AnalysisConfig(post_ms=0.0)


def test_responsiveness_config_as_kwargs():
    config = ResponsivenessConfig(
        method="heading",
        threshold_deg_s=500.0,
        heading_threshold_deg=45.0,
        baseline_window_ms=(-90.0, -10.0),
    )

    assert config.as_kwargs() == {
        "threshold_deg_s": 500.0,
        "window_ms": 200.0,
        "zscore_k": 3.0,
        "baseline_window_ms": (-90.0, -10.0),
        "heading_threshold_deg": 45.0,
        "impulse_threshold_deg": 20.0,
        "method": "heading",
        "rdp_epsilon": 0.5,
    }
