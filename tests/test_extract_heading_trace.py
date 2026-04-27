from __future__ import annotations

import numpy as np
import polars as pl

from looming_analysis.extract import extract_responses


def test_extract_responses_includes_heading_trace():
    frames = np.arange(0, 20, dtype=np.int64)
    df_kalman = pl.DataFrame(
        {
            "obj_id": [1] * len(frames),
            "frame": frames,
            "xvel": np.ones(len(frames)),
            "yvel": np.zeros(len(frames)),
        }
    )
    df_stim = pl.DataFrame(
        {
            "obj_id": [1],
            "frame": [10],
            "timestamp": [0.1],
            "stimulus_offset_deg": [90],
            "expansion_duration_ms": [50],
        }
    )

    responses = extract_responses(
        df_kalman,
        df_stim,
        pre_frames=-5,
        post_frames=10,
        verbose=False,
    )

    assert len(responses) == 1
    response = responses[0]
    assert response["heading"].shape == response["time"].shape
    assert response["heading_deg"].shape == response["time"].shape
    assert np.allclose(response["heading"], 0.0)
    assert np.allclose(response["heading_deg"], 0.0)
