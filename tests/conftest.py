from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def responsive_trace_response() -> dict:
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    ang_vel_deg_s[(time >= 0.20) & (time <= 0.24)] = 600.0
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "end_expansion_time": 0.30,
        "heading_change": 60.0,
        "stimulus_offset_deg": 0,
        "group": "control",
    }


@pytest.fixture
def heading_only_response() -> dict:
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "end_expansion_time": 0.30,
        "heading_change": 60.0,
        "stimulus_offset_deg": 0,
        "group": "control",
    }
