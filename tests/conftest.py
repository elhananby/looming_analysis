from __future__ import annotations

import numpy as np
import pytest


def _make_vel_with_heading_change(time: np.ndarray, turn_idx: int, heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (xvel, yvel) with a heading change of heading_deg at turn_idx."""
    speed = 0.1
    xvel = np.full(len(time), speed)
    yvel = np.zeros(len(time))
    angle_rad = np.deg2rad(heading_deg)
    xvel[turn_idx:] = speed * np.cos(angle_rad)
    yvel[turn_idx:] = speed * np.sin(angle_rad)
    return xvel, yvel


@pytest.fixture
def responsive_trace_response() -> dict:
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    ang_vel_deg_s[(time >= 0.20) & (time <= 0.24)] = 600.0
    peak_idx = int(np.argmin(np.abs(time - 0.20)))
    xvel, yvel = _make_vel_with_heading_change(time, peak_idx, 60.0)
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
        "heading_change": 60.0,
        "stimulus_offset_deg": 0,
        "group": "control",
    }


@pytest.fixture
def heading_only_response() -> dict:
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    # No saccade; heading change at end_expansion_time used as fallback ref.
    turn_idx = int(np.argmin(np.abs(time - 0.30)))
    xvel, yvel = _make_vel_with_heading_change(time, turn_idx, 60.0)
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
        "heading_change": 60.0,
        "stimulus_offset_deg": 0,
        "group": "control",
    }


@pytest.fixture
def negative_saccade_response() -> dict:
    time = np.arange(-0.1, 0.5, 0.01)
    ang_vel_deg_s = np.zeros_like(time)
    center = np.argmin(np.abs(time - 0.30))
    ang_vel_deg_s[center - 2 : center + 3] = -650.0
    xvel, yvel = _make_vel_with_heading_change(time, int(center), -60.0)
    return {
        "time": time,
        "ang_vel": np.deg2rad(ang_vel_deg_s),
        "xvel": xvel,
        "yvel": yvel,
        "end_expansion_time": 0.30,
        "heading_change": -60.0,
        "stimulus_offset_deg": 0,
        "group": "control",
    }
