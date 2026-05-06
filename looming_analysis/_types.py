"""Shared type aliases and constants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import Optional


DT_SECONDS = 0.01  # 100 Hz sampling rate


def _circ_diff_deg(
    a_rad: float | np.ndarray,
    b_rad: float | np.ndarray,
) -> float | np.ndarray:
    """Signed circular difference (a − b) in degrees, range (−180, 180]."""
    return np.rad2deg(np.arctan2(np.sin(a_rad - b_rad), np.cos(a_rad - b_rad)))


class Response(dict):
    """Per-stimulus response dict.

    Mandatory keys set by :func:`extract_responses`:

    * ``ang_vel``  – angular velocity trace (rad/s), shape (N,)
    * ``xvel``     – x-velocity trace (m/s), shape (N,)
    * ``yvel``     – y-velocity trace (m/s), shape (N,)
    * ``heading``  – heading angle trace (rad), shape (N,)
    * ``heading_deg`` – heading in degrees, shape (N,)
    * ``time``     – time axis relative to stimulus onset (s), shape (N,)
    * ``heading_change`` – circmean-based net heading change (deg)
    * ``heading_change_stim_vector`` – vector-based heading change at stimulus midpoint (deg)
    * ``end_expansion_time`` – duration of stimulus expansion (s)

    Additional keys are added by :func:`classify_responsiveness` (``is_responsive``,
    ``saccade_peak_time_ms``, ``heading_change_peak_aligned``, etc.) and by
    :func:`compute_turn_direction` (``turn_direction``, ``signed_peak_ang_vel_deg_s``).

    Because the actual key set varies with optional processing steps, this class
    extends ``dict`` rather than using ``TypedDict``, so it accepts arbitrary keys
    at runtime while still serving as a descriptive annotation target.
    """
