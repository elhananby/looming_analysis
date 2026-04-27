"""Angular velocity computation from Cartesian velocity components."""

from __future__ import annotations

import numpy as np
from pynumdiff.smooth_finite_difference import butterdiff

from ._types import DT_SECONDS


def calculate_angular_velocity(
    xvel: np.ndarray,
    yvel: np.ndarray,
    dt: float = DT_SECONDS,
    params: list | None = None,
) -> np.ndarray:
    """Compute angular velocity by differentiating unwrapped arctan2(yvel, xvel).

    Uses `pynumdiff.butterdiff` (Butterworth-smoothed finite difference) to
    get a stable derivative of the unwrapped heading angle.

    Args:
        xvel: x-component of velocity.
        yvel: y-component of velocity.
        dt: Timestep between samples. Defaults to 0.01s (100 Hz).
        params: `[order, cutoff]` for the Butterworth filter. Defaults to
            `[1, 0.1]`.

    Returns:
        Angular velocity in rad/s.
    """
    if params is None:
        params = [1, 0.1]
    theta = np.arctan2(yvel, xvel)
    theta_unwrap = np.unwrap(theta)
    _, angular_velocity = butterdiff(theta_unwrap, dt=dt, params=params)
    return angular_velocity
