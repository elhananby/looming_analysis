"""Angular velocity computation from Cartesian velocity components."""

from __future__ import annotations

import numpy as np
from pynumdiff.smooth_finite_difference import butterdiff


def calculate_angular_velocity(
    xvel: np.ndarray,
    yvel: np.ndarray,
    dt: float = 0.01,
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


def angular_velocity_from_velocity(
    xvel: np.ndarray,
    yvel: np.ndarray,
    dt: float,
    min_speed: float = 0.1,
    order: int = 2,
    cutoff: float = 0.2,
) -> np.ndarray:
    """Compute angular velocity directly from Cartesian velocity components.

    Avoids `arctan2` + unwrapping entirely — numerically stable except near
    zero speed, where heading is undefined.

    Args:
        xvel: x-component of velocity.
        yvel: y-component of velocity.
        dt: Timestep between samples.
        min_speed: Speed below which output is NaN (heading undefined).
        order: Butterworth filter order.
        cutoff: Butterworth cutoff frequency.

    Returns:
        Angular velocity in rad/s, NaN where speed falls below `min_speed`.
    """
    speed_sq = xvel**2 + yvel**2

    _, dxdt = butterdiff(xvel, dt, [order, cutoff])
    _, dydt = butterdiff(yvel, dt, [order, cutoff])

    omega = (xvel * dydt - yvel * dxdt) / speed_sq
    omega[speed_sq < min_speed**2] = np.nan

    return omega
