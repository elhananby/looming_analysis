"""Looming analysis: load `.braidz`, extract responses, plot faceted results."""

from .extract import extract_responses, process_all_files, process_file_groups
from .io import load_braidz
from .plots import (
    plot_heading_changes,
    plot_peak_velocity,
    plot_responses,
    plot_responses_by_responsiveness,
    plot_responsiveness_rates,
    plot_turn_proportions,
)
from .responsiveness import classify_responsiveness, compute_turn_direction
from .signal import angular_velocity_from_velocity, calculate_angular_velocity

__all__ = [
    "angular_velocity_from_velocity",
    "calculate_angular_velocity",
    "classify_responsiveness",
    "extract_responses",
    "load_braidz",
    "compute_turn_direction",
    "plot_heading_changes",
    "plot_peak_velocity",
    "plot_responses",
    "plot_responses_by_responsiveness",
    "plot_responsiveness_rates",
    "plot_turn_proportions",
    "process_all_files",
    "process_file_groups",
]
