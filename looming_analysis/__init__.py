"""Looming analysis: load `.braidz`, extract responses, plot faceted results."""

from .dataframe import responses_to_dataframe
from .config import AnalysisConfig, ResponsivenessConfig
from .extract import extract_responses, process_all_files, process_file_groups
from .files import find_braidz
from .io import load_braidz
from .plots import (
    plot_heading_changes,
    plot_heading_changes_polar,
    plot_heading_traces,
    plot_inter_trigger_interval,
    plot_latency_by_direction,
    plot_peak_aligned_traces,
    plot_peak_velocity,
    plot_response_latency,
    plot_responses,
    plot_responses_by_responsiveness,
    plot_responsiveness_rates,
    plot_screen_position_effect,
    plot_turn_proportions,
)
from .pipeline import AnalysisResult, normalize_file_selection, run_analysis
from .plots.peak_aligned import compute_peak_latency
from .responsiveness import classify_responsiveness, compute_turn_direction
from .signal import calculate_angular_velocity

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "calculate_angular_velocity",
    "classify_responsiveness",
    "compute_peak_latency",
    "compute_turn_direction",
    "extract_responses",
    "find_braidz",
    "load_braidz",
    "normalize_file_selection",
    "responses_to_dataframe",
    "plot_heading_changes",
    "plot_heading_changes_polar",
    "plot_heading_traces",
    "plot_inter_trigger_interval",
    "plot_latency_by_direction",
    "plot_peak_aligned_traces",
    "plot_peak_velocity",
    "plot_response_latency",
    "plot_responses",
    "plot_responses_by_responsiveness",
    "plot_responsiveness_rates",
    "plot_screen_position_effect",
    "plot_turn_proportions",
    "process_all_files",
    "process_file_groups",
    "ResponsivenessConfig",
    "run_analysis",
]
