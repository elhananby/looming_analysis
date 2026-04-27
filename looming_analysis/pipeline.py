"""Beginner-facing analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from matplotlib.figure import Figure

from ._types import Response
from .config import AnalysisConfig, ResponsivenessConfig
from .dataframe import responses_to_dataframe
from .extract import process_file_groups
from .plots import (
    plot_heading_changes,
    plot_heading_traces,
    plot_peak_velocity,
    plot_responses,
    plot_responses_by_responsiveness,
    plot_responsiveness_rates,
    plot_turn_proportions,
    plot_heading_change_comparison,
)
from .responsiveness import classify_responsiveness, compute_turn_direction


@dataclass
class AnalysisResult:
    """Container returned by `run_analysis`."""

    responses: list[Response]

    def to_dataframe(
        self, kind: Literal["scalar", "long"] = "scalar", backend: str = "polars"
    ):
        return responses_to_dataframe(self.responses, kind=kind, backend=backend)

    def plot_traces(self, **kwargs) -> Figure:
        return plot_responses(self.responses, **kwargs)

    def plot_heading_traces(self, **kwargs) -> Figure:
        return plot_heading_traces(self.responses, **kwargs)

    def plot_responsiveness_rates(self, **kwargs) -> Figure:
        return plot_responsiveness_rates(self.responses, **kwargs)

    def plot_responsive_traces(self, **kwargs) -> Figure:
        return plot_responses_by_responsiveness(self.responses, **kwargs)

    def plot_heading_changes(self, **kwargs) -> Figure:
        return plot_heading_changes(self.responses, **kwargs)

    def plot_heading_change_comparison(self, **kwargs) -> Figure:
        return plot_heading_change_comparison(self.responses, **kwargs)

    def plot_peak_velocity(self, **kwargs) -> Figure:
        return plot_peak_velocity(self.responses, **kwargs)

    def plot_turn_proportions(self, **kwargs) -> Figure:
        return plot_turn_proportions(self.responses, **kwargs)


def normalize_file_selection(
    selected_files: list[str] | dict[str, list[str]],
    *,
    default_group: str = "experiment",
) -> dict[str, list[str]]:
    """Normalize selected files into the grouped representation used internally."""
    if isinstance(selected_files, dict):
        return {
            str(group): [str(path) for path in paths]
            for group, paths in selected_files.items()
        }
    return {default_group: [str(path) for path in selected_files]}


def run_analysis(
    selected_files: list[str] | dict[str, list[str]],
    *,
    analysis: AnalysisConfig | None = None,
    responsiveness: ResponsivenessConfig | None = None,
    compute_turns: bool = True,
    verbose: bool = True,
) -> AnalysisResult:
    """Run extraction, responsiveness classification, and turn direction analysis."""
    analysis = analysis or AnalysisConfig()
    responsiveness = responsiveness or ResponsivenessConfig()
    file_groups = normalize_file_selection(selected_files)

    responses = process_file_groups(
        file_groups,
        pre_frames=analysis.pre_frames,
        post_frames=analysis.post_frames,
        verbose=verbose,
        debug=False,
        heading_ref_frames=analysis.heading_ref_frames,
        max_gap_frames=analysis.max_gap_frames,
        include_sham=analysis.include_sham,
        cache_dir=analysis.cache_dir,
    )
    classify_responsiveness(responses, **responsiveness.as_kwargs())
    if compute_turns:
        compute_turn_direction(responses)
    return AnalysisResult(responses)
