"""Beginner-facing analysis pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from matplotlib.figure import Figure

from ._types import Response
from .config import AnalysisConfig, ResponsivenessConfig
from .dataframe import responses_to_dataframe
from .extract import process_file_groups
from .plots import (
    plot_heading_changes,
    plot_heading_changes_polar,
    plot_heading_traces,
    plot_inter_trigger_interval,
    plot_peak_aligned_traces,
    plot_peak_velocity,
    plot_responses,
    plot_responses_by_responsiveness,
    plot_responsiveness_rates,
    plot_screen_position_effect,
    plot_sham_vs_real,
    plot_turn_proportions,
)
from .plots.peak_aligned import compute_peak_latency, plot_latency_by_direction, plot_response_latency
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

    def plot_heading_changes_polar(self, **kwargs) -> Figure:
        return plot_heading_changes_polar(self.responses, **kwargs)

    def plot_peak_velocity(self, **kwargs) -> Figure:
        return plot_peak_velocity(self.responses, **kwargs)

    def plot_turn_proportions(self, **kwargs) -> Figure:
        return plot_turn_proportions(self.responses, **kwargs)

    def plot_inter_trigger_interval(self, **kwargs) -> Figure:
        return plot_inter_trigger_interval(self.responses, **kwargs)

    def plot_peak_aligned_traces(self, **kwargs) -> Figure:
        return plot_peak_aligned_traces(self.responses, **kwargs)

    def plot_response_latency(self, **kwargs) -> Figure:
        return plot_response_latency(self.responses, **kwargs)

    def plot_latency_by_direction(self, **kwargs) -> Figure:
        return plot_latency_by_direction(self.responses, **kwargs)

    def plot_screen_position_effect(self, **kwargs) -> Figure:
        return plot_screen_position_effect(self.responses, **kwargs)

    def plot_sham_vs_real(self, **kwargs) -> Figure:
        return plot_sham_vs_real(self.responses, **kwargs)

    def compute_peak_latency(self, **kwargs) -> "AnalysisResult":
        compute_peak_latency(self.responses, **kwargs)
        return self

    def filter_by_iti(self, min_iti_s: float, *, verbose: bool = True) -> "AnalysisResult":
        """Return a new AnalysisResult dropping trials with ITI < min_iti_s."""
        filtered = filter_by_iti(self.responses, min_iti_s, verbose=verbose)
        return AnalysisResult(filtered)


def filter_by_iti(
    responses: list[Response],
    min_iti_s: float,
    *,
    verbose: bool = True,
) -> list[Response]:
    """Return responses where inter_trigger_interval >= min_iti_s.

    Trials with a missing or NaN ITI (the first trigger in each file) are
    always kept — they cannot have violated the threshold.

    Args:
        responses: Response list with ``inter_trigger_interval`` set.
        min_iti_s: Minimum allowed ITI in seconds.  Trials triggered sooner
            after the previous one are dropped.
        verbose: If True, print a summary of how many trials were dropped.

    Returns:
        Filtered list (new list, original is not modified).
    """
    kept, dropped = [], []
    for r in responses:
        iti = r.get("inter_trigger_interval")
        if iti is None or (isinstance(iti, float) and math.isnan(iti)):
            kept.append(r)
        elif float(iti) < min_iti_s:
            dropped.append(r)
        else:
            kept.append(r)
    if verbose and dropped:
        import numpy as np
        total = len(responses)
        print(
            f"filter_by_iti: dropped {len(dropped)}/{total} trials "
            f"with ITI < {min_iti_s} s  "
            f"(kept {len(kept)}, {100*len(kept)/total:.1f}%)"
        )
        itv_dropped = [float(r["inter_trigger_interval"]) for r in dropped
                       if r.get("inter_trigger_interval") is not None]
        if itv_dropped:
            print(
                f"  dropped ITI range: [{min(itv_dropped):.2f}, {max(itv_dropped):.2f}] s  "
                f"median={float(np.median(itv_dropped)):.2f} s"
            )
    return kept


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
