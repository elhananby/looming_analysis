"""Beginner-facing analysis pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._types import Response
from .config import AnalysisConfig, ResponsivenessConfig, RunConfig
from .dataframe import responses_to_dataframe
from .extract import process_file_groups
from .responsiveness import classify_responsiveness, compute_turn_direction

if TYPE_CHECKING:
    pass


@dataclass
class AnalysisResult:
    """Container returned by `run_analysis`.

    Use the top-level ``plot_*(result.responses, ...)`` functions to visualise
    results — e.g. ``plot_responses(result.responses, col_by="stimulus_offset_deg")``.
    """

    responses: list[Response]
    config: RunConfig | None = field(default=None, repr=False)

    def to_dataframe(self, kind: str = "scalar", backend: str = "polars"):
        return responses_to_dataframe(self.responses, kind=kind, backend=backend)

    def compute_peak_latency(self, **kwargs) -> "AnalysisResult":
        from .plots.peak_aligned import compute_peak_latency
        compute_peak_latency(self.responses, **kwargs)
        return self

    def filter_by_iti(self, min_iti_s: float, *, verbose: bool = True) -> "AnalysisResult":
        """Return a new AnalysisResult dropping trials with ITI < min_iti_s."""
        filtered = filter_by_iti(self.responses, min_iti_s, verbose=verbose)
        return AnalysisResult(filtered, self.config)


def filter_by_iti(
    responses: list[Response],
    min_iti_s: float,
    *,
    verbose: bool = True,
) -> list[Response]:
    """Return responses where inter_trigger_interval >= min_iti_s.

    Trials with a missing or NaN ITI (the first trigger in each file) are
    always kept — they cannot have violated the threshold.
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
