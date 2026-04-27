"""Configuration objects for beginner-facing analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ._types import DT_SECONDS


def _ms_to_frames(ms: float) -> int:
    return int(round((ms / 1000.0) / DT_SECONDS))


@dataclass(frozen=True)
class AnalysisConfig:
    """Extraction settings expressed in milliseconds for user-facing workflows."""

    pre_ms: float = 500.0
    post_ms: float = 1000.0
    max_gap_ms: float = 50.0
    heading_ref_ms: float = 100.0
    include_sham: bool = False
    cache_dir: str | None = ".braidz_cache"

    def __post_init__(self) -> None:
        if self.pre_ms < 0:
            raise ValueError("pre_ms must be >= 0.")
        if self.post_ms <= 0:
            raise ValueError("post_ms must be > 0.")
        if self.max_gap_ms < 0:
            raise ValueError("max_gap_ms must be >= 0.")
        if self.heading_ref_ms <= 0:
            raise ValueError("heading_ref_ms must be > 0.")

    @property
    def pre_frames(self) -> int:
        return -_ms_to_frames(self.pre_ms)

    @property
    def post_frames(self) -> int:
        return _ms_to_frames(self.post_ms)

    @property
    def max_gap_frames(self) -> int:
        return _ms_to_frames(self.max_gap_ms)

    @property
    def heading_ref_frames(self) -> int:
        return _ms_to_frames(self.heading_ref_ms)


@dataclass(frozen=True)
class ResponsivenessConfig:
    """Responsiveness settings for `classify_responsiveness`."""

    threshold_deg_s: float = 300.0
    window_ms: float | tuple[float, float] | list[float] = 200.0
    zscore_k: float = 3.0
    baseline_window_ms: tuple[float, float] = (-400.0, -100.0)
    heading_threshold_deg: float = 30.0
    impulse_threshold_deg: float = 20.0
    method: str = "combined"

    def as_kwargs(self) -> dict:
        return {
            "threshold_deg_s": self.threshold_deg_s,
            "window_ms": self.window_ms,
            "zscore_k": self.zscore_k,
            "baseline_window_ms": self.baseline_window_ms,
            "heading_threshold_deg": self.heading_threshold_deg,
            "impulse_threshold_deg": self.impulse_threshold_deg,
            "method": self.method,
        }
