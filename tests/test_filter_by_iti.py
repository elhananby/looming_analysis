from __future__ import annotations

import math

import numpy as np

from looming_analysis.pipeline import filter_by_iti


def _r(iti):
    return {"inter_trigger_interval": iti, "value": 1}


def test_keeps_trials_above_threshold():
    responses = [_r(2.0), _r(3.0), _r(5.0)]
    kept = filter_by_iti(responses, min_iti_s=2.0, verbose=False)
    assert len(kept) == 3


def test_drops_trials_below_threshold():
    responses = [_r(0.5), _r(1.0), _r(3.0)]
    kept = filter_by_iti(responses, min_iti_s=2.0, verbose=False)
    assert len(kept) == 1
    assert kept[0]["inter_trigger_interval"] == 3.0


def test_keeps_trials_with_nan_iti():
    responses = [_r(float("nan")), _r(0.5), _r(3.0)]
    kept = filter_by_iti(responses, min_iti_s=2.0, verbose=False)
    assert len(kept) == 2
    assert kept[0]["inter_trigger_interval"] != kept[0]["inter_trigger_interval"]  # NaN


def test_keeps_trials_with_none_iti():
    responses = [_r(None), _r(0.5), _r(3.0)]
    kept = filter_by_iti(responses, min_iti_s=2.0, verbose=False)
    assert len(kept) == 2
    assert kept[0]["inter_trigger_interval"] is None


def test_does_not_mutate_original_list():
    responses = [_r(0.5), _r(3.0)]
    kept = filter_by_iti(responses, min_iti_s=2.0, verbose=False)
    assert len(responses) == 2
    assert len(kept) == 1
