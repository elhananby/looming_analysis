"""Extract per-stimulus response trajectories from Braid data."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import polars as pl
from scipy.stats import circmean

from ._types import DT_SECONDS, Response
from .io import load_braidz, load_trigger_config
from .signal import calculate_angular_velocity


def _slice_trial(
    df_obj: pl.DataFrame,
    stim_frame: int,
    pre_frames: int,
    post_frames: int,
    max_gap_frames: int,
) -> pl.DataFrame | None:
    """Slice a trial window. Returns filled DataFrame, or None if too gappy."""
    target_frames = pl.DataFrame(
        {
            "frame": np.arange(
                stim_frame + pre_frames, stim_frame + post_frames, dtype=np.int64
            )
        }
    )
    df_res = target_frames.join(df_obj, on="frame", how="left")
    if df_res["xvel"].null_count() > max_gap_frames:
        return None
    return df_res.fill_null(strategy="forward").fill_null(strategy="backward")


def _compute_heading_change_vector(
    xvel: np.ndarray,
    yvel: np.ndarray,
    ref_idx: int,
    window: int = 10,
) -> float:
    """Heading change (degrees) between the mean velocity vectors before and after *ref_idx*.

    The "before" vector is the mean of xvel/yvel over [ref_idx-window, ref_idx) and the
    "after" vector over [ref_idx, ref_idx+window). Returns nan if either window is empty.
    """
    n = len(xvel)
    pre_slice = slice(max(0, ref_idx - window), ref_idx)
    post_slice = slice(ref_idx, min(n, ref_idx + window))
    pre_x, pre_y = xvel[pre_slice], yvel[pre_slice]
    post_x, post_y = xvel[post_slice], yvel[post_slice]
    if len(pre_x) == 0 or len(post_x) == 0:
        return float("nan")
    before_angle = np.arctan2(np.mean(pre_y), np.mean(pre_x))
    after_angle = np.arctan2(np.mean(post_y), np.mean(post_x))
    return float(
        np.rad2deg(
            np.arctan2(
                np.sin(after_angle - before_angle),
                np.cos(after_angle - before_angle),
            )
        )
    )


def _compute_rdp_turn_angle(
    xvel: np.ndarray,
    yvel: np.ndarray,
    ref_idx: int,
    epsilon: float,
    half_window: int = 50,
) -> dict:
    """RDP-based turn angle at *ref_idx*.

    Integrates xvel/yvel to get a relative 2D trajectory, simplifies it with
    the Ramer-Douglas-Peucker algorithm, then measures the angle at the
    simplified vertex nearest to *ref_idx*.

    Returns a dict with keys:
        angle               – turn angle in degrees (nan if vertex is at endpoint)
        raw_points          – (N+1, 2) array of integrated positions in the window
        simplified          – (M, 2) simplified polyline vertices
        simplified_indices  – local frame indices of each simplified vertex
        turn_vertex_local   – index into *simplified* of the chosen vertex
        local_ref           – index of ref_idx within raw_points
        start_frame         – absolute start frame of the window
    """
    from rdp import rdp as _rdp

    n = len(xvel)
    start = max(0, ref_idx - half_window)
    end = min(n, ref_idx + half_window + 1)

    x_pos = np.concatenate([[0.0], np.cumsum(xvel[start:end])])
    y_pos = np.concatenate([[0.0], np.cumsum(yvel[start:end])])
    raw_points = np.column_stack([x_pos, y_pos])

    local_ref = ref_idx - start

    mask = _rdp(raw_points, epsilon=epsilon, return_mask=True)
    simplified = raw_points[mask]
    simplified_indices = np.where(mask)[0]

    dists = np.abs(simplified_indices - local_ref)
    nearest = int(np.argmin(dists))

    if nearest == 0 or nearest == len(simplified) - 1:
        angle = float("nan")
    else:
        v_in = simplified[nearest] - simplified[nearest - 1]
        v_out = simplified[nearest + 1] - simplified[nearest]
        a_in = np.arctan2(v_in[1], v_in[0])
        a_out = np.arctan2(v_out[1], v_out[0])
        angle = float(
            np.rad2deg(
                np.arctan2(np.sin(a_out - a_in), np.cos(a_out - a_in))
            )
        )

    return {
        "angle": angle,
        "raw_points": raw_points,
        "simplified": simplified,
        "simplified_indices": simplified_indices,
        "turn_vertex_local": nearest,
        "local_ref": local_ref,
        "start_frame": start,
    }


def _compute_heading_change(
    headings: np.ndarray,
    stim_idx: int,
    end_expansion_idx: int,
    ref_frames: int,
) -> float:
    """Heading change in degrees, wrapped to [-180, 180]."""
    pre_window = (
        headings[max(0, stim_idx - ref_frames) : stim_idx]
        if stim_idx > 0
        else headings[:1]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        heading_before = circmean(pre_window, low=-np.pi, high=np.pi)

    post_window = headings[end_expansion_idx : end_expansion_idx + ref_frames]
    if len(post_window) == 0:
        post_window = headings[-ref_frames:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        heading_after = circmean(post_window, low=-np.pi, high=np.pi)

    return float(
        np.rad2deg(
            np.arctan2(
                np.sin(heading_after - heading_before),
                np.cos(heading_after - heading_before),
            )
        )
    )


def extract_responses(
    df_kalman: pl.DataFrame,
    df_stim: pl.DataFrame,
    pre_frames: int = -50,
    post_frames: int = 100,
    max_gap_frames: int = 5,
    debug: bool = False,
    verbose: bool = True,
    heading_ref_frames: int = 10,
    include_sham: bool = False,
) -> list[Response]:
    """Extract one response trajectory per stimulus row.

    For each stimulus, slices the Kalman track from `pre_frames` to
    `post_frames` relative to stimulus onset. Trajectories with more than
    `max_gap_frames` missing frames are skipped; small gaps are
    forward/backward filled.

    Args:
        df_kalman: Kalman estimates DataFrame from `load_braidz`.
        df_stim: Stimulus DataFrame from `load_braidz`.
        pre_frames: Frames before stimulus onset (must be <= 0).
        post_frames: Frames after stimulus onset (must be positive).
        max_gap_frames: Maximum number of missing frames tolerated per trial.
            Trials exceeding this are skipped; smaller gaps are filled.
        debug: If True, print a line for every skipped stimulus row.
        verbose: If True, print a summary line after extraction.
        heading_ref_frames: Number of frames averaged to compute the heading
            reference before stimulus onset and after expansion end.
            Default 10 frames = 100 ms at 100 Hz.
        include_sham: If True, extract sham trials (where `sham` column is truthy)
            alongside real looming trials. If False, skip sham trials.

    Returns:
        List of response dicts, each containing:
            - `ang_vel`: angular velocity trace (rad/s)
            - `heading_change`: net heading change (deg, wrapped to [-180, 180])
            - `end_expansion_time`: seconds from stim onset to end of expansion
            - `time`: time axis relative to stim onset (s)
            - plus all scalar columns from the stim row except frame/timestamp/obj_id
    """
    if pre_frames > 0 or post_frames <= 0:
        raise ValueError("pre_frames must be <= 0 and post_frames must be positive.")

    responses: list[Response] = []
    n_total = len(df_stim)
    n_no_track = 0
    n_too_many_gaps = 0
    n_sham_total = 0
    n_sham_skipped = 0
    dt = DT_SECONDS

    kalman_grouped = df_kalman.partition_by("obj_id", as_dict=True)

    for row in df_stim.iter_rows(named=True):
        obj_id = row["obj_id"]
        stim_frame = row["frame"]
        is_sham = bool(row.get("sham", False))

        if is_sham:
            n_sham_total += 1

        if is_sham and not include_sham:
            n_sham_skipped += 1
            if debug:
                print(f"  [skip] obj_id={obj_id} frame={stim_frame}: sham trial")
            continue

        obj_key = (obj_id,)
        if obj_key not in kalman_grouped:
            n_no_track += 1
            if debug:
                print(f"  [skip] obj_id={obj_id} frame={stim_frame}: no Kalman track")
            continue

        df_obj = kalman_grouped[obj_key]

        df_res = _slice_trial(
            df_obj, stim_frame, pre_frames, post_frames, max_gap_frames
        )
        if df_res is None:
            n_too_many_gaps += 1
            if debug:
                print(
                    f"  [skip] obj_id={obj_id} frame={stim_frame}: too many missing frames"
                )
            continue

        xvel = df_res["xvel"].to_numpy()
        yvel = df_res["yvel"].to_numpy()
        headings = np.arctan2(yvel, xvel)

        stim_idx = abs(pre_frames)
        expansion_duration_ms = row.get("expansion_duration_ms", 500)
        expansion_frames = int(expansion_duration_ms / 10)

        heading_change = _compute_heading_change(
            headings, stim_idx, stim_idx + expansion_frames, heading_ref_frames
        )
        stim_mid_idx = stim_idx + expansion_frames // 2
        heading_change_stim_vector = _compute_heading_change_vector(
            xvel, yvel, stim_mid_idx, window=heading_ref_frames
        )
        ang_vel = calculate_angular_velocity(xvel, yvel, dt, params=[2, 0.2])
        heading_deg = np.rad2deg(np.arctan2(np.sin(headings), np.cos(headings)))

        response_data: Response = {
            "ang_vel": ang_vel,
            "xvel": xvel,
            "yvel": yvel,
            "heading": headings,
            "heading_deg": heading_deg,
            "heading_change": heading_change,
            "heading_change_stim_vector": heading_change_stim_vector,
            "end_expansion_time": expansion_frames * dt,
            "time": (df_res["frame"].to_numpy() - stim_frame) * dt,
            **{
                key: val
                for key, val in row.items()
                if key not in ["frame", "timestamp", "obj_id", "xvel", "yvel"]
            },
        }
        responses.append(response_data)

    n_kept = len(responses)
    n_skipped = n_total - n_kept
    if verbose:
        sham_note = (
            f"sham found: {n_sham_total}, skipped: {n_sham_skipped}"
            if n_sham_total > 0
            else "sham: 0"
        )
        print(
            f"  extract_responses: {n_total} stimuli → {n_kept} kept, "
            f"{n_skipped} skipped "
            f"(no track: {n_no_track}, too many gaps: {n_too_many_gaps}, {sham_note})"
        )
    return responses


def process_all_files(
    file_paths: list[str],
    pre_frames: int = -50,
    post_frames: int = 100,
    group_name: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
    heading_ref_frames: int = 10,
    max_gap_frames: int = 5,
    include_sham: bool = False,
    cache_dir: Optional[str] = None,
) -> list[Response]:
    """Process multiple `.braidz` files and combine the responses.

    Args:
        file_paths: Paths to the `.braidz` files.
        pre_frames: Frames before stimulus onset.
        post_frames: Frames after stimulus onset.
        group_name: If set, each response gets `'group': group_name`.
        verbose: If True, print per-file progress.
        debug: If True, print a line for each skipped stimulus row.
        heading_ref_frames: Frames averaged for pre/post heading reference.
        max_gap_frames: Maximum missing frames tolerated per trial.
        include_sham: If True, extract sham trials alongside real looming trials.
        cache_dir: Optional parquet cache directory. If set, data is cached
            on first load and loaded from cache on subsequent runs.

    Returns:
        Flat list of response dicts from all files.
    """
    all_responses: list[Response] = []
    for path in file_paths:
        if verbose:
            print(f"Processing {path}...")
        df_kalman, df_stim = load_braidz(path, cache_dir=cache_dir)
        if df_stim is None:
            if verbose:
                print(f"  No stim data found in {path}.")
            continue

        trigger_params = load_trigger_config(path)
        if "timestamp" in df_stim.columns:
            df_stim = df_stim.sort("timestamp").with_columns(
                pl.col("timestamp")
                .diff()
                .fill_null(float("nan"))
                .alias("inter_trigger_interval")
            )
        responses = extract_responses(
            df_kalman,
            df_stim,
            pre_frames=pre_frames,
            post_frames=post_frames,
            debug=debug,
            verbose=verbose,
            heading_ref_frames=heading_ref_frames,
            max_gap_frames=max_gap_frames,
            include_sham=include_sham,
        )
        for r in responses:
            r.update(trigger_params)
            if group_name is not None:
                r["group"] = group_name
        if verbose:
            print(f"  Extracted {len(responses)} valid responses.")
        all_responses.extend(responses)
    return all_responses


def process_file_groups(
    file_groups: dict[str, list[str]],
    pre_frames: int = -50,
    post_frames: int = 100,
    verbose: bool = True,
    debug: bool = False,
    heading_ref_frames: int = 10,
    max_gap_frames: int = 5,
    include_sham: bool = False,
    cache_dir: Optional[str] = None,
) -> list[Response]:
    """Process multiple groups of `.braidz` files.

    Args:
        file_groups: Mapping `{group_name: [file_paths, ...]}`.
        pre_frames: Frames before stimulus onset.
        post_frames: Frames after stimulus onset.
        verbose: If True, print per-group and per-file progress.
        debug: If True, print a line for each skipped stimulus row.
        heading_ref_frames: Frames averaged for pre/post heading reference.
        max_gap_frames: Maximum missing frames tolerated per trial.
        include_sham: If True, extract sham trials alongside real looming trials.
        cache_dir: Optional parquet cache directory. If set, data is cached
            on first load and loaded from cache on subsequent runs.

    Returns:
        Combined list of responses, each tagged with a `'group'` key.
    """
    all_responses: list[Response] = []
    for group_name, file_paths in file_groups.items():
        if verbose:
            print(f"\n=== Group: {group_name} ===")
        responses = process_all_files(
            file_paths,
            pre_frames,
            post_frames,
            group_name=group_name,
            verbose=verbose,
            debug=debug,
            heading_ref_frames=heading_ref_frames,
            max_gap_frames=max_gap_frames,
            include_sham=include_sham,
            cache_dir=cache_dir,
        )
        if verbose:
            print(f"  Total for group '{group_name}': {len(responses)} responses.")
        all_responses.extend(responses)
    if verbose:
        print(f"\nTotal responses across all groups: {len(all_responses)}")
    return all_responses
