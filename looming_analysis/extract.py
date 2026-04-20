"""Extract per-stimulus response trajectories from Braid data."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import polars as pl
from scipy.stats import circmean

from .io import load_braidz
from .signal import calculate_angular_velocity

Response = dict


def extract_responses(
    df_kalman: pl.DataFrame,
    df_stim: pl.DataFrame,
    pre_frames: int = -50,
    post_frames: int = 100,
    max_gap_frames: int = 5,
    debug: bool = False,
    heading_ref_frames: int = 10,
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
        heading_ref_frames: Number of frames averaged to compute the heading
            reference before stimulus onset and after expansion end.
            Default 10 frames = 100 ms at 100 Hz.

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
    dt = 0.01  # 100 Hz

    kalman_grouped = df_kalman.partition_by("obj_id", as_dict=True)

    for row in df_stim.iter_rows(named=True):
        obj_id = row["obj_id"]
        stim_frame = row["frame"]

        obj_key = (obj_id,)
        if obj_key not in kalman_grouped:
            n_no_track += 1
            if debug:
                print(f"  [skip] obj_id={obj_id} frame={stim_frame}: no Kalman track")
            continue

        df_obj = kalman_grouped[obj_key]

        target_frames = pl.DataFrame(
            {
                "frame": np.arange(
                    stim_frame + pre_frames, stim_frame + post_frames, dtype=np.int64
                )
            }
        )
        df_res = target_frames.join(df_obj, on="frame", how="left")

        null_count = df_res["xvel"].null_count()
        if null_count > max_gap_frames:
            n_too_many_gaps += 1
            if debug:
                print(
                    f"  [skip] obj_id={obj_id} frame={stim_frame}: "
                    f"{null_count} missing frames (limit={max_gap_frames})"
                )
            continue

        df_res = df_res.fill_null(strategy="forward").fill_null(strategy="backward")

        xvel = df_res["xvel"].to_numpy()
        yvel = df_res["yvel"].to_numpy()

        headings = np.arctan2(yvel, xvel)

        stim_idx = abs(pre_frames)
        pre_window = (
            headings[max(0, stim_idx - heading_ref_frames) : stim_idx]
            if stim_idx > 0
            else headings[:1]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heading_before = circmean(pre_window, low=-np.pi, high=np.pi)

        expansion_duration_ms = row.get("expansion_duration_ms", 500)
        expansion_frames = int(expansion_duration_ms / 10)
        end_expansion_idx = stim_idx + expansion_frames

        if end_expansion_idx < len(headings):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heading_after = circmean(
                    headings[
                        end_expansion_idx : end_expansion_idx + heading_ref_frames
                    ],
                    low=-np.pi,
                    high=np.pi,
                )
        else:
            heading_after = headings[-1]

        heading_change = np.rad2deg(
            np.arctan2(
                np.sin(heading_after - heading_before),
                np.cos(heading_after - heading_before),
            )
        )

        ang_vel = calculate_angular_velocity(xvel, yvel, dt, params=[2, 0.2])

        response_data: Response = {
            "ang_vel": ang_vel,
            "heading_change": heading_change,
            "end_expansion_time": expansion_frames * dt,
            "time": (df_res["frame"].to_numpy() - stim_frame) * dt,
            **{
                key: val
                for key, val in row.items()
                if key not in ["frame", "timestamp", "obj_id"]
            },
        }
        responses.append(response_data)

    n_kept = len(responses)
    n_skipped = n_total - n_kept
    print(
        f"  extract_responses: {n_total} stimuli → {n_kept} kept, "
        f"{n_skipped} skipped "
        f"(no track: {n_no_track}, too many gaps: {n_too_many_gaps})"
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

    Returns:
        Flat list of response dicts from all files.
    """
    all_responses: list[Response] = []
    for path in file_paths:
        if verbose:
            print(f"Processing {path}...")
        df_kalman, df_stim = load_braidz(path)
        if df_stim is None:
            if verbose:
                print(f"  No stim data found in {path}.")
            continue

        responses = extract_responses(
            df_kalman,
            df_stim,
            pre_frames=pre_frames,
            post_frames=post_frames,
            debug=debug,
            heading_ref_frames=heading_ref_frames,
            max_gap_frames=max_gap_frames,
        )
        if group_name is not None:
            for r in responses:
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
        )
        if verbose:
            print(f"  Total for group '{group_name}': {len(responses)} responses.")
        all_responses.extend(responses)
    if verbose:
        print(f"\nTotal responses across all groups: {len(all_responses)}")
    return all_responses
