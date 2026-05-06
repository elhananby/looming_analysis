"""Diagnostic plot for RDP-based turn angle computation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .._types import Response
from ..extract import _compute_rdp_turn_angle


def plot_rdp_debug(
    responses: list[Response],
    *,
    epsilon: float = 0.5,
    half_window_frames: int = 50,
    n_responsive: int = 3,
    n_nonresponsive: int = 3,
    seed: int = 0,
) -> Figure:
    """Diagnostic plot comparing raw vs. RDP-simplified trajectories.

    For each selected trial shows two panels side by side:
      - Left:  2D trajectory (integrated velocity) with the raw path, RDP-simplified
               polyline, turn vertex, and angle annotation.
      - Right: Angular velocity trace with the reference index and window marked.

    Args:
        responses: Response list (must have ``xvel``, ``yvel``, ``ang_vel``, ``time``).
        epsilon: RDP tolerance (same units as cumsum(xvel), i.e. velocity × frames).
        half_window_frames: Half-width of the trajectory window around ref_idx.
        n_responsive: Number of responsive trials to sample.
        n_nonresponsive: Number of non-responsive trials to sample.
        seed: Random seed for trial selection.

    Returns:
        The matplotlib Figure.
    """
    rng = np.random.default_rng(seed)

    responsive = [r for r in responses if r.get("is_responsive")]
    nonresponsive = [r for r in responses if not r.get("is_responsive")]

    # Pre-compute the population mean peak time from responsive flies so
    # non-responsive trials use the same fallback as responsiveness.py.
    resp_saccade_times = [
        r.get("saccade_peak_time_ms") / 1000.0
        for r in responsive
        if not np.isnan(r.get("saccade_peak_time_ms", float("nan")))
    ]
    mean_peak_time = float(np.mean(resp_saccade_times)) if resp_saccade_times else None

    selected_r = list(rng.choice(responsive, min(n_responsive, len(responsive)), replace=False))
    selected_nr = list(rng.choice(nonresponsive, min(n_nonresponsive, len(nonresponsive)), replace=False))
    trials = [(r, True) for r in selected_r] + [(r, False) for r in selected_nr]

    n_rows = len(trials)
    if n_rows == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No trials available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (r, is_resp) in enumerate(trials):
        ax_traj = axes[row, 0]
        ax_vel = axes[row, 1]

        xvel = r["xvel"]
        yvel = r["yvel"]
        time = r["time"]
        ang_vel_deg = np.rad2deg(r["ang_vel"])
        end_t = r["end_expansion_time"]

        # Determine ref_idx matching responsiveness.py second-pass logic:
        #   1. Own detected saccade (even if classified non-responsive)
        #   2. Population mean peak time of responsive flies
        #   3. End of expansion as last resort
        saccade_ms = r.get("saccade_peak_time_ms", float("nan"))
        if not np.isnan(saccade_ms):
            ref_idx = int(np.argmin(np.abs(time - saccade_ms / 1000.0)))
            ref_label = "saccade peak"
        elif mean_peak_time is not None:
            ref_idx = int(np.argmin(np.abs(time - mean_peak_time)))
            ref_label = "mean peak (pop.)"
        else:
            ref_idx = int(np.argmin(np.abs(time - end_t)))
            ref_label = "stim end"

        rdp_result = _compute_rdp_turn_angle(
            xvel, yvel, ref_idx, epsilon=epsilon, half_window=half_window_frames
        )
        angle = rdp_result["angle"]
        raw_pts = rdp_result["raw_points"]
        simplified = rdp_result["simplified"]
        turn_v = rdp_result["turn_vertex_local"]
        local_ref = rdp_result["local_ref"]

        # ── Left panel: 2D trajectory ─────────────────────────────────────
        # Full raw trajectory (whole trial) in light gray
        x_full = np.concatenate([[0.0], np.cumsum(xvel)])
        y_full = np.concatenate([[0.0], np.cumsum(yvel)])
        ax_traj.plot(x_full, y_full, color="lightgray", lw=0.8, zorder=1)

        # Windowed raw path
        color = "#2196F3" if is_resp else "#FF9800"
        ax_traj.plot(raw_pts[:, 0], raw_pts[:, 1], color=color, lw=1.2,
                     alpha=0.7, zorder=2, label="raw (window)")

        # RDP simplified polyline
        ax_traj.plot(simplified[:, 0], simplified[:, 1], "k-o",
                     lw=2, ms=4, zorder=3, label="RDP simplified")

        # Mark all simplified vertices
        ax_traj.scatter(simplified[:, 0], simplified[:, 1],
                        color="k", s=20, zorder=4)

        # Turn vertex
        if not (turn_v == 0 or turn_v == len(simplified) - 1):
            tv = simplified[turn_v]
            ax_traj.scatter([tv[0]], [tv[1]], color="red", s=100, zorder=5,
                            label=f"turn vertex")

            # Draw incoming/outgoing arrows
            v_in = simplified[turn_v] - simplified[turn_v - 1]
            v_out = simplified[turn_v + 1] - simplified[turn_v]
            for vec, col in [(v_in, "#1565C0"), (v_out, "#B71C1C")]:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    uv = vec / norm * min(norm, np.ptp(raw_pts[:, 0]) * 0.2 + 1e-9)
                    ax_traj.annotate(
                        "", xy=(tv[0] + uv[0], tv[1] + uv[1]),
                        xytext=(tv[0], tv[1]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                        zorder=6,
                    )

        # ref_idx position in raw_points
        ref_pos = raw_pts[local_ref]
        ax_traj.scatter([ref_pos[0]], [ref_pos[1]], marker="x", s=80,
                        color="purple", zorder=5, label="ref_idx")

        angle_str = f"{angle:.1f}°" if not np.isnan(angle) else "nan"
        resp_str = "responsive" if is_resp else "non-responsive"
        ax_traj.set_title(f"{resp_str}  |  RDP angle = {angle_str}  |  ref: {ref_label}", fontsize=9)
        ax_traj.set_xlabel("∑xvel (arb.)")
        ax_traj.set_ylabel("∑yvel (arb.)")
        ax_traj.set_aspect("equal", adjustable="datalim")
        ax_traj.legend(fontsize=7, loc="best")
        ax_traj.grid(True, alpha=0.3)

        # Colour the row spine to signal responsiveness
        spine_color = "#388E3C" if is_resp else "#757575"
        for spine in ax_traj.spines.values():
            spine.set_edgecolor(spine_color)
            spine.set_linewidth(2)

        # ── Right panel: angular velocity trace ───────────────────────────
        ax_vel.plot(time, ang_vel_deg, color=color, lw=1.2)
        ax_vel.axvline(time[ref_idx], color="red", lw=1.5, linestyle="--",
                       label=f"{ref_label} (t={time[ref_idx]:.3f}s)")
        ax_vel.axvline(0, color="k", lw=0.8, linestyle=":", label="stim onset")
        ax_vel.axvline(end_t, color="gray", lw=0.8, linestyle=":", label="stim end")

        # Shade the half_window around ref_idx
        t_start = time[max(0, ref_idx - half_window_frames)]
        t_end = time[min(len(time) - 1, ref_idx + half_window_frames)]
        ax_vel.axvspan(t_start, t_end, alpha=0.12, color=color, label="window")

        ax_vel.set_xlabel("time (s)")
        ax_vel.set_ylabel("ang vel (deg/s)")
        ax_vel.set_title(f"ε={epsilon}", fontsize=9)
        ax_vel.legend(fontsize=7, loc="best")
        ax_vel.grid(True, alpha=0.3)

        for spine in ax_vel.spines.values():
            spine.set_edgecolor(spine_color)
            spine.set_linewidth(2)

    fig.suptitle(
        f"RDP turn angle debug  (ε={epsilon}, half_window={half_window_frames} frames)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
