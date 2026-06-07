"""
algo_compare.py — Overlay comparison of MUSIC, Capon, and Bartlett reprocess results.

Layout (3 rows × 2 cols):
  [0,0] Az vs time  — all algorithms overlaid
  [0,1] Skyplot     — all algorithms overlaid (polar)
  [1,0] PAPR dist   — histogram overlay
  [1,1] El dist     — histogram overlay
  [2, :] Summary table (coloured monospace rows)
"""

from __future__ import annotations

import json
import os

import numpy as np

from .offline_viz import BG, BG2, C_BDR, C_MUT, C_TEXT, load_session_meta
from .track_clusterer import load_tracks_json

# Colorblind-safe palette (Wong) — blue / orange / green, well separated on dark BG
_ALGO_STYLE: dict[str, dict] = {
    "music": {"color": "#56B4E9", "marker": "o", "label": "MUSIC"},
    "capon": {"color": "#E69F00", "marker": "s", "label": "CAPON"},
    "bartlett": {"color": "#009E73", "marker": "^", "label": "BARTLETT"},
}


def _algo_style(algo: str) -> dict:
    return _ALGO_STYLE.get(algo.lower(), {
        "color": C_TEXT, "marker": "D", "label": algo.upper(),
    })


def _load_algo_summary(session_dir: str, subdir: str) -> dict | None:
    jsonl = os.path.join(session_dir, subdir, "doa_multi.jsonl")
    if not os.path.isfile(jsonl):
        return None

    n_bursts = 0
    paprs: list[float] = []
    azs: list[float] = []
    els: list[float] = []
    peak_times: list[float] = []
    t_global_min: float | None = None

    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_bursts += 1
            t_burst = float(row.get("t", 0))
            if t_global_min is None:
                t_global_min = t_burst
            for peak in row.get("peaks", []):
                peak_times.append(t_burst)
                azs.append(float(peak[0]))
                els.append(float(peak[1]))
                paprs.append(float(peak[3]))

    tracks_path = os.path.join(session_dir, subdir, "tracks.json")
    tracks: list[dict] = []
    if os.path.isfile(tracks_path):
        tracks, _ = load_tracks_json(tracks_path)

    algo = subdir.replace("doa_multi_", "") if subdir.startswith("doa_multi_") else "music"
    if subdir == "doa_multi":
        if os.path.isfile(tracks_path):
            with open(tracks_path, encoding="utf-8") as f:
                tr_meta = json.load(f).get("meta", {})
                algo = tr_meta.get("algo", algo)

    pt = np.asarray(peak_times, dtype=float)
    t0 = float(t_global_min) if t_global_min is not None else 0.0

    return {
        "subdir": subdir,
        "algo": algo,
        "n_bursts": n_bursts,
        "n_tracks": len(tracks),
        "n_long_tracks": sum(1 for tr in tracks if tr.get("n_peaks", 0) >= 30),
        "mean_papr_db": float(np.mean(paprs)) if paprs else 0.0,
        "peak_times": pt,
        "t_rel": pt - t0,
        "azs": np.asarray(azs, dtype=float),
        "els": np.asarray(els, dtype=float),
        "paprs": np.asarray(paprs, dtype=float),
        "tracks": tracks,
    }


def show_algo_comparison(
    session_dir: str,
    *,
    algos: tuple[str, ...] = ("music", "capon", "bartlett"),
    save_dir: str = "",
    show: bool = True,
    no_tracks: bool = False,
) -> None:
    """Overlay comparison of available algorithm outputs in a 3×2 panel."""
    session_dir = session_dir.rstrip("/")
    summaries: list[dict] = []

    for algo in algos:
        subdir = f"doa_multi_{algo}"
        if algo == "music" and not os.path.isdir(os.path.join(session_dir, subdir)):
            subdir = "doa_multi"
        s = _load_algo_summary(session_dir, subdir)
        if s is not None:
            summaries.append(s)

    if not summaries:
        raise FileNotFoundError(
            f"No doa_multi_* directories found in {session_dir}. "
            "Run batch_reprocess.py first."
        )

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    meta = load_session_meta(session_dir)
    title = os.path.basename(session_dir)
    if meta:
        title += f"  {meta.get('freq_hz', 0)/1e6:.3f} MHz"

    fig = plt.figure(figsize=(14, 10.5), facecolor=BG)
    fig.suptitle(f"DOA algorithm comparison — {title}", color=C_TEXT, fontsize=12, y=0.99)

    gs = fig.add_gridspec(
        3, 2,
        left=0.07, right=0.97, top=0.90, bottom=0.07,
        hspace=0.42, wspace=0.28,
        height_ratios=[1.0, 1.0, 0.45],
    )

    ax_az = fig.add_subplot(gs[0, 0])
    ax_sky = fig.add_subplot(gs[0, 1], projection="polar")
    ax_papr = fig.add_subplot(gs[1, 0])
    ax_el = fig.add_subplot(gs[1, 1])
    ax_tbl = fig.add_subplot(gs[2, :])

    for ax in (ax_az, ax_papr, ax_el):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=C_MUT, labelsize=8)
        ax.grid(True, color=C_BDR, alpha=0.4)
    ax_sky.set_facecolor(BG2)
    ax_sky.tick_params(colors=C_MUT, labelsize=7)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    ax_sky.grid(color=C_BDR, alpha=0.4)

    legend_handles = []
    t_global_min = min(
        (float(s["peak_times"][0]) for s in summaries if len(s["peak_times"])),
        default=0.0,
    )

    for s in summaries:
        st = _algo_style(s["algo"])
        color = st["color"]
        marker = st["marker"]
        label = st["label"]

        legend_handles.append(mlines.Line2D(
            [], [], color=color, marker=marker, linestyle="None",
            markersize=8, markeredgewidth=0.6, markeredgecolor="white",
            label=label,
        ))

        if len(s["peak_times"]):
            t_rel = s["peak_times"] - t_global_min
            ax_az.scatter(
                t_rel, s["azs"], s=14, c=color, marker=marker,
                alpha=0.65, edgecolors="white", linewidths=0.35, zorder=3,
            )

        if len(s["azs"]):
            th = np.radians(s["azs"])
            r = s["els"]
            ax_sky.scatter(
                th, r, s=12, c=color, marker=marker,
                alpha=0.55, edgecolors="white", linewidths=0.3, zorder=3,
            )

        if len(s["paprs"]):
            ax_papr.hist(
                s["paprs"], bins=30, histtype="step", linewidth=2.2,
                color=color, alpha=0.95, label=label,
            )

        if len(s["els"]):
            ax_el.hist(
                s["els"], bins=18, range=(5, 90), histtype="step", linewidth=2.2,
                color=color, alpha=0.95, label=label,
            )

    ax_az.set_title("Az vs time", color=C_TEXT, fontsize=9, loc="left")
    ax_az.set_ylabel("Az [°]", color=C_MUT, fontsize=8)
    ax_az.set_xlabel("Time [s]", color=C_MUT, fontsize=8)
    ax_az.set_ylim(0, 360)

    ax_sky.set_title("Skyplot", color=C_TEXT, fontsize=9, pad=10)

    ax_papr.set_title("PAPR distribution (step curves)", color=C_TEXT, fontsize=9, loc="left")
    ax_papr.set_xlabel("PAPR [dB]", color=C_MUT, fontsize=8)
    ax_papr.set_ylabel("Count", color=C_MUT, fontsize=8)

    ax_el.set_title("Elevation distribution (step curves)", color=C_TEXT, fontsize=9, loc="left")
    ax_el.set_xlabel("El [°]", color=C_MUT, fontsize=8)
    ax_el.set_ylabel("Count", color=C_MUT, fontsize=8)
    ax_el.set_xlim(5, 90)

    fig.legend(
        handles=legend_handles, loc="upper center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.935), fontsize=9, framealpha=0.85,
        labelcolor="white", facecolor=BG2, edgecolor=C_BDR,
        handletextpad=0.6, columnspacing=1.8,
    )

    ax_tbl.set_facecolor(BG2)
    ax_tbl.axis("off")
    header = f"{'Algo':<10}{'Bursts':>8}{'Peaks':>7}{'Tracks':>8}{'Long≥30':>9}{'MeanPAPR':>10}"
    ax_tbl.text(
        0.01, 0.92, header, transform=ax_tbl.transAxes,
        color=C_MUT, fontsize=8.5, family="monospace", va="top",
    )
    for row_i, s in enumerate(summaries):
        st = _algo_style(s["algo"])
        row_txt = (
            f"{st['label']:<10}"
            f"{s['n_bursts']:>8}"
            f"{len(s['azs']):>7}"
            f"{s['n_tracks']:>8}"
            f"{s['n_long_tracks']:>9}"
            f"{s['mean_papr_db']:>9.1f}dB"
        )
        ax_tbl.text(
            0.01, 0.65 - row_i * 0.28, row_txt,
            transform=ax_tbl.transAxes,
            color=st["color"], fontsize=8.5, family="monospace", va="top",
            fontweight="bold",
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "algo_compare.png")
        fig.savefig(out, dpi=150, facecolor=BG)
        print(f"Saved {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)
