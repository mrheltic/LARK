"""
offline_viz.py — Visualize offline / recorded DOA results without loading raw IQ.

Loads incrementally from:
  - JSONL produced by run_doa_offline.py
  - session_.../doa/est_*.npz (live recording estimates)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

# Dark theme (matches run_doa.py)
BG = "#1a1d27"
BG2 = "#21253a"
C_BDR = "#3b4263"
C_MUT = "#8891b0"
C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"
C_TEAL = "#4ecdc4"
C_AMBER = "#f4a431"
C_ROSE = "#f16b6f"


def load_jsonl(path: str, *, stride: int = 1, max_rows: int = 0) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if stride > 1 and i % stride != 0:
                continue
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def load_doa_dir(
    session_dir: str,
    *,
    stride: int = 1,
    max_rows: int = 0,
    load_spec: bool = False,
) -> list[dict]:
    """Load est_*.npz one file at a time (memory-safe)."""
    doa_dir = os.path.join(session_dir, "doa")
    if not os.path.isdir(doa_dir):
        raise FileNotFoundError(f"No doa/ in {session_dir}")

    rows: list[dict] = []
    i = 0
    while True:
        path = os.path.join(doa_dir, f"est_{i:06d}.npz")
        if not os.path.isfile(path):
            break
        if stride <= 1 or i % stride == 0:
            d = np.load(path)
            row: dict[str, Any] = {
                "n": i + 1,
                "az": float(d["az_deg"]),
                "el": float(d["el_deg"]),
                "snr_db": float(d["snr_db"]),
                "papr_db": float(d["papr_db"]),
            }
            if "t" in d.files:
                row["t"] = float(d["t"])
            rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break
        i += 1

    if load_spec and rows:
        last_idx = (len(rows) - 1) * stride
        path = os.path.join(doa_dir, f"est_{last_idx:06d}.npz")
        if os.path.isfile(path):
            d = np.load(path)
            if "spec2d" in d.files:
                rows[-1]["spec2d"] = np.asarray(d["spec2d"])
    return rows


def load_session_meta(session_dir: str) -> dict:
    path = os.path.join(session_dir, "meta.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _pick_spectrum(rows: list[dict]) -> np.ndarray | None:
    for row in reversed(rows):
        spec = row.get("spec2d")
        if spec is not None:
            return np.asarray(spec)
    return None


def show_results(
    rows: list[dict],
    *,
    title: str = "Offline DOA",
    meta: dict | None = None,
    save_dir: str = "",
    show: bool = True,
) -> str | None:
    """Plot skyplot + time series. Optionally save PNG to save_dir."""
    if not rows:
        raise ValueError("No DOA results to plot")

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG2,
        "axes.edgecolor": C_BDR,
        "axes.labelcolor": C_MUT,
        "text.color": C_TEXT,
        "xtick.color": C_MUT,
        "ytick.color": C_MUT,
        "grid.color": C_BDR,
        "grid.alpha": 0.45,
    })

    az = np.array([float(r["az"]) for r in rows], dtype=np.float64)
    el = np.array([float(r["el"]) for r in rows], dtype=np.float64)
    snr = np.array([float(r.get("snr_db", 0)) for r in rows], dtype=np.float64)
    papr = np.array([float(r.get("papr_db", 0)) for r in rows], dtype=np.float64)
    if "frame" in rows[0]:
        x = np.array([int(r["frame"]) for r in rows], dtype=np.int64)
        x_label = "CPI frame #"
    else:
        x = np.arange(len(rows), dtype=np.int64)
        x_label = "Estimate #"

    cfo = None
    if "cfo_hz" in rows[0]:
        cfo = np.array([float(r["cfo_hz"]) for r in rows], dtype=np.float64)

    spec = _pick_spectrum(rows)
    has_spec = spec is not None

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    nrows = 3 if has_spec else 2
    gs = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.35, wspace=0.28,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    # Skyplot
    ax_sky = fig.add_subplot(gs[0, 0], polar=True)
    ax_sky.set_facecolor(BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    theta = np.radians(az)
    colors = np.linspace(0, 1, len(az))
    ax_sky.scatter(theta, el, c=colors, cmap="plasma", s=12, alpha=0.65)
    ax_sky.scatter([np.radians(az[-1])], [el[-1]], s=80, color=C_AMBER,
                   edgecolors="white", linewidths=0.8, zorder=5)
    ax_sky.set_title("Skyplot (colour = time)", color=C_TEXT, fontsize=10, pad=12)
    ax_sky.grid(color=C_BDR, alpha=0.5)

    # Az / El vs frame
    ax_az = fig.add_subplot(gs[0, 1])
    ax_az.plot(x, az, color=C_BLUE, lw=0.8, alpha=0.85)
    ax_az.set_ylabel("Azimuth [°]", color=C_BLUE)
    ax_az.set_xlabel(x_label, color=C_MUT)
    ax_az.set_ylim(0, 360)
    ax_az.grid(True)
    ax_az.set_title("Azimuth", color=C_TEXT, loc="left")

    ax_el = fig.add_subplot(gs[1, 1], sharex=ax_az)
    ax_el.plot(x, el, color=C_TEAL, lw=0.8, alpha=0.85)
    ax_el.set_ylabel("Elevation [°]", color=C_TEAL)
    ax_el.set_xlabel(x_label, color=C_MUT)
    ax_el.set_ylim(0, 90)
    ax_el.grid(True)
    ax_el.set_title("Elevation", color=C_TEXT, loc="left")

    # SNR / PAPR
    ax_q = fig.add_subplot(gs[1, 0])
    ax_q.scatter(snr, papr, c=colors, cmap="plasma", s=10, alpha=0.6)
    ax_q.set_xlabel("SNR [dB]", color=C_MUT)
    ax_q.set_ylabel("PAPR [dB]", color=C_MUT)
    ax_q.grid(True)
    ax_q.set_title("Quality (SNR vs PAPR)", color=C_TEXT, loc="left")

    # Stats box
    subtitle = ""
    if meta:
        subtitle = (f"{meta.get('freq_hz', 0)/1e6:.3f} MHz  "
                      f"gain={meta.get('gain_db', '?')} dB  "
                      f"{meta.get('mode', '')}/{meta.get('algo', '')}")
    fig.suptitle(f"{title}  —  n={len(rows)}  {subtitle}", color=C_TEXT, fontsize=11)

    stats = (
        f"Az  med {np.median(az):5.1f}°  std {np.std(az):4.1f}°\n"
        f"El  med {np.median(el):5.1f}°  std {np.std(el):4.1f}°\n"
        f"SNR med {np.median(snr):4.1f} dB\n"
        f"PAPR med {np.median(papr):4.1f} dB"
    )
    if cfo is not None:
        stats += f"\nCFO med {np.median(cfo):+.0f} Hz"
    ax_sky.text(1.15, 0.5, stats, transform=ax_sky.transAxes, fontsize=8,
                color=C_TEXT, va="center", family="monospace")

    if has_spec and spec is not None:
        ax_spec = fig.add_subplot(gs[2, :])
        im = ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="inferno",
                            extent=[0, 360, 5, 90])
        ax_spec.set_xlabel("Azimuth [°]", color=C_MUT)
        ax_spec.set_ylabel("Elevation [°]", color=C_MUT)
        ax_spec.set_title("Last available 2D MUSIC spectrum [dB]", color=C_TEXT, loc="left")
        fig.colorbar(im, ax=ax_spec, fraction=0.02, pad=0.02)

    saved = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        saved = os.path.join(save_dir, "offline_doa_summary.png")
        fig.savefig(saved, dpi=150, facecolor=BG)
        print(f"[plot] Saved → {saved}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved


def load_results_for_plot(
    input_path: str,
    *,
    stride: int = 1,
    max_rows: int = 5000,
    from_doa: bool = False,
) -> tuple[list[dict], dict, str]:
    """
    Resolve input to result rows + meta.
    input_path: session dir, jsonl file, or doa_music.npz
    """
    meta: dict = {}
    title = os.path.basename(input_path.rstrip("/"))

    if input_path.endswith(".jsonl") and os.path.isfile(input_path):
        rows = load_jsonl(input_path, stride=stride, max_rows=max_rows)
        parent = os.path.dirname(input_path)
        if os.path.isfile(os.path.join(parent, "meta.json")):
            meta = load_session_meta(parent)
        return rows, meta, title

    if os.path.isdir(input_path):
        meta = load_session_meta(input_path)
        jsonl_default = os.path.join(input_path, "offline_doa.jsonl")
        if not from_doa and os.path.isfile(jsonl_default):
            rows = load_jsonl(jsonl_default, stride=stride, max_rows=max_rows)
            return rows, meta, title
        rows = load_doa_dir(
            input_path, stride=stride, max_rows=max_rows, load_spec=True,
        )
        return rows, meta, title

    if input_path.endswith(".npz") and os.path.isfile(input_path):
        d = np.load(input_path)
        n = len(d["az_deg"]) if "az_deg" in d.files else 0
        rows = []
        for i in range(0, n, max(1, stride)):
            if max_rows > 0 and len(rows) >= max_rows:
                break
            row = {
                "n": i + 1,
                "az": float(d["az_deg"][i]),
                "el": float(d["el_deg"][i]),
                "snr_db": float(d["snr_db"][i]) if "snr_db" in d.files else 0.0,
                "papr_db": float(d["papr_db"][i]) if "papr_db" in d.files else 0.0,
            }
            if "spec2d" in d.files:
                row["spec2d"] = d["spec2d"][i]
            rows.append(row)
        if "freq_hz" in d.files:
            meta["freq_hz"] = float(d["freq_hz"])
        return rows, meta, os.path.basename(input_path)

    raise ValueError(f"Cannot load plot data from {input_path!r}")
