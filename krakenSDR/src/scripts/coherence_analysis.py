#!/usr/bin/env python3
"""
coherence_analysis.py — Measure the inter-channel coherence of a recorded session.

The DOA pipeline assumes the five channels sample the same instant and hold a
stable inter-channel phase.  This script measures that assumption instead of
taking it for granted, and produces the figures used in the thesis chapter on
channel coherence.

Four questions, four figures:

  1. Are the channels sample-aligned?     ``coh_xcorr_lag``
     Cross-correlation |r_k0(tau)| against lag.  A peak away from tau = 0 is a
     sample offset that no phase calibration can repair.

  2. Is the inter-channel phase stable?   ``coh_phase_stability``
     arg(mu_k) across the session, and |mu_k| on bursts against noise windows.
     Note that arg(mu_k) is *supposed* to change during a pass: it follows the
     satellite.  Its dispersion over a whole session therefore measures source
     motion, not hardware drift.  The hardware residual is what remains after
     the TLE-predicted steering phase is removed, and that is reported by
     ``fit_array_cal.py`` as ``circ_std_deg``.

  3. How coherent is the received power?  ``coh_gamma`` and ``coh_eigen``
     Coherence matrix |Gamma| and the eigenvalue spectrum of the narrowband
     covariance.  Coherence efficiency eta = lambda_1 / tr(R) runs from 1/M on
     independent noise to 1 on a single coherent wavefront.

  4. Does it match the array model?       left to the spatial PAPR reported by
     ``reprocess_session.py`` / ``eval_doa_accuracy.py`` — coherence alone does
     not imply agreement with the steering manifold.

Covariances are formed on the band-pass-filtered preamble *before* matched
filtering.  The matched-filter covariance used by the pipeline is rank one by
construction, so its eta is identically 1 and measures nothing.

Usage
-----
    python3 scripts/coherence_analysis.py ../data/doa_iridium/session_20260605_110716/ \
        --frames 300 --out /tmp/coherence

    # include the calibrated phases in the phase-stability plot
    python3 scripts/coherence_analysis.py <session> --cal cal_tle.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

try:                                          # noqa: E402
    from core.burst_processing import (
        apply_bpf_and_normalize,
        detect_energy_bursts,
        scan_preamble_tones,
    )
except ImportError:                           # --replot needs none of these
    apply_bpf_and_normalize = detect_energy_bursts = scan_preamble_tones = None

NOM_TONE_HZ = 3125.0


# =============================================================================
# Metrics
# =============================================================================

def coherence_efficiency(R: np.ndarray) -> float:
    """lambda_1 / tr(R) — in [1/M, 1].  1/M on independent noise, 1 rank-one."""
    ev = np.abs(np.linalg.eigvalsh(R))
    return float(ev[-1] / (np.sum(ev) + 1e-30))


def eigen_spread_db(R: np.ndarray) -> float:
    """10 log10( lambda_1 / mean(lambda_2..lambda_M) ) [dB]."""
    ev = np.sort(np.abs(np.linalg.eigvalsh(R)))
    return float(10.0 * np.log10(ev[-1] / (np.mean(ev[:-1]) + 1e-30) + 1e-30))


def condition_number(R: np.ndarray) -> float:
    ev = np.sort(np.abs(np.linalg.eigvalsh(R)))
    return float(ev[-1] / (ev[0] + 1e-30))


def xcorr_vs_lag(X: np.ndarray, max_lag: int) -> np.ndarray:
    """
    |r_k0(tau)| for k = 1..M-1, normalised to its own peak.

    Returns (M-1, 2*max_lag+1) real array.
    """
    n_ant, N = X.shape
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    F = np.fft.fft(X, n=nfft, axis=1)
    out = np.empty((n_ant - 1, 2 * max_lag + 1))
    for k in range(1, n_ant):
        r = np.fft.ifft(F[k] * np.conj(F[0]))
        r = np.concatenate([r[-max_lag:], r[: max_lag + 1]])
        mag = np.abs(r)
        out[k - 1] = mag / (mag.max() + 1e-30)
    return out


# =============================================================================
# Session walk
# =============================================================================

def frame_paths(session: str) -> list[str]:
    raw = os.path.join(session, "raw")
    if not os.path.isdir(raw):
        raise SystemExit(f"No raw/ directory in {session}")
    names = sorted(f for f in os.listdir(raw) if f.endswith(".npy"))
    return [os.path.join(raw, f) for f in names]


def analyse(
    session: str,
    n_frames: int,
    fs: float,
    pre_samples: int,
    window_samples: int,
    threshold_factor: float,
    max_lag: int,
    min_tone_snr_db: float,
) -> dict:
    paths = frame_paths(session)
    stride = max(1, len(paths) // n_frames)
    picked = paths[::stride][:n_frames]
    print(f"{len(paths)} frames in session, analysing {len(picked)} (stride {stride})")

    rows: list[dict] = []
    stats = {"no_burst": 0, "weak_tone": 0, "kept": 0}
    xcorr_acc: list[np.ndarray] = []
    gamma_burst: list[np.ndarray] = []
    gamma_noise: list[np.ndarray] = []
    eig_burst: list[np.ndarray] = []
    eig_noise: list[np.ndarray] = []

    for i, path in enumerate(picked):
        X = np.load(path)
        if X.ndim != 2 or X.shape[1] < window_samples:
            continue

        starts = detect_energy_bursts(X, fs, threshold_factor=threshold_factor)
        starts = [s for s in starts if s + window_samples <= X.shape[1]]
        if not starts:
            stats["no_burst"] += 1
            continue
        s0 = starts[0]
        win = X[:, s0 : s0 + window_samples]

        peaks = scan_preamble_tones(win, fs, NOM_TONE_HZ)
        tone_hz, tone_snr = peaks[0]
        if tone_snr < min_tone_snr_db:
            stats["weak_tone"] += 1
            continue
        stats["kept"] += 1

        # --- narrowband covariance on the preamble (before matched filtering) ---
        Xb = apply_bpf_and_normalize(win, pre_samples, fs, tone_hz)
        R = (Xb @ Xb.conj().T) / Xb.shape[1]
        d = np.sqrt(np.abs(np.diag(R)))
        G = R / np.outer(d, d + 1e-30)          # coherence matrix, unit diagonal

        # --- a noise window from the same CPI, same filter ---
        far = _noise_offset(X.shape[1], starts, window_samples)
        Rn = Gn = None
        if far is not None:
            Xn = apply_bpf_and_normalize(
                X[:, far : far + window_samples], pre_samples, fs, tone_hz
            )
            Rn = (Xn @ Xn.conj().T) / Xn.shape[1]
            dn = np.sqrt(np.abs(np.diag(Rn)))
            Gn = Rn / np.outer(dn, dn + 1e-30)

        rows.append(
            {
                "frame": os.path.basename(path),
                "index": i * stride,
                "tone_hz": float(tone_hz),
                "cfo_hz": float(tone_hz - NOM_TONE_HZ),
                "tone_snr_db": float(tone_snr),
                "mu_abs": [float(abs(G[k, 0])) for k in range(1, G.shape[0])],
                "mu_deg": [float(np.degrees(np.angle(G[k, 0]))) for k in range(1, G.shape[0])],
                "eta": coherence_efficiency(R),
                "eig_spread_db": eigen_spread_db(R),
                "cond": condition_number(R),
                "eta_noise": coherence_efficiency(Rn) if Rn is not None else None,
            }
        )

        gamma_burst.append(np.abs(G))
        eig_burst.append(np.sort(np.abs(np.linalg.eigvalsh(R)))[::-1])
        if Gn is not None:
            gamma_noise.append(np.abs(Gn))
            eig_noise.append(np.sort(np.abs(np.linalg.eigvalsh(Rn)))[::-1])

        # Sample alignment must be measured on the *wideband* window: after the
        # 15 kHz BPF the coherence length is fs/BW ~ 68 samples and the
        # correlation peak is too broad to localise.
        if tone_snr > 12.0 and len(xcorr_acc) < 200:
            xcorr_acc.append(xcorr_vs_lag(win, max_lag))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(picked)} frames, {len(rows)} bursts kept")

    if not rows:
        raise SystemExit("No usable bursts found — lower --threshold or use another session.")

    return {
        "rows": rows,
        "xcorr": np.mean(xcorr_acc, axis=0) if xcorr_acc else None,
        "xcorr_n": len(xcorr_acc),
        "gamma_burst": np.median(gamma_burst, axis=0),
        "gamma_noise": np.median(gamma_noise, axis=0) if gamma_noise else None,
        "eig_burst": np.array(eig_burst),
        "eig_noise": np.array(eig_noise) if eig_noise else None,
        "max_lag": max_lag,
        "stats": stats,
    }


def _noise_offset(n_samples: int, starts: list[int], window: int) -> int | None:
    """Offset of a window at least 2 windows away from every detected burst."""
    guard = 2 * window
    for cand in range(0, n_samples - window, window):
        if all(abs(cand - s) > guard for s in starts):
            return cand
    return None


# =============================================================================
# Figures
# =============================================================================

def make_figures(res: dict, out_dir: str, fs: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import thesis_style as ts

    ts.apply(plt)
    C = ts.SERIES

    os.makedirs(out_dir, exist_ok=True)
    rows = res["rows"]
    n_ant = len(rows[0]["mu_abs"]) + 1

    def save(fig, name: str) -> None:
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {name}.png / .pdf")

    # --- 1. sample alignment -------------------------------------------------
    if res["xcorr"] is not None:
        lags = np.arange(-res["max_lag"], res["max_lag"] + 1)
        fig, ax = plt.subplots(figsize=(5.4, 2.9))
        for k in range(res["xcorr"].shape[0]):
            ax.plot(lags, res["xcorr"][k], lw=1.4, color=C[k],
                    label=f"ch {k + 1} vs ch 0")
        # reference rule at zero lag: solid hairline above the grid, never
        # dashed, and no text label -- the x tick and the caption say it
        ax.axvline(0, color=ts.INK_MUTED, lw=0.8, zorder=1.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("lag $\\tau$ [samples]")
        ax.set_ylabel("$|r_{k0}(\\tau)|$ (normalised)")
        n = res.get("xcorr_n") or 0
        ax.set_title(f"Sample alignment, mean over {n} bursts", color=ts.INK)
        # legend below the axes: the curves overlap almost exactly, so any
        # in-plot position sits on data
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  handlelength=1.6, columnspacing=1.2, borderpad=0.4)
        ts.strip_spines(ax)
        save(fig, "coh_xcorr_lag")

    # --- 2. phase stability --------------------------------------------------
    idx = np.array([r["index"] for r in rows], float)
    t_min = idx * 65536 / fs / 60.0
    mu_abs = np.array([r["mu_abs"] for r in rows])
    mu_deg = np.array([r["mu_deg"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.0), sharex=True)
    handles = []
    for k in range(n_ant - 1):
        h, = axes[0].plot(t_min, mu_deg[:, k], ".", ms=2.6, color=C[k],
                          label=f"ch {k + 1}")
        axes[1].plot(t_min, mu_abs[:, k], ".", ms=2.6, color=C[k])
        handles.append(h)
    axes[0].set_ylabel(r"$\arg \mu_k$  [deg]")
    axes[0].set_ylim(-180, 180)
    axes[0].set_yticks([-180, -90, 0, 90, 180])
    axes[1].set_ylabel(r"$|\mu_k|$")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("time into session [min]")
    for ax in axes:
        ts.strip_spines(ax)
    # legend above the panels, clear of the data entirely
    fig.legend(handles=handles, ncol=n_ant - 1, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), handletextpad=0.3,
               columnspacing=1.4, markerscale=3.2)
    fig.subplots_adjust(hspace=0.12)
    save(fig, "coh_phase_stability")

    # --- 3. coherence matrices ----------------------------------------------
    mats = [("burst", res["gamma_burst"])]
    if res["gamma_noise"] is not None:
        mats.append(("noise", res["gamma_noise"]))
    cmap = ts.sequential_cmap()
    fig, axes = plt.subplots(1, len(mats), figsize=(2.85 * len(mats) + 0.7, 2.9))
    axes = np.atleast_1d(axes)
    edges = np.arange(n_ant + 1) - 0.5
    for j, (ax, (label, M)) in enumerate(zip(axes, mats)):
        # pcolormesh, not imshow: imshow embeds the 5x5 matrix as a raster and
        # the PDF viewer interpolates it when scaling up, shearing the cells
        # into parallelograms.  A quadmesh emits one real vector rectangle per
        # cell, so the grid stays square at any zoom.
        im = ax.pcolormesh(edges, edges, M, vmin=0, vmax=1, cmap=cmap,
                           shading="flat", edgecolors="none",
                           rasterized=False)
        ax.set_aspect("equal")
        ax.invert_yaxis()          # row 0 at the top, as imshow had it
        ax.set_title(f"$|\\Gamma|$, {label} (median)", color=ts.INK)
        ax.set_xticks(range(n_ant))
        ax.set_yticks(range(n_ant))
        ax.set_xlabel("channel")
        if j == 0:
            ax.set_ylabel("channel")
        ax.grid(False)
        ax.tick_params(length=0)
        for a in range(n_ant):
            for b in range(n_ant):
                v = M[a, b]
                # ink chosen from the cell's own luminance: never white on light
                rgba = cmap(float(np.clip(v, 0, 1)))
                hexc = "#%02x%02x%02x" % tuple(int(255 * c) for c in rgba[:3])
                ax.text(b, a, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.2, color=ts.on_color(hexc))
    # one shared scale: the two panels use identical limits
    cb = fig.colorbar(im, ax=list(axes), fraction=0.030, pad=0.03)
    cb.outline.set_edgecolor(ts.GRID)
    cb.ax.tick_params(length=0, labelsize=8, colors=ts.INK_SOFT)
    save(fig, "coh_gamma")

    # --- 4. eigenvalue spectrum ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    eb = res["eig_burst"]
    ax = axes[0]
    order = np.arange(1, n_ant + 1)
    eb_db = 10 * np.log10(eb / eb.sum(axis=1, keepdims=True) + 1e-30)
    ax.plot(order, np.median(eb_db, axis=0), "o-", color=C[0], ms=5,
            label="burst", zorder=3)
    ax.fill_between(order, np.percentile(eb_db, 10, axis=0),
                    np.percentile(eb_db, 90, axis=0), color=C[0], alpha=0.16,
                    lw=0)
    if res["eig_noise"] is not None:
        en = res["eig_noise"]
        en_db = 10 * np.log10(en / en.sum(axis=1, keepdims=True) + 1e-30)
        ax.plot(order, np.median(en_db, axis=0), "s-", color=C[1], ms=4.5,
                label="noise", zorder=3)
        ax.fill_between(order, np.percentile(en_db, 10, axis=0),
                        np.percentile(en_db, 90, axis=0), color=C[1],
                        alpha=0.16, lw=0)
    ref = 10 * np.log10(1.0 / n_ant)
    ax.axhline(ref, color=ts.INK_MUTED, lw=0.7, zorder=1)
    ax.annotate("white-noise floor", xy=(n_ant, ref), xytext=(-3, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7.5, color=ts.INK_MUTED)
    ax.set_xticks(order)
    ax.set_xlabel("eigenvalue index $i$")
    ax.set_ylabel(r"$10\log_{10}(\lambda_i / \mathrm{tr}\,\mathbf{R})$  [dB]")
    ax.set_title("Eigenvalue spectrum", color=ts.INK)
    ax.legend(loc="lower left")
    ts.strip_spines(ax)

    ax = axes[1]
    eta_b = np.array([r["eta"] for r in rows])
    bins = np.linspace(1.0 / n_ant, 1.0, 40)
    ax.hist(eta_b, bins=bins, color=C[0], alpha=0.85, label="burst", lw=0)
    eta_n = np.array([r["eta_noise"] for r in rows if r["eta_noise"] is not None])
    if eta_n.size:
        ax.hist(eta_n, bins=bins, color=C[1], alpha=0.72, label="noise", lw=0)
    ax.axvline(1.0 / n_ant, color=ts.INK_MUTED, lw=0.7, zorder=1)
    # annotate inside the axes with headroom, not flush against the frame
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    ax.annotate(f"$1/M = {1 / n_ant:.2f}$", xy=(1.0 / n_ant, ax.get_ylim()[1]),
                xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", fontsize=7.5, color=ts.INK_MUTED)
    ax.set_xlabel(r"coherence efficiency $\eta = \lambda_1/\mathrm{tr}\,\mathbf{R}$")
    ax.set_ylabel("bursts")
    ax.set_title("Coherence efficiency", color=ts.INK)
    ax.legend(loc="upper center")
    ts.strip_spines(ax)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "coh_eigen")


# =============================================================================
# CLI
# =============================================================================

def _cached_burst_count(out_dir: str) -> int:
    """Number of bursts behind a cached run, from the summary written beside the
    npz.  Returns 0 only if the summary is genuinely missing."""
    try:
        with open(os.path.join(out_dir, "coherence_summary.json"),
                  encoding="utf-8") as fh:
            return int(json.load(fh).get("bursts_analysed", 0))
    except (OSError, ValueError, TypeError):
        return 0


def replot(out_dir: str, fs: float) -> None:
    """Redraw the figures from a previous run's cache, without touching the raw
    frames again.  Re-reading a session costs minutes and gigabytes; changing a
    figure size should not."""
    d = np.load(os.path.join(out_dir, "coherence_data.npz"))
    # eta_noise is not stored directly, but it is recoverable from the cached
    # noise eigenvalues: eta = lambda_1 / sum(lambda).
    en = d["eig_noise"]
    eta_n = (en[:, 0] / en.sum(axis=1)) if en.size else None
    rows = [{"mu_abs": list(a), "mu_deg": list(b), "eta": float(e),
             "index": int(i),
             "eta_noise": (float(eta_n[k]) if eta_n is not None and k < eta_n.size
                           else None)}
            for k, (a, b, e, i) in enumerate(
                zip(d["mu_abs"], d["mu_deg"], d["eta"], d["index"]))]
    res = {
        "rows": rows,
        "xcorr": d["xcorr"] if d["xcorr"].size else None,
        # the cache predates storing this; the run that produced it recorded the
        # count in the summary, so recover it there rather than printing zero
        "xcorr_n": _cached_burst_count(out_dir),
        "gamma_burst": d["gamma_burst"],
        "gamma_noise": d["gamma_noise"] if d["gamma_noise"].size else None,
        "eig_burst": d["eig_burst"],
        "eig_noise": d["eig_noise"] if d["eig_noise"].size else None,
        "max_lag": (d["xcorr"].shape[1] - 1) // 2 if d["xcorr"].size else 0,
    }
    make_figures(res, out_dir, fs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session_YYYYMMDD_HHMMSS directory")
    ap.add_argument("--replot", action="store_true",
                    help="Redraw the figures from a previous run's "
                         "coherence_data.npz instead of re-analysing the session")
    ap.add_argument("--frames", type=int, default=300,
                    help="how many CPI frames to sample across the session")
    ap.add_argument("--fs", type=float, default=1_024_000.0)
    ap.add_argument("--pre-samples", type=int, default=2621)
    ap.add_argument("--window-samples", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=6.0,
                    help="energy-detector threshold factor (6 = clean outdoor)")
    ap.add_argument("--min-tone-snr", type=float, default=6.0,
                    help="minimum preamble-tone SNR to accept a burst [dB]")
    ap.add_argument("--max-lag", type=int, default=32,
                    help="half-width of the cross-correlation lag axis [samples]")
    ap.add_argument("--out", default=None, help="output directory (default <session>/coherence)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.session, "coherence")

    if args.replot:
        print(f"Redrawing from {out_dir}/coherence_data.npz")
        replot(out_dir, args.fs)
        return

    res = analyse(
        args.session, args.frames, args.fs, args.pre_samples,
        args.window_samples, args.threshold, args.max_lag, args.min_tone_snr,
    )
    print(f"frames without a burst: {res['stats']['no_burst']}, "
          f"rejected on tone SNR: {res['stats']['weak_tone']}, "
          f"kept: {res['stats']['kept']}")

    rows = res["rows"]
    mu_abs = np.array([r["mu_abs"] for r in rows])
    mu_deg = np.array([r["mu_deg"] for r in rows])
    eta = np.array([r["eta"] for r in rows])
    eta_n = np.array([r["eta_noise"] for r in rows if r["eta_noise"] is not None])

    # circular statistics on the inter-channel phases
    z = np.exp(1j * np.radians(mu_deg))
    Rbar = np.abs(np.mean(z, axis=0))
    circ_mean = np.degrees(np.angle(np.mean(z, axis=0)))
    circ_std = np.degrees(np.sqrt(-2.0 * np.log(np.clip(Rbar, 1e-12, 1.0))))

    peak_lag = None
    if res["xcorr"] is not None:
        peak_lag = (np.argmax(res["xcorr"], axis=1) - res["max_lag"]).tolist()

    summary = {
        "session": os.path.abspath(args.session),
        "bursts_analysed": len(rows),
        "xcorr_peak_lag_samples": peak_lag,
        "mu_abs_median": np.median(mu_abs, axis=0).tolist(),
        "phase_circ_mean_deg": circ_mean.tolist(),
        "phase_circ_std_deg": circ_std.tolist(),
        "phase_circ_std_note": "includes source motion; hardware residual is "
                               "fit_array_cal.py circ_std_deg",
        "eta_median_burst": float(np.median(eta)),
        "eta_median_noise": float(np.median(eta_n)) if eta_n.size else None,
        "eig_spread_db_median": float(np.median([r["eig_spread_db"] for r in rows])),
        "cond_median": float(np.median([r["cond"] for r in rows])),
    }

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "coherence_data.npz"),
        gamma_burst=res["gamma_burst"],
        gamma_noise=res["gamma_noise"] if res["gamma_noise"] is not None else np.zeros(0),
        eig_burst=res["eig_burst"],
        eig_noise=res["eig_noise"] if res["eig_noise"] is not None else np.zeros(0),
        xcorr=res["xcorr"] if res["xcorr"] is not None else np.zeros(0),
        mu_abs=mu_abs, mu_deg=mu_deg, eta=eta,
        index=np.array([r["index"] for r in rows]),
    )
    with open(os.path.join(out_dir, "coherence_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Coherence summary ===")
    print(f"bursts analysed        : {summary['bursts_analysed']}")
    print(f"xcorr peak lag [smp]   : {peak_lag}   (0 = aligned)")
    print(f"|mu_k| median          : {np.round(summary['mu_abs_median'], 3).tolist()}")
    print(f"phase mean [deg]       : {np.round(circ_mean, 1).tolist()}")
    print(f"phase circ. std [deg]  : {np.round(circ_std, 1).tolist()}"
          "   (includes satellite motion — not a hardware-drift figure)")
    print(f"eta burst / noise      : {summary['eta_median_burst']:.3f} / "
          f"{summary['eta_median_noise']:.3f}" if summary["eta_median_noise"]
          else f"eta burst              : {summary['eta_median_burst']:.3f}")
    print(f"eigen spread [dB]      : {summary['eig_spread_db_median']:.1f}")

    print("\nFigures:")
    make_figures(res, out_dir, args.fs)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
