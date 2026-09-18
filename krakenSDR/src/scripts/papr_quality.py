#!/usr/bin/env python3
"""
papr_quality.py — Is the spatial PAPR a justified acceptance gate?

The pipeline rejects bursts whose DOA spectrum is too flat, on the argument that
a flat spectrum means no steering vector explains the measured phases.  That is
plausible, but the threshold in ``doa_config.toml`` was set by experience.  This
script tests the argument against the data and produces the three figures the
thesis uses for it:

  ``coh_papr_cal``        distribution of PAPR_s with and without the per-channel
                          phase calibration, on exactly the same bursts.  The
                          data are equally coherent in both cases; only agreement
                          with the array model changes.

  ``coh_papr_error``      PAPR_s against the absolute angular error measured with
                          respect to the TLE bearing of the associated satellite.
                          If the gate is well chosen, the accepted region should
                          hold the low-error population.

  ``coh_multiburst``      coherence efficiency, eigenvalue spread and angular
                          error as the covariance is averaged over B bursts of
                          one track.  Growing B raises the rank; satellite motion
                          inside the window eventually limits it.

Everything is recomputed from the per-burst ``y_per_peak`` vectors saved by
``reprocess_session.py`` — the matched-filter output, five complex numbers per
burst — so the raw session is never re-read.  The calibration is stripped by
multiplying it back out, which is exact: ``apply_phase_correction`` is a
per-channel unit-modulus rotation.

Usage (run from krakenSDR/src/):

    python3 scripts/papr_quality.py ../data/doa_iridium/session_.../ \
        --subdir doa_multi_music --out /tmp/papr
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.doa_uca_2d import (  # noqa: E402
    UcaConfig,
    doa_music_uca_2d,
    find_peak_uca_2d,
)
from core.pnt_solver import (  # noqa: E402
    associate_by_doppler,
    azel_to_enu,
    observer_ecef,
    predicted_los_enu,
)
from core.recording import session_frame_times  # noqa: E402
from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
)

F0_HZ = 1_626_270_000.0


# =============================================================================
# metrics
# =============================================================================

def eta(R: np.ndarray) -> float:
    ev = np.abs(np.linalg.eigvalsh(R))
    return float(ev[-1] / (ev.sum() + 1e-30))


def eig_spread_db(R: np.ndarray) -> float:
    ev = np.sort(np.abs(np.linalg.eigvalsh(R)))
    return float(10.0 * np.log10(ev[-1] / (np.mean(ev[:-1]) + 1e-30) + 1e-30))


def solve_azel(R: np.ndarray, cfg: UcaConfig) -> tuple[float, float, float]:
    """(az, el, PAPR_s) from a covariance, via the pipeline's own estimator."""
    spec = doa_music_uca_2d(np.zeros((cfg.n_ant, 1), dtype=complex), cfg, R_in=R)
    return find_peak_uca_2d(spec, cfg)


def angular_error_deg(az_a, el_a, az_b, el_b) -> float:
    """Great-circle angle between two bearings, in degrees."""
    ua = azel_to_enu(az_a, el_a)
    ub = azel_to_enu(az_b, el_b)
    return float(np.degrees(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0))))


# =============================================================================
# loading
# =============================================================================

def load_bursts(session: str, subdir: str, cal_deg: np.ndarray) -> list[dict]:
    """One record per (burst, peak), with calibrated and uncalibrated y."""
    files = sorted(glob.glob(os.path.join(session, subdir, "burst_*.npz")))
    if not files:
        raise SystemExit(f"No burst_*.npz in {os.path.join(session, subdir)}")
    frame_ts = session_frame_times(session)
    undo = np.exp(1j * np.deg2rad(cal_deg))     # inverse of apply_phase_correction

    out: list[dict] = []
    for path in files:
        d = np.load(path, allow_pickle=True)
        y_all = np.atleast_2d(d["y_per_peak"])
        cfo_all = np.atleast_1d(d["cfo_per_peak"])
        frame = int(d["frame"])
        t_abs = float(frame_ts[frame]) if frame_ts is not None and frame < len(frame_ts) \
            else float(d["t"])
        for k in range(y_all.shape[0]):
            y = np.asarray(y_all[k], dtype=complex)
            if not np.isfinite(y).all() or np.allclose(y, 0):
                continue
            out.append({
                "t": t_abs,
                "frame": frame,
                "peak": k,
                "cfo_hz": float(cfo_all[k]) if k < cfo_all.size else float(d["cfo_hz"]),
                "y_cal": y,
                "y_raw": y * undo[: y.size],
            })
    print(f"[data] {len(out)} (burst, peak) records from {len(files)} bursts")
    return out


def truth_bearings(session: str, recs: list[dict], tol_hz: float,
                   min_el_deg: float = 5.0):
    """TLE azimuth/elevation for the satellite each burst is assigned to."""
    from scripts.pnt_solve import TleSource
    src = TleSource(session)
    t = np.array([r["t"] for r in recs])
    obs = observer_ecef(OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT)

    # Doppler alone cannot arbitrate between 80 catalogue satellites: with that
    # many candidates some below-horizon satellite almost always fits within the
    # tolerance, and the assignment becomes meaningless.  Gate on visibility
    # first — a satellite under the horizon is not the source of the burst —
    # and let Doppler choose among what is actually up.
    states = {}
    for sid in src.satellites:
        pos, vel = src.states(sid, t)
        ok = np.isfinite(pos).all(axis=1)
        if not ok.any():
            continue
        u = np.full((t.size, 3), np.nan)
        u[ok] = predicted_los_enu(pos[ok], obs, OBSERVER_LAT, OBSERVER_LON)
        vis = np.isfinite(u[:, 2]) & (np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
                                      >= min_el_deg)
        if not vis.any():
            continue
        pos = pos.copy(); vel = vel.copy()
        pos[~vis] = np.nan
        vel[~vis] = np.nan
        states[sid] = (pos, vel)
    print(f"[truth] {len(states)} satellites above {min_el_deg:g} deg at some "
          f"point in the session")

    cfo = np.array([r["cfo_hz"] for r in recs])
    ids, _res = associate_by_doppler(cfo, states, obs, f0_hz=F0_HZ, tol_hz=tol_hz)

    az_t = np.full(t.size, np.nan)
    el_t = np.full(t.size, np.nan)
    for s in np.unique(ids[ids >= 0]):
        m = ids == s
        pos, _v = states[int(s)]
        u = predicted_los_enu(pos[m], obs, OBSERVER_LAT, OBSERVER_LON)
        az_t[m] = np.degrees(np.arctan2(u[:, 0], u[:, 1])) % 360.0
        el_t[m] = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
    n_ok = int((ids >= 0).sum())
    print(f"[truth] {n_ok}/{t.size} records matched to a satellite "
          f"({len(np.unique(ids[ids >= 0]))} satellites)")
    return ids, az_t, el_t, src.label if hasattr(src, "label") else {}


# =============================================================================
# figures
# =============================================================================

def fig_papr_cal(recs, cfg, out_base, gate: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import thesis_style as ts

    ts.apply(plt)
    C = ts.SERIES

    p_cal = np.array([r["papr_cal"] for r in recs])
    p_raw = np.array([r["papr_raw"] for r in recs])

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    bins = np.linspace(0, max(p_cal.max(), p_raw.max()) * 1.02, 45)
    ax.hist(p_raw, bins=bins, alpha=0.8, lw=0, color=C[1],
            label="calibration removed")
    ax.hist(p_cal, bins=bins, alpha=0.8, lw=0, color=C[0],
            label="calibration applied")
    # threshold marker: a solid hairline, with headroom so the label is not
    # jammed against the top frame
    ax.set_ylim(top=ax.get_ylim()[1] * 1.14)
    ax.axvline(gate, color=ts.INK, lw=0.9, zorder=4)
    ax.annotate(f"gate = {gate:g} dB", xy=(gate, ax.get_ylim()[1]),
                xytext=(5, -5), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=ts.INK_SOFT)
    ax.set_xlabel(r"spatial PAPR$_\mathrm{s}$ [dB]")
    ax.set_ylabel("bursts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"papr_cal_median": float(np.median(p_cal)),
            "papr_raw_median": float(np.median(p_raw)),
            "pass_gate_cal": float((p_cal >= gate).mean()),
            "pass_gate_raw": float((p_raw >= gate).mean())}


def fig_papr_error(recs, out_base, gate: float):
    """PAPR_s against angular error.

    Both populations are plotted: the same bursts with the calibration applied
    and with it removed.  Without the second set the calibrated data alone
    barely populate the low-PAPR region — nearly every calibrated burst clears
    the gate — and the plot could not show what a low PAPR implies.  Removing
    the calibration does not change the data's coherence, only its agreement
    with the array model, so the two sets together sweep the axis for the same
    physical measurements.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import thesis_style as ts

    ts.apply(plt)
    C = ts.SERIES

    ok = [r for r in recs if np.isfinite(r.get("err_deg", np.nan))
          and np.isfinite(r.get("err_deg_raw", np.nan))]
    p_c = np.array([r["papr_cal"] for r in ok])
    e_c = np.array([r["err_deg"] for r in ok])
    p_r = np.array([r["papr_raw"] for r in ok])
    e_r = np.array([r["err_deg_raw"] for r in ok])

    p = np.concatenate([p_c, p_r])
    e = np.concatenate([e_c, e_r])

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(p_r, e_r, ".", ms=3, alpha=0.30, color=C[1],
            label="calibration removed")
    ax.plot(p_c, e_c, ".", ms=3, alpha=0.30, color=C[0],
            label="calibration applied")

    edges = np.unique(np.quantile(p, np.linspace(0, 1, 13)))
    ctr, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() < 10:
            continue
        ctr.append(0.5 * (a + b))
        med.append(np.median(e[m]))
        q1.append(np.percentile(e[m], 25))
        q3.append(np.percentile(e[m], 75))
    ax.fill_between(ctr, q1, q3, color=ts.INK, alpha=0.14, lw=0, zorder=3)
    ax.plot(ctr, med, "-o", color=ts.INK, lw=1.6, ms=4, zorder=4,
            label="median (both sets)")
    ax.axvline(gate, color=ts.INK, lw=0.9, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 200)
    ax.set_xlabel(r"spatial PAPR$_\mathrm{s}$ [dB]")
    ax.set_ylabel("angular error vs TLE [deg]")
    ax.annotate(f"gate = {gate:g} dB", xy=(gate, ax.get_ylim()[1]),
                xytext=(5, -5), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=ts.INK_SOFT)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3, which="both")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    below, above = e[p < gate], e[p >= gate]
    return {"n_points": int(e.size),
            "median_err_below_gate": float(np.median(below)) if below.size else None,
            "median_err_above_gate": float(np.median(above)) if above.size else None,
            "frac_below_gate": float((p < gate).mean()),
            "spearman": float(_spearman(p, e))}


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-30))


def fig_multiburst(recs, ids, az_t, el_t, cfg, b_values, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import thesis_style as ts

    ts.apply(plt)
    C = ts.SERIES

    order = np.argsort([r["t"] for r in recs])
    rows = []
    for B in b_values:
        etas, spreads, errs = [], [], []
        for s in np.unique(ids[ids >= 0]):
            idx = [i for i in order if ids[i] == s]
            for j in range(0, len(idx) - B + 1, max(1, B // 2)):
                win = idx[j:j + B]
                R = np.zeros((cfg.n_ant, cfg.n_ant), dtype=complex)
                for i in win:
                    y = recs[i]["y_cal"]
                    R += np.outer(y, y.conj()) / (np.vdot(y, y).real + 1e-30)
                R /= len(win)
                az, el, _p = solve_azel(R, cfg)
                mid = win[len(win) // 2]
                if np.isfinite(az_t[mid]):
                    errs.append(angular_error_deg(az, el, az_t[mid], el_t[mid]))
                etas.append(eta(R))
                spreads.append(eig_spread_db(R))
        if etas:
            rows.append((B, np.median(etas), np.median(spreads),
                         np.median(errs) if errs else np.nan, len(etas)))
        print(f"  B={B:3d}  windows={len(etas):5d}", end="\r", flush=True)
    print(" " * 40, end="\r")

    B = np.array([r[0] for r in rows], float)
    et = np.array([r[1] for r in rows])
    sp = np.array([r[2] for r in rows])
    er = np.array([r[3] for r in rows])

    # At B = 1 the matched-filter covariance is rank one by construction: eta is
    # identically 1 and the eigenvalue spread is unbounded (the noise eigenvalues
    # are exactly zero).  That point is plotted for eta, where it is meaningful
    # as the degenerate limit, and suppressed for the spread, where it is not.
    sp_plot = sp.copy()
    sp_plot[B < 2] = np.nan

    # Three measures of different scale share one x.  They are stacked as
    # small multiples rather than folded onto twin y-axes: the alignment of two
    # independent scales on one frame is arbitrary and invents a correlation.
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 4.8), sharex=True,
                            gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})

    axs[0].plot(B, et, "-o", color=C[0], ms=4.5)
    axs[0].set_ylabel(r"$\eta$")
    axs[0].annotate(r"coherence efficiency $\eta$", xy=(0.98, 0.90),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=8, color=ts.INK_SOFT)

    axs[1].plot(B, sp_plot, "-s", color=C[1], ms=4.5)
    axs[1].set_ylabel(r"$\Delta_\mathrm{eig}$ [dB]")
    axs[1].annotate(r"eigenvalue spread $\Delta_\mathrm{eig}$", xy=(0.98, 0.90),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=8, color=ts.INK_SOFT)

    axs[2].plot(B, er, "-^", color=C[2], ms=4.5)
    axs[2].set_ylabel("median error [deg]")
    axs[2].annotate("median angular error", xy=(0.98, 0.90),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=8, color=ts.INK_SOFT)

    axs[2].set_xscale("log", base=2)
    axs[2].set_xlabel("bursts averaged, $B$")
    axs[2].set_xticks(B)
    axs[2].set_xticklabels([f"{int(b)}" for b in B])
    for ax in axs:
        ax.grid(True, which="major")
        ts.strip_spines(ax)
    fig.subplots_adjust(hspace=0.16)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [{"B": int(b), "eta": float(e), "spread_db": float(s),
             "median_err_deg": (None if not np.isfinite(x) else float(x)),
             "windows": int(n)} for b, e, s, x, n in rows]


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--subdir", default="doa_multi_music")
    ap.add_argument("--cal", default="cal_tle.npz",
                    help="calibration file, relative to the session directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--assoc-tol", type=float, default=1500.0)
    ap.add_argument("--min-el", type=float, default=5.0,
                    help="Visibility gate before Doppler association [deg]")
    ap.add_argument("--radius-lambda", type=float, default=0.4253)
    ap.add_argument("--ant0-offset", type=float, default=-4.0)
    ap.add_argument("--gate", type=float, default=2.5)
    ap.add_argument("--b-values", default="1,2,4,8,16,32,64")
    args = ap.parse_args()

    session = args.session.rstrip("/")
    out_dir = args.out or os.path.join(session, "coherence")
    os.makedirs(out_dir, exist_ok=True)

    cal_path = os.path.join(session, args.cal)
    cal = np.load(cal_path)["phase_offsets_deg"]
    print(f"[cal] {os.path.basename(cal_path)}: "
          f"{np.round(cal, 1).tolist()} deg")

    cfg = UcaConfig(n_ant=5, radius_lambda=args.radius_lambda,
                    n_az=360, n_el=86, el_min_deg=5.0, el_max_deg=90.0,
                    num_expected_signals=1, ant0_offset_deg=args.ant0_offset)

    recs = load_bursts(session, args.subdir, cal)

    print("[doa] recomputing spectra with and without calibration ...")
    for n, r in enumerate(recs):
        for tag in ("cal", "raw"):
            y = r["y_" + tag]
            R = np.outer(y, y.conj()) / (np.vdot(y, y).real + 1e-30)
            az, el, papr = solve_azel(R, cfg)
            r["az_" + tag], r["el_" + tag], r["papr_" + tag] = az, el, papr
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(recs)}", end="\r", flush=True)
    print(" " * 40, end="\r")

    ids, az_t, el_t, _lbl = truth_bearings(session, recs, args.assoc_tol,
                                          args.min_el)
    for i, r in enumerate(recs):
        if np.isfinite(az_t[i]):
            r["err_deg"] = angular_error_deg(r["az_cal"], r["el_cal"],
                                             az_t[i], el_t[i])
            r["err_deg_raw"] = angular_error_deg(r["az_raw"], r["el_raw"],
                                                 az_t[i], el_t[i])
        else:
            r["err_deg"] = r["err_deg_raw"] = np.nan

    summary = {"session": os.path.abspath(session), "subdir": args.subdir,
               "n_records": len(recs), "gate_db": args.gate}
    print("\n[fig] coh_papr_cal")
    summary["papr_cal"] = fig_papr_cal(
        recs, cfg, os.path.join(out_dir, "coh_papr_cal"), args.gate)
    print("[fig] coh_papr_error")
    summary["papr_error"] = fig_papr_error(
        recs, os.path.join(out_dir, "coh_papr_error"), args.gate)
    print("[fig] coh_multiburst")
    b_values = [int(v) for v in args.b_values.split(",")]
    summary["multiburst"] = fig_multiburst(recs, ids, az_t, el_t, cfg, b_values,
                                           os.path.join(out_dir, "coh_multiburst"))

    with open(os.path.join(out_dir, "papr_quality.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:1400])
    print(f"\nOutput: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
