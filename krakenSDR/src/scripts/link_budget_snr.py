#!/usr/bin/env python3
"""
link_budget_snr.py — measured ring-alert tone SNR vs satellite elevation.

Turns the Chapter-3 link budget from an estimate into a measurement for the
reference session: for every detected IRA preamble tone in the raw IQ it
estimates the *absolute* per-element tone SNR in the 40 kHz reference
bandwidth of the budget table, Doppler-assigns the tone to a satellite using
the session TLE snapshot, and plots the result against the TLE true
elevation together with the budget prediction.

Method (per tone):
  1. locate bursts with detect_energy_bursts() on the raw CPI frame;
  2. find the preamble tone with scan_preamble_tones() (per-bin FFT SNR);
  3. matched-filter project the 2621-sample preamble onto the tone frequency
     -> per-antenna complex amplitude a_k (raw ADC counts, before the
     pipeline's amplitude normalisation, so the absolute level survives);
  4. noise power per sample from the median |X|^2 over the whole frame;
  5. SNR40_k = |a_k|^2 / (sigma_k^2 * 40 kHz / fs)  ->  median over antennas;
  6. satellite = argmin |measured CFO - LO - predicted Doppler|, kept if the
     residual is < 3 kHz; elevation from the session TLE at the frame epoch.

Two independent SNR estimates (FFT per-bin and matched filter) agree to
~0.3 dB on average (expected +18.3 dB offset between the two metrics),
which is the internal consistency check.

Usage (from LARK/krakenSDR/src/):
  python3 scripts/link_budget_snr.py <session_dir> \
      --out <figdir> --cache <cache.json>

  --replot          redraw figure + summary from the cache, no raw IQ access
                    (the cache is canonical; full mode writes it)

Outputs:
  <cache>.json          per-tone rows (frame, satellite, elevation, SNR40, ...)
  <out>/link_budget_snr.{png,pdf}
  <cache>_summary.json  binned statistics, per-satellite medians, path-loss fit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)              # krakenSDR/src/
_ROOT = os.path.dirname(os.path.dirname(_SRC))  # LARK project root
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.burst_processing import detect_energy_bursts, scan_preamble_tones  # noqa: E402
from core.recording import session_frame_times                               # noqa: E402
from shared.iridium_tle import load_catalogue, use_session_tle               # noqa: E402

FS = 1_024_000.0
F_C = 1626.27e6
NOM_TONE = 3125.0
B_REF = 40_000.0            # budget reference noise bandwidth [Hz]
WIN = 3000                  # burst window [samples]
N_PRE = 2621                # IRA preamble [samples]
C_LIGHT = 299_792_458.0
LO_OFFSET_HZ = 3354.0       # receiver LO offset measured by eval_doa_accuracy.py
MIN_SCAN_SNR_DB = 15.0      # genuine-tone threshold (per-bin); noise peaks ~never
MAX_DOPPLER_RES_HZ = 3000.0
FRAME_STRIDE = 7            # frames sampled beyond those carrying burst records

# ── budget model (Chapter 3, Table link-budget-elev) ──────────────────────────
H_ORBIT, R_EARTH = 780.0, 6371.0                    # km
SNR_ZENITH_DB = 23.5                                # with external LNA
EL_ROLLOFF = np.array([5.0, 10.0, 15.0, 25.0, 90.0])   # deg
GR_ROLLOFF = np.array([-7.0, -6.0, -5.0, -3.0, 1.5])   # dBic


def slant_range_km(el_deg: float) -> float:
    e = np.radians(el_deg)
    return float(np.sqrt((R_EARTH + H_ORBIT) ** 2 - (R_EARTH * np.cos(e)) ** 2)
                 - R_EARTH * np.sin(e))


def fspl_db(d_km: float) -> float:
    return 20.0 * np.log10(4.0 * np.pi * d_km * 1e3 * F_C / C_LIGHT)


def budget_snr_db(el_deg):
    """Predicted per-element SNR in 40 kHz (with LNA) at elevation el."""
    el = np.atleast_1d(np.asarray(el_deg, float))
    d = np.array([slant_range_km(x) for x in el])
    gr = np.interp(el, EL_ROLLOFF, GR_ROLLOFF)
    out = (SNR_ZENITH_DB
            - (fspl_db(d) - fspl_db(slant_range_km(90.0)))
            - (1.5 - gr))
    return out if np.ndim(el_deg) else float(out[0])


# ── full mode: measure from raw IQ ────────────────────────────────────────────
def measure(session_dir: str) -> list[dict]:
    ts = session_frame_times(session_dir)
    cpi_dur = 65536 / FS

    use_session_tle(session_dir)
    cat = load_catalogue()
    gt = json.load(open(os.path.join(session_dir, "groundtruth.json")))
    lat, lon, alt = gt["observer"]["lat"], gt["observer"]["lon"], gt["observer"]["alt_m"]
    sat_names = sorted({p["name"] for p in gt["passes"]})

    # frames that carry pipeline burst records + a strided sample of the rest
    rec_frames = set()
    for f in sorted(os.listdir(os.path.join(session_dir, "doa_multi_music"))):
        if f.startswith("burst_") and f.endswith(".npz"):
            d = np.load(os.path.join(session_dir, "doa_multi_music", f),
                        allow_pickle=True)
            rec_frames.add(int(d["frame"]))
    frames = sorted(rec_frames | set(range(0, len(ts), FRAME_STRIDE)))
    print(f"frames to process: {len(frames)} (with burst records: {len(rec_frames)})")

    rows: list[dict] = []
    for i, fi in enumerate(frames):
        path = os.path.join(session_dir, "raw", f"frame_{fi:06d}.npy")
        if not os.path.isfile(path):
            continue
        X = np.load(path)
        if X.shape[1] < WIN + 16:
            continue
        sigma2 = np.median(np.abs(X) ** 2, axis=1)      # (5,) full-band noise
        starts = detect_energy_bursts(X, FS, threshold_factor=2.5)
        if not starts:
            continue
        epoch_s = ts[fi]
        pred = {}
        for sat in sat_names:
            try:
                fd, el = _predict_doppler_el(cat, sat, lat, lon, alt, epoch_s)
            except Exception:
                continue
            if fd is not None and el > 0.0:
                pred[sat] = (fd, el)

        for b0 in starts:
            bend = min(b0 + WIN, X.shape[1])
            if bend - b0 < N_PRE + 128:
                continue
            win = X[:, b0:bend]
            tones = scan_preamble_tones(win, FS, NOM_TONE, scan_bw_hz=45_000.0,
                                        n_peaks=3, min_snr_db=MIN_SCAN_SNR_DB,
                                        dc_guard_hz=500.0)
            for tone_hz, scan_snr in tones:
                ref = np.exp(-2j * np.pi * tone_hz / FS * np.arange(N_PRE))
                y = (win[:, :N_PRE] * ref[None, :]).mean(axis=1)
                tone_p = np.maximum(np.abs(y) ** 2 - sigma2 / N_PRE, 1e-30)
                snr40 = 10.0 * np.log10(tone_p / (sigma2 * B_REF / FS))
                meas_dop = tone_hz - NOM_TONE
                best, best_res = None, 1e18
                for sat, (fd, el) in pred.items():
                    res = abs(meas_dop - fd - LO_OFFSET_HZ)
                    if res < best_res:
                        best, best_res = sat, res
                if best is None or best_res > MAX_DOPPLER_RES_HZ:
                    continue
                rows.append(dict(frame=fi, t_off=epoch_s - ts[0] + cpi_dur,
                                 sat=best, el=pred[best][1],
                                 snr40=float(np.median(snr40)),
                                 scan_snr=float(scan_snr),
                                 tone_hz=float(tone_hz),
                                 meas_dop=float(meas_dop),
                                 pred_dop=pred[best][0], res=best_res))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(frames)} frames, {len(rows)} tones")
    print(f"total genuine tones: {len(rows)}")
    return rows


def _predict_doppler_el(cat, sat, lat, lon, alt, epoch_s):
    from datetime import timedelta
    t = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    _, el, rng = cat.sat_azel(sat, lat, lon, alt, t)
    if el < 0:
        return None, el
    r = []
    for dt in (-0.25, 0.25):
        _, _, rr = cat.sat_azel(sat, lat, lon, alt, t + timedelta(seconds=dt))
        r.append(rr)
    fd = -(r[1] - r[0]) / 0.5 * 1e3 / C_LIGHT * F_C
    return fd, el


# ── statistics + figure ──────────────────────────────────────────────────────
ELEV_BINS = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
             (30, 40), (40, 55), (55, 91)]
MAIN_SATS = ("IRIDIUM 100", "IRIDIUM 125", "IRIDIUM 133", "IRIDIUM 136")
SAT_COLORS = {  # light-theme palette
    "IRIDIUM 100": "#1f77b4",
    "IRIDIUM 125": "#d62728",
    "IRIDIUM 133": "#2ca02c",
    "IRIDIUM 136": "#9467bd",
}


def summarise(rows: list[dict]) -> dict:
    el = np.array([r["el"] for r in rows])
    snr = np.array([r["snr40"] for r in rows])
    bins = []
    for lo, hi in ELEV_BINS:
        m = (el >= lo) & (el < hi)
        if m.sum() == 0:
            bins.append(dict(el_bin=[lo, hi], n=0))
            continue
        bins.append(dict(el_bin=[lo, hi], n=int(m.sum()),
                         p50=float(np.median(snr[m])),
                         p75=float(np.percentile(snr[m], 75)),
                         p90=float(np.percentile(snr[m], 90)),
                         max=float(snr[m].max()),
                         thesis=float(budget_snr_db((lo + hi) / 2)),
                         n_snr8=int((snr[m] >= 8.0).sum())))

    per_sat = {}
    for sat in sorted({r["sat"] for r in rows}):
        s = np.array([r["snr40"] for r in rows if r["sat"] == sat])
        e = np.array([r["el"] for r in rows if r["sat"] == sat])
        per_sat[sat] = dict(n=int(len(s)), el_min=float(e.min()),
                            el_max=float(e.max()), p50=float(np.median(s)))

    # common path-loss slope with per-satellite intercepts, on per-(sat,bin)
    # medians well above the detection truncation (p50 >= 3 dB, n >= 5)
    sats, x, y = [], [], []
    for sat in per_sat:
        for lo, hi in ELEV_BINS:
            v = [r["snr40"] for r in rows if r["sat"] == sat and lo <= r["el"] < hi]
            if len(v) >= 5 and np.median(v) >= 3.0:
                sats.append(sat)
                x.append(20.0 * np.log10(slant_range_km((lo + hi) / 2)))
                y.append(np.median(v))
    fit = None
    if len(set(sats)) >= 2 and len(sats) >= 6:
        A = np.zeros((len(sats), len(set(sats)) + 1))
        uniq = sorted(set(sats))
        for i, (s_, xi) in enumerate(zip(sats, x)):
            A[i, uniq.index(s_)] = 1.0
            A[i, -1] = -xi
        coef, *_ = np.linalg.lstsq(A, np.array(y), rcond=None)
        fit = dict(slope_per_fspl=float(coef[-1]),
                   n_points=len(sats), n_sats=len(uniq),
                   note="slope 1.0 = pure free-space path loss; "
                        "fitted on per-(sat,bin) medians with p50>=3dB, n>=5")
    return dict(n_tones=len(rows), bins=bins, per_satellite=per_sat, pathloss_fit=fit)


def make_figure(rows: list[dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    el = np.array([r["el"] for r in rows])
    snr = np.array([r["snr40"] for r in rows])
    sats = np.array([r["sat"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 3.9))

    # budget prediction (with LNA)
    elg = np.linspace(5, 90, 200)
    ax.plot(elg, budget_snr_db(elg), color="#444444", lw=1.6,
            label="link-budget prediction (Table 3, with LNA)", zorder=4)

    # measured tones
    for sat in MAIN_SATS:
        m = sats == sat
        ax.plot(el[m], snr[m], ".", ms=2.4, color=SAT_COLORS[sat],
                alpha=0.45, label=f"{sat.replace('IRIDIUM ', 'Iridium ')}", zorder=2)
    other = ~np.isin(sats, MAIN_SATS)
    ax.plot(el[other], snr[other], ".", ms=2.0, color="#999999", alpha=0.35,
            label="other satellites", zorder=1)

    # binned percentiles
    bx, p50, p90 = [], [], []
    for lo, hi in ELEV_BINS:
        m = (el >= lo) & (el < hi)
        if m.sum() >= 3:
            bx.append((lo + hi) / 2)
            p50.append(np.median(snr[m]))
            p90.append(np.percentile(snr[m], 90))
    ax.plot(bx, p50, "o-", color="#c0392b", ms=3.5, lw=1.2, zorder=5,
            label="measured median per bin")
    ax.plot(bx, p90, "s--", color="#e67e22", ms=3.0, lw=1.0, zorder=5,
            label="measured p90 per bin")

    ax.axvline(9.0, color="#888888", lw=0.9, ls=":", zorder=3)
    ax.annotate("el_min gate of the reference\nprocessing (9°)", xy=(9.5, 26),
                fontsize=7.5, ha="left", color="#666666")

    ax.set_xlabel("satellite elevation [deg]  (TLE, session snapshot)")
    ax.set_ylabel("per-element tone SNR in 40 kHz [dB]")
    ax.set_xlim(0, 92)
    ax.set_ylim(-22, 32)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"link_budget_snr.{ext}"), dpi=200,
                    bbox_inches="tight")
    print(f"  wrote link_budget_snr.png/.pdf in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("session", nargs="?", help="session directory (raw/ + TLE)")
    p.add_argument("--out", default=".", help="figure output directory")
    p.add_argument("--cache", required=True, help="row-cache JSON path")
    p.add_argument("--summary", default=None, help="summary JSON path "
                   "(default: <cache>_summary.json)")
    p.add_argument("--replot", action="store_true",
                   help="load rows from cache, skip raw-IQ measurement")
    args = p.parse_args(argv)

    if args.replot:
        rows = json.load(open(args.cache))
        print(f"loaded {len(rows)} tones from cache")
    else:
        if not args.session:
            p.error("session directory required without --replot")
        rows = measure(args.session)
        json.dump(rows, open(args.cache, "w"))
        print(f"cache written: {args.cache}")

    summary = summarise(rows)
    sp = args.summary or (args.cache + "_summary.json")
    json.dump(summary, open(sp, "w"), indent=1)
    print(f"summary written: {sp}")
    make_figure(rows, args.out)


if __name__ == "__main__":
    main()
