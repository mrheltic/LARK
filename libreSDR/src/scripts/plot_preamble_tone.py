#!/usr/bin/env python3
"""
plot_preamble_tone.py — thesis figure: why the IRA preamble is a CW tone.

Generates a two-panel figure from a synthetic IRA burst produced by
`iridium/realistic_sim.py`. No hardware and no recorded session are needed:
the burst is built in software with the same constants the LibreSDR
transmitter uses.

  (a) unwrapped carrier phase at symbol instants. The 64 all-zero preamble
      dibits each advance the phase by exactly +pi/4, so the preamble is a
      straight line whose slope is Rs/8 = 3125 Hz. The unique word and the
      payload carry data, so their phase random-walks.

  (b) magnitude spectrum of the preamble and of an equally long slice of the
      payload, over the same observation window so the resolution matches.
      The preamble collapses into one line at +3125 Hz; the payload spreads
      over the full occupied bandwidth.

Usage (from libreSDR/src/):
    python3 scripts/plot_preamble_tone.py -o preamble_tone.pdf
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iridium.realistic_sim import (  # noqa: E402
    IRA_PREAMBLE_SYMS,
    IRA_UW_SYMS,
    RRC_BETA,
    RRC_NUM_TAPS,
    SAMPLE_RATE,
    SPS,
    SYMBOL_RATE,
    generate_ira_burst,
    generate_rrc_filter,
)

TONE_HZ = SYMBOL_RATE / 8.0   # 3125 Hz — the preamble tone offset


def symbol_phase(iq, start, stop):
    """Unwrapped baseband phase sampled at symbol instants over [start, stop).

    generate_ira_burst pulse-shapes with mode="same", so the RRC group delay is
    already compensated and symbol k sits exactly on sample k * SPS.
    """
    idx = np.arange(start, stop, SPS)
    return idx, np.unwrap(np.angle(iq[idx]))


def spectrum_db(seg, n_fft, ref=None):
    """Zero-padded, Hann-windowed magnitude spectrum in dB.

    Both segments are referenced to the SAME level (`ref`, the preamble peak),
    so the two curves can be read against each other. Normalising each curve to
    its own peak would make the payload look as strong as the tone, which is
    the opposite of the point: the two segments carry comparable power, and the
    preamble is louder per bin only because it concentrates that power.
    """
    win = np.hanning(len(seg))
    spec = np.fft.fftshift(np.fft.fft(seg * win, n_fft))
    mag = np.abs(spec)
    freq = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / SAMPLE_RATE))
    if ref is None:
        ref = mag.max()
    return freq, 20.0 * np.log10(np.maximum(mag / ref, 1e-12)), mag.max()


def build_figure(out_path, n_fft=32768, frame_count=1234):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sized for the thesis text block (~5.4 in) so the figure is placed at
    # roughly 1:1 and the labels stay legible next to 12 pt body text.
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
    })

    rrc = generate_rrc_filter(RRC_BETA, SPS, RRC_NUM_TAPS)
    # frame_count seeds the payload scrambler, whose LFSR is degenerate at 0:
    # with frame_count=0 the payload is all-zero dibits and its phase ramps
    # exactly like the preamble, which would defeat the point of panel (a).
    iq, marks = generate_ira_burst(rrc, sat_id=42, beam_id=7,
                                   frame_count=frame_count)

    pm_s, pm_e = marks["preamble"]
    uw_s, uw_e = marks["uw"]
    da_s, da_e = marks["data"]

    fig, (ax_ph, ax_sp) = plt.subplots(2, 1, figsize=(5.4, 4.9))

    # ── (a) phase ─────────────────────────────────────────────────────────
    idx, ph = symbol_phase(iq, pm_s, da_e)
    t0 = pm_s / SAMPLE_RATE * 1e3
    t_ms = idx / SAMPLE_RATE * 1e3 - t0
    ph = ph - ph[0]
    ax_ph.plot(t_ms, ph, color="#1f77b4", lw=1.4, zorder=3)

    n_pre = IRA_PREAMBLE_SYMS
    t_pre = t_ms[:n_pre]
    ideal = 2 * np.pi * TONE_HZ * (t_pre - t_pre[0]) * 1e-3
    ax_ph.plot(t_pre, ideal, color="#d62728", ls="--", lw=1.2, zorder=4,
               label=r"ideal ramp, $2\pi \cdot 3125\,\mathrm{Hz} \cdot t$")

    uw_ms = uw_s / SAMPLE_RATE * 1e3 - t0
    da_ms = da_s / SAMPLE_RATE * 1e3 - t0
    end_ms = da_e / SAMPLE_RATE * 1e3 - t0
    ax_ph.axvspan(t_ms[0], uw_ms, color="#1f77b4", alpha=0.07, zorder=0)
    ax_ph.axvspan(uw_ms, da_ms, color="0.5", alpha=0.12, zorder=0)
    ax_ph.axvspan(da_ms, end_ms, color="0.5", alpha=0.05, zorder=0)

    trans = ax_ph.get_xaxis_transform()
    ax_ph.text((t_ms[0] + uw_ms) / 2, 0.06, "preamble\n64 sym", transform=trans,
               ha="center", va="bottom", fontsize=8)
    ax_ph.text((uw_ms + da_ms) / 2, 0.35, "UW\n12 sym", transform=trans,
               ha="center", va="center", fontsize=8)
    ax_ph.text((da_ms + end_ms) / 2, 0.92, "payload + tail, 169 sym", transform=trans,
               ha="center", va="top", fontsize=8)

    # tie the ramp to the arithmetic: 64 increments of pi/4 is 16 pi, i.e. 8 turns
    ax_ph.annotate(r"$64 \times \pi/4 = 16\pi$ rad (8 turns)",
                   xy=(t_pre[len(t_pre) // 2], ph[len(t_pre) // 2]),
                   xytext=(0.52, 0.14), textcoords="axes fraction",
                   fontsize=8, color="#d62728",
                   arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9))

    ax_ph.set_xlabel("time within the burst [ms]")
    ax_ph.set_ylabel("unwrapped phase [rad]")
    ax_ph.set_xlim(t_ms[0], end_ms)
    lo, hi = ph.min(), ph.max()
    ax_ph.set_ylim(lo - 0.08 * (hi - lo), hi + 0.30 * (hi - lo))
    ax_ph.grid(alpha=0.3)
    ax_ph.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax_ph.set_title("(a) baseband phase at symbol instants", fontsize=9, loc="left")

    # ── (b) spectra, identical observation window ────────────────────────
    n_seg = IRA_PREAMBLE_SYMS * SPS
    pre = iq[pm_s:pm_s + n_seg]
    pay = iq[da_s:da_s + n_seg]

    f_pre, db_pre, ref = spectrum_db(pre, n_fft)
    f_pay, db_pay, _ = spectrum_db(pay, n_fft, ref=ref)

    ax_sp.plot(f_pay * 1e-3, db_pay, color="0.55", lw=1.0,
               label=f"payload, first {IRA_PREAMBLE_SYMS} symbols")
    ax_sp.plot(f_pre * 1e-3, db_pre, color="#1f77b4", lw=1.4,
               label=f"preamble, {IRA_PREAMBLE_SYMS} symbols")
    ax_sp.axvline(TONE_HZ * 1e-3, color="#d62728", ls="--", lw=1.2)
    ax_sp.annotate(r"$f_c + R_s/8 = f_c + 3125\,\mathrm{Hz}$",
                   xy=(TONE_HZ * 1e-3, 0.5), xytext=(11.5, 7.5),
                   fontsize=9, color="#d62728",
                   ha="center", va="center",
                   arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

    ax_sp.set_xlabel("frequency offset from the burst carrier [kHz]")
    ax_sp.set_ylabel("magnitude [dB rel. tone peak]")
    ax_sp.set_xlim(-20, 20)
    ax_sp.set_ylim(-60, 13)
    ax_sp.grid(alpha=0.3)
    ax_sp.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax_sp.set_title("(b) spectrum over the same observation window",
                    fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")

    # numbers worth quoting in the caption
    peak = f_pre[np.argmax(db_pre)]
    print(f"preamble spectral peak: {peak:.1f} Hz (expected {TONE_HZ:.1f} Hz)")
    slope = np.polyfit(t_pre * 1e-3, ph[:n_pre], 1)[0] / (2 * np.pi)
    print(f"preamble phase slope:   {slope:.1f} Hz")
    print(f"payload peak vs tone:   {db_pay.max():.1f} dB")
    print(f"phase at end of preamble: {ph[n_pre - 1]:.2f} rad = "
          f"{ph[n_pre - 1] / np.pi:.2f} pi")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="preamble_tone.pdf",
                    help="output figure path (pdf or png)")
    ap.add_argument("--nfft", type=int, default=32768, help="FFT length")
    ap.add_argument("--frame-count", type=int, default=1234,
                    help="payload scrambler seed; must be non-zero")
    args = ap.parse_args()
    build_figure(args.out, args.nfft, args.frame_count)


if __name__ == "__main__":
    main()
