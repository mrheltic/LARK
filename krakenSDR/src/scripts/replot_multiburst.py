#!/usr/bin/env python3
"""Redraw coh_multiburst from the cached papr_quality.json.

fig_multiburst() in papr_quality.py needs the raw per-burst records to compute
the curve; the summary JSON already holds the computed points, so this reuses
the drawing half without re-reading the session.
"""
import json
import sys

import numpy as np


def main() -> int:
    src, out_base = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as fh:
        rows = json.load(fh)["multiburst"]

    B = np.array([r["B"] for r in rows], float)
    et = np.array([r["eta"] for r in rows], float)
    sp = np.array([r["spread_db"] for r in rows], float)
    er = np.array([r["median_err_deg"] for r in rows], float)

    # B = 1 is the rank-one matched-filter limit: eta is identically 1 and the
    # spread is unbounded because the noise eigenvalues are exactly zero.
    sp_plot = sp.copy()
    sp_plot[B < 2] = np.nan

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import thesis_style as ts

    ts.apply(plt)
    C = ts.SERIES

    fig, axs = plt.subplots(3, 1, figsize=(5.2, 4.8), sharex=True)

    panels = [
        (axs[0], et,      C[0], "-o", r"$\eta$",                    r"coherence efficiency $\eta$"),
        (axs[1], sp_plot, C[1], "-s", r"$\Delta_\mathrm{eig}$ [dB]", r"eigenvalue spread $\Delta_\mathrm{eig}$"),
        (axs[2], er,      C[2], "-^", "median error [deg]",          "median angular error"),
    ]
    for ax, y, col, mk, ylab, note in panels:
        ax.plot(B, y, mk, color=col, ms=4.5)
        ax.set_ylabel(ylab)
        ax.annotate(note, xy=(0.98, 0.90), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8, color=ts.INK_SOFT)
        ax.grid(True, which="major")
        ts.strip_spines(ax)

    axs[2].set_xscale("log", base=2)
    axs[2].set_xlabel("bursts averaged, $B$")
    axs[2].set_xticks(B)
    axs[2].set_xticklabels([f"{int(b)}" for b in B])
    fig.subplots_adjust(hspace=0.16)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {out_base}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
