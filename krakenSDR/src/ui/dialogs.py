"""
ui.dialogs
==========
Reusable startup dialogs for KrakenSDR applications.

Public API
----------
IridiumConfig          : dataclass  returned by run_iridium_dialog()
run_iridium_dialog()   : show startup config dialog, return IridiumConfig or None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# IridiumConfig — plain dataclass, zero tkinter dependency
# ---------------------------------------------------------------------------
@dataclass
class IridiumConfig:
    freq_hz:    float
    gain_db:    float
    burst_snr:  float
    burst_papr: float
    burst_pwr:  float


# ---------------------------------------------------------------------------
# Startup dialog
# ---------------------------------------------------------------------------
def run_iridium_dialog(
    def_freq:      float,
    def_gain:      float,
    def_snr:       float,
    def_papr:      float,
    def_pwr:       float,
    iridium_chans: dict,
) -> Optional[IridiumConfig]:
    """
    Display the Iridium burst detector startup dialog.

    Parameters
    ----------
    def_freq / def_gain / def_snr / def_papr / def_pwr
        Default values pre-filled in the UI widgets.
    iridium_chans
        ``{label: freq_hz}`` dict used to populate the channel combo-box
        (typically ``core.burst.IRD_CHANS``).

    Returns
    -------
    IridiumConfig
        Filled with the user-chosen values when the user clicks *Start*.
    None
        If the user clicks *Cancel* / closes the window (caller should exit).
    """
    import tkinter as tk
    import tkinter.ttk as ttk
    import tkinter.messagebox as tmsg

    from ui.theme import apply_tk_theme, BLUE

    root = tk.Tk()
    root.title("Iridium Burst Detector – KrakenSDR")
    root.resizable(False, False)

    sty = ttk.Style(root)
    TK_BG, _TK_BG2, _TK_BG3, TK_FG, TK_ACC = apply_tk_theme(sty)
    root.configure(bg=TK_BG)

    FONT = ("Segoe UI", 9)
    P    = {"padx": 10, "pady": 4}

    # ── Title band ────────────────────────────────────────────────────────────
    hdr = ttk.Frame(root, padding="14 10 14 4")
    hdr.grid(row=0, column=0, sticky="ew")
    ttk.Label(
        hdr,
        text="Iridium L-Band  ·  Single-Antenna Burst Detector",
        font=("Segoe UI", 13, "bold"), foreground=TK_ACC,
    ).pack(anchor="w")
    ttk.Label(
        hdr,
        text="Passive TDMA burst detection · 1 antenna · no DoA",
        foreground="#8891b0", font=FONT,
    ).pack(anchor="w")
    ttk.Separator(root).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 4))

    # ── RF / channel section ──────────────────────────────────────────────────
    fr_rf = ttk.LabelFrame(root, text=" RF / Iridium Channel ", padding="12 6")
    fr_rf.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 6))

    chan_names = list(iridium_chans.keys())
    best_chan  = min(chan_names, key=lambda k: abs(iridium_chans[k] - def_freq))
    chan_var   = tk.StringVar(value=best_chan)
    freq_var   = tk.StringVar(value=str(int(def_freq)))
    gain_var   = tk.StringVar(value=str(int(def_gain)))

    ttk.Label(fr_rf, text="Channel preset").grid(row=0, column=0, sticky="w", **P)
    cb = ttk.Combobox(
        fr_rf, textvariable=chan_var, values=chan_names, width=50, state="readonly"
    )
    cb.grid(row=0, column=1, **P)

    ttk.Label(fr_rf, text="Frequency (Hz)").grid(row=1, column=0, sticky="w", **P)
    ttk.Entry(fr_rf, textvariable=freq_var, width=14).grid(
        row=1, column=1, sticky="w", **P
    )

    def _chan_select(e=None):
        sel = chan_var.get()
        freq_var.set(str(int(iridium_chans.get(sel, def_freq))))
    cb.bind("<<ComboboxSelected>>", _chan_select)

    ttk.Label(fr_rf, text="IF Gain (dB)").grid(row=2, column=0, sticky="w", **P)
    ttk.Entry(fr_rf, textvariable=gain_var, width=6).grid(
        row=2, column=1, sticky="w", **P
    )
    ttk.Label(
        fr_rf,
        text=(
            "\u2139  RTL-SDR R820T2 tunes 24\u20131766 MHz \u2714  covers full Iridium band\n"
            "   Gain 30\u201345 dB typical for good LEO reception quality\n"
            "   daq_chain_config.ini: set sample_rate = 1024000"
        ),
        foreground=TK_ACC, font=("Segoe UI", 8),
    ).grid(row=3, column=0, columnspan=2, sticky="w", **P)

    # ── Burst thresholds section ──────────────────────────────────────────────
    fr_th = ttk.LabelFrame(
        root, text=" Burst Detection Thresholds ", padding="12 6"
    )
    fr_th.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 6))

    snr_var  = tk.StringVar(value=str(def_snr))
    papr_var = tk.StringVar(value=str(def_papr))
    pwr_var  = tk.StringVar(value=str(def_pwr))

    def _thresh_row(parent, r, lbl, var, hint):
        ttk.Label(parent, text=lbl).grid(row=r, column=0, sticky="w", **P)
        ttk.Entry(parent, textvariable=var, width=7).grid(
            row=r, column=1, sticky="w", **P
        )
        ttk.Label(parent, text=hint, foreground=TK_ACC,
                  font=("Segoe UI", 8)).grid(row=r, column=2, sticky="w", **P)

    _thresh_row(fr_th, 0, "Burst SNR min (dB)",  snr_var,
                "in-band peak / noise mean  \u2014  start at 6\u20138 dB")
    _thresh_row(fr_th, 1, "Burst PAPR min (dB)", papr_var,
                "in-band peak / in-band mean \u2014 Iridium DQPSK: 4\u20138 dB typical")
    _thresh_row(fr_th, 2, "Power floor (dBW)",   pwr_var,
                "absolute squelch gate  \u2014  lower to \u221292 for weak passes")
    ttk.Label(
        fr_th,
        text=(
            "\u2139  Detector uses non-coherent |FFT(x)|\u00b2, BURST_N=4096 bins\n"
            "   Freq resolution \u2248 250 Hz @ 1.024 MSPS\n"
            "   Signal band |f| \u2264 40 kHz  |  Noise ref |f| > 100 kHz"
        ),
        foreground=TK_ACC, font=("Segoe UI", 8),
    ).grid(row=3, column=0, columnspan=3, sticky="w", **P)

    # ── Buttons ───────────────────────────────────────────────────────────────
    result: list[Optional[IridiumConfig]] = [None]

    def _start():
        try:
            result[0] = IridiumConfig(
                freq_hz    = float(freq_var.get()),
                gain_db    = float(gain_var.get()),
                burst_snr  = float(snr_var.get()),
                burst_papr = float(papr_var.get()),
                burst_pwr  = float(pwr_var.get()),
            )
            root.destroy()
        except ValueError as exc:
            tmsg.showerror("Input error", str(exc), parent=root)

    def _cancel():
        root.destroy()
        raise SystemExit(0)

    bf = ttk.Frame(root, padding="14 0 14 14")
    bf.grid(row=4, column=0, sticky="ew")
    ttk.Button(bf, text="\u25b6  Start Burst Detector", command=_start).pack(
        side="left", fill="x", expand=True, padx=(0, 6)
    )
    ttk.Button(bf, text="\u2715  Cancel", command=_cancel).pack(
        side="left", fill="x", expand=True
    )

    # Centre on screen
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(
        f"+{sw // 2 - root.winfo_width() // 2}+{sh // 2 - root.winfo_height() // 2}"
    )
    root.lift()
    root.focus_force()
    root.mainloop()
    return result[0]
