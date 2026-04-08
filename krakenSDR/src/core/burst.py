"""
core.burst
==========
Iridium TDMA burst detector — pure DSP, zero I/O, fully unit-testable.

Public API
----------
IRD_CHANS          : dict[str, float]   channel label → centre frequency [Hz]
TDMA_FRAME_S       : float
TDMA_SLOT_S        : float
MAX_DOP_HZ         : float
NEW_PASS_HZ        : float
PASS_TIMEOUT_S     : float

BurstResult        : dataclass   per-frame detection result
BurstDetector      : class       stateless FFT-based detector
PassTracker        : class       stateful Doppler pass tracker
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Iridium L-band constants (refs: ITU-R M.1031, ETSI EN 300 461)
# ---------------------------------------------------------------------------
IRD_CHANS: dict[str, float] = {
    "Simplex  1626.270 MHz  [ring alerts — most active]": 1626.270e6,
    "NEXT ring 1626.104 MHz":                             1626.104e6,
    "Duplex DL 1621.5 MHz":                               1621.500e6,
    "Duplex DL 1623.5 MHz":                               1623.500e6,
}

TDMA_FRAME_S   = 0.090         # one Iridium super-frame [s]
TDMA_SLOT_S    = 0.00828       # one simplex slot [s]
MAX_DOP_HZ     = 40_000.0      # maximum Doppler shift from LEO orbit [Hz]
NEW_PASS_HZ    = 12_000.0      # Doppler jump that marks a new satellite [Hz]
PASS_TIMEOUT_S = 6.0           # silence after which a new pass is assumed [s]

# Pilot tone produced by the Iridium IRA preamble (64 symbols, all-zero
# dibits → Δφ = +π/4 per symbol → pure tone at carrier + Rs/8).
PILOT_TONE_OFFSET_HZ = 25_000.0 / 8.0   # = 3125.0 Hz above carrier
PILOT_TONE_BW_HZ     = 500.0            # ±500 Hz search window around pilot
PILOT_SNR_THRESHOLD  = 6.0              # default pilot SNR threshold [dB]


# ---------------------------------------------------------------------------
# BurstResult — immutable output of BurstDetector.process()
# ---------------------------------------------------------------------------
@dataclass
class BurstResult:
    """Per-frame output of :class:`BurstDetector`. All fields are read-only."""
    spec_db:        np.ndarray   # float32[fft_n] — normalised full-BW spectrum [dB]
    zoom_db:        np.ndarray   # float32[n_spec_cols] — zoom spectrum for spectrogram
    doppler_hz:     float        # FFT peak relative to centre [Hz]
    burst_snr_db:   float        # peak vs noise floor [dB]
    burst_papr_db:  float        # in-band peak-to-average [dB]
    abs_pwr_db:     float        # absolute frame power [dBW]
    is_burst:       bool
    pilot_snr_db:   float = 0.0  # preamble pilot tone SNR vs noise floor [dB]
    pilot_detected: bool  = False # True when pilot_snr_db ≥ detector threshold


# ---------------------------------------------------------------------------
# BurstDetector — stateless per-frame processor
# ---------------------------------------------------------------------------
class BurstDetector:
    """
    Non-coherent Iridium TDMA burst detector.

    All parameters are fixed at construction time.  The three threshold
    attributes (`burst_snr`, `burst_papr`, `burst_pwr`) may be updated at
    runtime between calls to :meth:`process`.

    The object pre-computes all frequency axes: use the read-only properties
    `fft_freqs_kHz`, `spec_freq_kHz` and `n_spec_cols` to size plot axes.
    """

    def __init__(
        self,
        *,
        fs:          float,
        burst_n:     int   = 4096,
        fft_n:       int   = 512,
        max_dop_hz:  float = MAX_DOP_HZ,
        burst_snr:   float = 8.0,
        burst_papr:  float = 5.0,
        burst_pwr:   float = -90.0,
        zoom_factor: float = 1.6,
        pilot_snr:   float = PILOT_SNR_THRESHOLD,
    ) -> None:
        self.fs          = fs
        self.burst_n     = burst_n
        self.fft_n       = fft_n
        self.max_dop_hz  = max_dop_hz
        # Mutable detection thresholds
        self.burst_snr   = burst_snr
        self.burst_papr  = burst_papr
        self.burst_pwr   = burst_pwr
        self.pilot_snr   = pilot_snr

        # ── Precomputed frequency axes (immutable after construction) ──────
        self.fft_freqs_kHz  = np.fft.fftshift(np.fft.fftfreq(fft_n)) * fs / 1e3
        self.burst_freqs_hz = np.fft.fftshift(np.fft.fftfreq(burst_n)) * fs

        zoom_limit      = max_dop_hz * zoom_factor
        self._zoom_mask = np.abs(self.burst_freqs_hz) <= zoom_limit
        self._zoom_idx  = np.where(self._zoom_mask)[0]

        # Public axis for spectrogram columns
        self.spec_freq_kHz: np.ndarray = self.burst_freqs_hz[self._zoom_mask] / 1e3
        self.n_spec_cols:   int        = int(self._zoom_idx.size)

        self._sig_mask   = np.abs(self.burst_freqs_hz) <= max_dop_hz
        self._noise_mask = np.abs(self.burst_freqs_hz) >  max_dop_hz * 2.5
        if not np.any(self._noise_mask):
            self._noise_mask = ~self._sig_mask
        # Last-resort fallback: when fs < 2*max_dop_hz (e.g. narrow-band
        # recordings) the whole spectrum falls inside the signal window and
        # _noise_mask stays empty.  Use the outer 10 % of the available band
        # as a noise reference so the computation never returns nan.
        if not np.any(self._noise_mask):
            boundary = np.abs(self.burst_freqs_hz) >= fs * 0.45
            self._noise_mask = boundary
        if not np.any(self._noise_mask):   # absolute last resort
            self._noise_mask = np.ones(len(self.burst_freqs_hz), dtype=bool)

        # ── Pilot tone mask (absolute offset; Doppler-corrected at runtime) ──
        # Pre-compute a half-bandwidth mask relative to zero; at runtime we
        # apply a roll to centre it on doppler_hz + PILOT_TONE_OFFSET_HZ.
        self._pilot_bw_hz = PILOT_TONE_BW_HZ

    # ------------------------------------------------------------------
    def process(self, x: np.ndarray) -> BurstResult:
        """
        Process one 1-D complex IQ frame.

        Parameters
        ----------
        x : np.ndarray
            Complex IQ samples (any length ≥ 1).  If shorter than `burst_n` or
            `fft_n` the arrays are zero-padded by the FFT automatically.

        Returns
        -------
        BurstResult
            All fields are freshly allocated; the method does not mutate any
            instance attribute.
        """
        N   = min(len(x), self.burst_n)
        win = np.hanning(N)

        # ── Full-bandwidth spectrum (panel A in the UI) ────────────────────
        N_sp  = min(len(x), self.fft_n)
        fa_sp = np.abs(np.fft.fft(x[:N_sp] * np.hanning(N_sp), n=self.fft_n)) ** 2
        sp_db = np.fft.fftshift(10.0 * np.log10(fa_sp + 1e-20))
        sp_db -= float(np.max(sp_db))

        # ── High-res burst FFT ─────────────────────────────────────────────
        fa = np.abs(np.fft.fft(x[:N] * win, n=self.burst_n)) ** 2
        fa = np.fft.fftshift(fa)

        noise_avg     = float(np.mean(fa[self._noise_mask])) + 1e-30
        sig_fa        = fa * self._sig_mask
        pk_bin        = int(np.argmax(sig_fa))
        doppler_hz    = float(self.burst_freqs_hz[pk_bin])
        pk_power      = float(fa[pk_bin])

        burst_snr_db  = float(10.0 * np.log10(pk_power / noise_avg + 1e-12))
        in_band       = fa[self._sig_mask]
        burst_papr_db = float(
            10.0 * np.log10(np.max(in_band) / (np.mean(in_band) + 1e-12) + 1e-12)
        )
        abs_pwr_db    = float(10.0 * np.log10(np.mean(np.abs(x) ** 2) + 1e-15))

        is_burst = (
            burst_snr_db  >= self.burst_snr  and
            burst_papr_db >= self.burst_papr and
            abs_pwr_db    >= self.burst_pwr
        )

        # ── Pilot tone detection ─────────────────────────────────────────
        # The IRA preamble consists of 64 all-zero dibits.  Each zero-dibit
        # maps to +π/4 phase rotation, so the preamble looks like a pure
        # sinusoid at +Rs/8 = +3125 Hz above the burst carrier frequency.
        # After Doppler correction the pilot sits at doppler_hz + 3125 Hz.
        pilot_centre = doppler_hz + PILOT_TONE_OFFSET_HZ
        pilot_mask   = (
            (self.burst_freqs_hz >= pilot_centre - self._pilot_bw_hz) &
            (self.burst_freqs_hz <= pilot_centre + self._pilot_bw_hz)
        )
        if np.any(pilot_mask):
            pilot_power  = float(np.max(fa[pilot_mask]))
            pilot_snr_db = float(10.0 * np.log10(pilot_power / noise_avg + 1e-12))
        else:
            pilot_snr_db = 0.0
        pilot_detected = bool(pilot_snr_db >= self.pilot_snr)

        # ── Zoom spectrum for spectrogram ──────────────────────────────────
        zoom_lin = fa[self._zoom_idx].astype(np.float32)
        zoom_db  = 10.0 * np.log10(zoom_lin + 1e-20)
        zoom_db -= float(np.max(zoom_db))

        return BurstResult(
            spec_db        = sp_db.astype(np.float32),
            zoom_db        = zoom_db,
            doppler_hz     = doppler_hz,
            burst_snr_db   = burst_snr_db,
            burst_papr_db  = burst_papr_db,
            abs_pwr_db     = abs_pwr_db,
            is_burst       = is_burst,
            pilot_snr_db   = pilot_snr_db,
            pilot_detected = pilot_detected,
        )


# ---------------------------------------------------------------------------
# PassTracker — stateful satellite-pass detector
# ---------------------------------------------------------------------------
class PassTracker:
    """
    Detect when the detector is seeing a new Iridium satellite pass.

    A new pass is declared when:
    * this is the first burst ever, OR
    * the Doppler frequency jumped by more than `jump_hz`, OR
    * no burst has been seen for `timeout_s` seconds.

    Usage::

        tracker = PassTracker()
        result  = detector.process(x)
        new_pass = tracker.update(result.doppler_hz, result.is_burst)
        if new_pass:
            print(f"Pass #{tracker.pass_count}")
    """

    def __init__(
        self,
        jump_hz:   float = NEW_PASS_HZ,
        timeout_s: float = PASS_TIMEOUT_S,
    ) -> None:
        self.jump_hz       = jump_hz
        self.timeout_s     = timeout_s
        self.burst_count:  int         = 0
        self.pass_count:   int         = 0
        self.doppler_prev: float | None = None
        self.last_burst_t: float        = 0.0

    # ------------------------------------------------------------------
    def update(
        self,
        dop_hz:   float,
        is_burst: bool,
        now:      float | None = None,
    ) -> bool:
        """
        Register a frame.

        Parameters
        ----------
        dop_hz   : detected Doppler frequency [Hz]
        is_burst : whether `BurstDetector` flagged a burst this frame
        now      : current timestamp (defaults to :func:`time.time`)

        Returns
        -------
        bool
            ``True`` if this burst marks the start of a NEW satellite pass.
        """
        if not is_burst:
            return False

        now      = now if now is not None else _time.time()
        new_pass = self._is_new_pass(dop_hz, now)

        if new_pass:
            self.pass_count += 1
        self.burst_count  += 1
        self.last_burst_t  = now
        self.doppler_prev  = dop_hz
        return new_pass

    # ------------------------------------------------------------------
    def _is_new_pass(self, dop_hz: float, now: float) -> bool:
        if self.doppler_prev is None:
            return True
        if now - self.last_burst_t > self.timeout_s:
            return True
        return abs(dop_hz - self.doppler_prev) > self.jump_hz

    # ------------------------------------------------------------------
    @property
    def time_since_last_burst(self) -> float:
        """Seconds elapsed since the last confirmed burst (0 if none yet)."""
        return _time.time() - self.last_burst_t if self.last_burst_t > 0 else 0.0
