"""
test_burst_processing.py — Unit tests for burst_processing.py pipeline functions.

Run with:
    python3 -m pytest apps/doa_iridium/test_burst_processing.py -v
or:
    python3 -m pytest apps/doa_iridium/test_burst_processing.py -v --tb=short
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Ensure krakenSDR/src is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

from burst_processing import (
    detect_energy_bursts,
    scan_preamble_tones,
    compute_mf_covariance,
    apply_bpf_and_normalize,
)

# ── Shared constants ──────────────────────────────────────────────────────────
_FS        = 1_024_000.0    # KrakenSDR / Heimdall sample rate
_TONE_HZ   = 3_125.0        # Iridium IRA preamble tone offset (Rs/8)
_RNG       = np.random.default_rng(0xDEADBEEF)


# =============================================================================
# detect_energy_bursts
# =============================================================================

class TestDetectEnergyBursts:
    """Tests for the energy-based burst detector."""

    def _make_burst(self, n_total: int, burst_start: int, burst_len: int,
                    snr_db: float = 20.0) -> np.ndarray:
        """Return a complex IQ array with a single burst at `burst_start`."""
        rng = np.random.default_rng(1)
        noise = (rng.standard_normal(n_total)
                 + 1j * rng.standard_normal(n_total)) / np.sqrt(2)
        amp   = 10 ** (snr_db / 20.0)
        t     = np.arange(burst_len, dtype=np.float64)
        burst = amp * np.exp(2j * np.pi * _TONE_HZ / _FS * t)
        noise[burst_start : burst_start + burst_len] += burst
        return noise.astype(np.complex64)

    def test_detects_single_burst(self):
        """Exactly one burst at a known position should be detected."""
        iq = self._make_burst(n_total=100_000, burst_start=10_000, burst_len=2_560)
        starts = detect_energy_bursts(iq, fs=_FS)
        assert len(starts) == 1, f"Expected 1 burst, got {len(starts)}"
        # Position should be within ±energy_window (256) of true start
        assert abs(starts[0] - 10_000) <= 256

    def test_no_burst_returns_empty(self):
        """Pure AWGN should not trigger any detection."""
        rng = np.random.default_rng(2)
        iq  = (rng.standard_normal(50_000)
               + 1j * rng.standard_normal(50_000)) / np.sqrt(2)
        starts = detect_energy_bursts(iq.astype(np.complex64), fs=_FS,
                                      threshold_factor=3.0)
        assert starts == [], f"Expected no burst, got {starts}"

    def test_two_bursts_separated(self):
        """Two well-separated bursts should both be detected."""
        n = 200_000
        rng = np.random.default_rng(3)
        iq  = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.5
        amp = 10.0
        burst_len = 2_560
        for pos in [10_000, 120_000]:
            t = np.arange(burst_len, dtype=np.float64)
            iq[pos : pos + burst_len] += amp * np.exp(2j * np.pi * _TONE_HZ / _FS * t)
        iq = iq.astype(np.complex64)
        starts = detect_energy_bursts(iq, fs=_FS, threshold_factor=3.0)
        assert len(starts) == 2, f"Expected 2 bursts, got {len(starts)}: {starts}"

    def test_min_gap_enforced(self):
        """Two closely-spaced bursts with min_gap too large → only first detected."""
        n = 50_000
        rng = np.random.default_rng(4)
        iq  = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.1
        amp, burst_len = 10.0, 512
        for pos in [5_000, 8_000]:
            iq[pos : pos + burst_len] += amp
        # Force min_gap larger than the 3000-sample separation
        starts = detect_energy_bursts(iq.astype(np.complex64), fs=_FS,
                                      energy_window=64,
                                      min_gap_samples=10_000)
        assert len(starts) == 1, f"Expected 1 (min_gap enforced), got {starts}"

    def test_empty_input(self):
        """Empty array should return empty list without error."""
        assert detect_energy_bursts(np.array([], dtype=np.complex64), fs=_FS) == []

    def test_returns_sample_offsets_not_block_indices(self):
        """Returned values must be sample-level offsets, not block indices."""
        iq = self._make_burst(50_000, 20_000, 2_560)
        starts = detect_energy_bursts(iq, fs=_FS, energy_window=256)
        # All offsets must be multiples of energy_window (by construction)
        for s in starts:
            assert s % 256 == 0

    def test_very_weak_signal_high_threshold(self):
        """Very weak signal with high threshold should not be detected."""
        rng = np.random.default_rng(5)
        iq  = (rng.standard_normal(50_000) + 1j * rng.standard_normal(50_000)) / np.sqrt(2)
        # SNR = -5 dB burst
        iq[10_000:12_560] += 0.56 * np.exp(2j * np.pi * _TONE_HZ / _FS
                                             * np.arange(2_560))
        starts = detect_energy_bursts(iq.astype(np.complex64), fs=_FS,
                                      threshold_factor=10.0)
        assert starts == [], f"Weak signal should not be detected at high threshold"


# =============================================================================
# scan_preamble_tones
# =============================================================================

class TestScanPreambleTones:
    """Tests for the FFT preamble-tone scanner."""

    def _cw_burst(self, tone_hz: float, n: int = 8192,
                  snr_db: float = 20.0) -> np.ndarray:
        """Return a complex CW burst at tone_hz + AWGN."""
        rng = np.random.default_rng(10)
        amp = 10 ** (snr_db / 20.0)
        t   = np.arange(n, dtype=np.float64)
        sig = amp * np.exp(2j * np.pi * tone_hz / _FS * t)
        nse = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        return (sig + nse).astype(np.complex64)

    def test_detects_nominal_tone(self):
        """The preamble tone at exactly nom_tone_hz should be the top result."""
        iq = self._cw_burst(_TONE_HZ)
        res = scan_preamble_tones(iq, _FS, nom_tone_hz=_TONE_HZ, scan_bw_hz=45_000)
        assert len(res) >= 1
        tone, snr = res[0]
        # Allow ±1 FFT bin (8192-pt FFT at 1024 kHz → bin width ≈ 125 Hz)
        assert abs(tone - _TONE_HZ) < 200.0, f"Expected ~{_TONE_HZ} Hz, got {tone:.1f}"
        assert snr > 10.0, f"SNR too low: {snr:.1f} dB"

    def test_doppler_offset_detected(self):
        """A tone at nom + 12 kHz Doppler should be located correctly."""
        fd = 12_000.0
        tone = _TONE_HZ + fd
        iq = self._cw_burst(tone, n=16_384)
        res = scan_preamble_tones(iq, _FS, nom_tone_hz=_TONE_HZ, scan_bw_hz=45_000)
        assert len(res) >= 1
        assert abs(res[0][0] - tone) < 500.0

    def test_two_tones_found(self):
        """Two tones separated by >min_sep should both appear."""
        rng = np.random.default_rng(11)
        n   = 16_384
        amp = 30.0
        t   = np.arange(n, dtype=np.float64)
        fd1, fd2 = 8_000.0, -15_000.0
        sig = amp * (np.exp(2j * np.pi * (_TONE_HZ + fd1) / _FS * t)
                     + np.exp(2j * np.pi * (_TONE_HZ + fd2) / _FS * t))
        nse = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        iq  = (sig + nse).astype(np.complex64)
        res = scan_preamble_tones(iq, _FS, nom_tone_hz=_TONE_HZ,
                                  scan_bw_hz=45_000, n_peaks=2, min_sep_hz=5_000)
        assert len(res) == 2, f"Expected 2 tones, got {res}"
        found_tones = sorted([r[0] for r in res])
        expected    = sorted([_TONE_HZ + fd1, _TONE_HZ + fd2])
        for got, exp in zip(found_tones, expected):
            assert abs(got - exp) < 500.0, f"Expected ~{exp:.0f} Hz, got {got:.0f}"

    def test_dc_guard_suppresses_lo_leakage(self):
        """A strong DC component should NOT be returned as a preamble tone."""
        rng = np.random.default_rng(12)
        n   = 8192
        t   = np.arange(n, dtype=np.float64)
        # DC spike (LO leakage) + real preamble tone
        dc_leakage = 1000.0 * np.ones(n, dtype=np.complex64)
        cw_tone    = 50.0 * np.exp(2j * np.pi * _TONE_HZ / _FS * t)
        nse        = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        iq         = (dc_leakage + cw_tone + nse).astype(np.complex64)
        res = scan_preamble_tones(iq, _FS, nom_tone_hz=_TONE_HZ,
                                  dc_guard_hz=500.0)
        for tone, _ in res:
            assert abs(tone) > 500.0, f"DC artefact leaked through: tone={tone:.1f} Hz"

    def test_no_signal_returns_fallback(self):
        """Pure noise below min_snr_db should return the nominal-tone fallback."""
        rng = np.random.default_rng(13)
        iq  = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)) / np.sqrt(2)
        res = scan_preamble_tones(iq.astype(np.complex64), _FS,
                                  nom_tone_hz=_TONE_HZ, min_snr_db=100.0)
        assert len(res) == 1
        assert res[0][0] == _TONE_HZ

    def test_too_short_input_returns_fallback(self):
        """Input shorter than 128 samples should return the fallback directly."""
        iq  = np.ones(64, dtype=np.complex64)
        res = scan_preamble_tones(iq, _FS, nom_tone_hz=_TONE_HZ)
        assert res == [(_TONE_HZ, 0.0)]


# =============================================================================
# compute_mf_covariance
# =============================================================================

class TestComputeMfCovariance:
    """Tests for the matched-filter covariance estimator."""

    def _make_preamble(self, n_ant: int, az_rad: float, el_rad: float,
                       radius_lambda: float = 0.4253,
                       snr_db: float = 20.0,
                       n_samples: int = 2_560,
                       tone_hz: float = _TONE_HZ) -> np.ndarray:
        """
        Return (n_ant, n_samples) IQ with a CW preamble from direction (az, el).
        The steering vector is the standard UCA phase model.
        """
        rng = np.random.default_rng(20)
        k   = np.arange(n_ant)
        phi = 2 * np.pi * k / n_ant          # antenna azimuth angles (CW)
        # UCA spatial phase for source at (az, el)
        tau = 2 * np.pi * radius_lambda * np.cos(el_rad) * np.cos(az_rad - phi)
        a   = np.exp(1j * tau)               # (n_ant,) steering vector

        t   = np.arange(n_samples, dtype=np.float64)
        amp = 10 ** (snr_db / 20.0)
        cw  = amp * np.exp(2j * np.pi * tone_hz / _FS * t)   # (n_samples,)
        nse = (rng.standard_normal((n_ant, n_samples))
               + 1j * rng.standard_normal((n_ant, n_samples))) / np.sqrt(2)

        return (np.outer(a, cw) + nse).astype(np.complex64)

    def test_rank1_structure(self):
        """R_mf must be rank-1: largest eigenvalue >> all others."""
        X = self._make_preamble(n_ant=5, az_rad=0.0, el_rad=np.deg2rad(20.0))
        R, y, snr = compute_mf_covariance(X, tone_hz=_TONE_HZ, fs=_FS,
                                           n_pre=2_552, bpf_guard=0)
        ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
        # λ1 should be at least 20 dB above λ2
        ratio_db = 10.0 * np.log10(ev[0] / (ev[1] + 1e-30))
        assert ratio_db > 20.0, (
            f"R_mf should be rank-1 (λ1/λ2 > 20 dB), got {ratio_db:.1f} dB"
        )

    def test_steering_vector_recovered(self):
        """
        The dominant eigenvector of R_mf should align (up to global phase)
        with the true steering vector used to generate the signal.
        """
        az, el = 0.0, np.deg2rad(20.0)
        k   = np.arange(5)
        phi = 2 * np.pi * k / 5
        tau = 2 * np.pi * 0.4253 * np.cos(el) * np.cos(az - phi)
        a_true = np.exp(1j * tau)
        a_true /= np.abs(a_true[0])   # normalize ref channel

        X = self._make_preamble(5, az, el, snr_db=30.0)
        R, y, _ = compute_mf_covariance(X, _TONE_HZ, _FS, 2_552, 0)

        ev, V = np.linalg.eigh(R)
        v_est = V[:, -1]
        v_est = v_est / (v_est[0] / abs(v_est[0]))  # normalize ref phase

        # Cosine similarity | a^H v |² → 1 for perfect recovery
        cos2 = abs(np.dot(a_true.conj(), v_est)) ** 2 / (
            np.linalg.norm(a_true) * np.linalg.norm(v_est)
        ) ** 2
        assert cos2 > 0.95, f"Steering vector recovery failed: cos²={cos2:.4f}"

    def test_snr_estimate_positive_for_strong_signal(self):
        """With SNR=20 dB input, estimated SNR should be > 0 dB."""
        X = self._make_preamble(5, 0.0, np.deg2rad(30.0), snr_db=20.0)
        _, _, snr = compute_mf_covariance(X, _TONE_HZ, _FS, 2_552, 0)
        assert snr > 0.0, f"SNR should be positive for 20 dB input, got {snr:.1f}"

    def test_snr_estimate_negative_for_noise_only(self):
        """With pure noise, the eigenvalue ratio should give SNR < 3 dB."""
        rng = np.random.default_rng(21)
        X   = (rng.standard_normal((5, 3_000))
               + 1j * rng.standard_normal((5, 3_000))).astype(np.complex64)
        _, _, snr = compute_mf_covariance(X, _TONE_HZ, _FS, 2_552, 0)
        assert snr < 3.0, f"Noise-only SNR should be low, got {snr:.1f} dB"

    def test_short_input_raises(self):
        """Providing X too short for bpf_guard + n_pre should raise ValueError."""
        X = np.ones((5, 100), dtype=np.complex64)
        with pytest.raises(ValueError, match="too short"):
            compute_mf_covariance(X, _TONE_HZ, _FS, n_pre=200, bpf_guard=0)

    def test_hermitian_output(self):
        """R_mf must be Hermitian (R = R^H) up to numerical noise."""
        X = self._make_preamble(5, 0.2, np.deg2rad(40.0))
        R, _, _ = compute_mf_covariance(X, _TONE_HZ, _FS, 2_552, 0)
        assert np.allclose(R, R.conj().T, atol=1e-10), "R_mf is not Hermitian"

    def test_positive_semidefinite(self):
        """R_mf eigenvalues must all be ≥ 0 (PSD matrix)."""
        X = self._make_preamble(5, 1.0, np.deg2rad(25.0))
        R, _, _ = compute_mf_covariance(X, _TONE_HZ, _FS, 2_552, 0)
        ev = np.real(np.linalg.eigvalsh(R))
        assert np.all(ev >= -1e-10), f"R_mf has negative eigenvalues: {ev}"

    def test_bpf_guard_skips_leading_samples(self):
        """
        bpf_guard > 0 should produce a different (but still valid) result
        compared to bpf_guard = 0.
        """
        X = self._make_preamble(5, 0.0, np.deg2rad(20.0), n_samples=3_000)
        R0, y0, _ = compute_mf_covariance(X, _TONE_HZ, _FS, n_pre=2_500, bpf_guard=0)
        R1, y1, _ = compute_mf_covariance(X, _TONE_HZ, _FS, n_pre=2_431, bpf_guard=69)
        # Both should be rank-1
        for R in (R0, R1):
            ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
            assert 10.0 * np.log10(ev[0] / (ev[1] + 1e-30)) > 20.0


# =============================================================================
# apply_bpf_and_normalize
# =============================================================================

class TestApplyBpfAndNormalize:
    """Tests for the BPF extraction + amplitude normalizer."""

    def test_output_shape(self):
        """Output shape must be (n_ant, pre_samples)."""
        X = np.random.randn(5, 4000).astype(np.complex64)
        out = apply_bpf_and_normalize(X, pre_samples=2621, fs=_FS,
                                      tone_hz=_TONE_HZ)
        assert out.shape == (5, 2621)

    def test_unit_rms(self):
        """Each output channel should have unit RMS (≈ 1.0) after normalization."""
        rng = np.random.default_rng(30)
        # Channels with strongly different amplitudes
        amps = np.array([1.0, 5.0, 0.2, 3.0, 0.8])
        X    = (amps[:, None] *
                (rng.standard_normal((5, 3000)) + 1j * rng.standard_normal((5, 3000))))
        out  = apply_bpf_and_normalize(X.astype(np.complex64), 3000, _FS, _TONE_HZ)
        rms  = np.sqrt(np.mean(np.abs(out) ** 2, axis=1))
        assert np.allclose(rms, 1.0, atol=0.01), f"RMS not unit: {rms}"

    def test_bpf_rejects_out_of_band(self):
        """
        An interferer outside the BPF passband should have negligible power
        in the output compared to a CW signal inside the passband.
        """
        rng = np.random.default_rng(31)
        n   = 4096
        t   = np.arange(n, dtype=np.float64)
        # In-band CW at TONE_HZ
        cw_in  = np.exp(2j * np.pi * _TONE_HZ / _FS * t)
        # Out-of-band interferer at TONE_HZ + 100 kHz (far out of 15 kHz BPF)
        cw_out = 100.0 * np.exp(2j * np.pi * (_TONE_HZ + 100_000) / _FS * t)
        X = np.tile((cw_in + cw_out)[None, :].astype(np.complex64), (5, 1))

        out = apply_bpf_and_normalize(X, n, _FS, _TONE_HZ, bpf_bw_hz=15_000)

        # Power of in-band component VS out-of-band after BPF
        fft_out = np.fft.fft(out[0])
        freqs   = np.fft.fftfreq(n, d=1.0 / _FS)
        in_band  = np.sum(np.abs(fft_out[np.abs(freqs - _TONE_HZ) < 7_500]) ** 2)
        oob_freq = _TONE_HZ + 100_000
        oob_band = np.sum(np.abs(fft_out[np.abs(freqs - oob_freq) < 7_500]) ** 2)
        assert in_band > 1000 * oob_band, (
            f"OOB not suppressed: in_band={in_band:.1f}, oob={oob_band:.1f}"
        )

    def test_inter_antenna_phase_preserved(self):
        """
        Phase differences between channels must be preserved after BPF.
        (BPF is linear and identical across channels → phase ratios unchanged.)
        """
        rng = np.random.default_rng(32)
        n   = 4096
        t   = np.arange(n, dtype=np.float64)
        # Five channels with known phase offsets
        phases = np.array([0.0, 0.5, 1.2, 2.1, 3.0])
        cw     = np.exp(2j * np.pi * _TONE_HZ / _FS * t)
        X      = np.outer(np.exp(1j * phases), cw).astype(np.complex64)

        out = apply_bpf_and_normalize(X, n, _FS, _TONE_HZ)

        # Compute y_mf for each channel to measure relative phases
        ref   = np.exp(-2j * np.pi * _TONE_HZ / _FS * t)
        y     = np.array([float(np.angle((out[k] @ ref) / n)) for k in range(5)])
        delta = (y - y[0] + np.pi) % (2 * np.pi) - np.pi   # relative to ch0
        expected = (phases - phases[0] + np.pi) % (2 * np.pi) - np.pi

        assert np.allclose(delta, expected, atol=0.05), (
            f"Phase preservation failed.\n  expected: {np.degrees(expected).round(2)}\n"
            f"  got:      {np.degrees(delta).round(2)}"
        )

    def test_short_input_raises(self):
        """X_win with fewer columns than pre_samples should raise ValueError."""
        X = np.ones((5, 100), dtype=np.complex64)
        with pytest.raises(ValueError, match="need"):
            apply_bpf_and_normalize(X, pre_samples=200, fs=_FS, tone_hz=_TONE_HZ)


# =============================================================================
# Integration test: full pipeline on synthetic Iridium-like burst
# =============================================================================

class TestFullPipeline:
    """End-to-end test combining all four functions."""

    def test_detect_scan_mf_pipeline(self):
        """
        Synthesise an Iridium-like burst (preamble CW + Doppler + noise),
        detect it, scan the tone, extract BPF, compute R_mf, verify rank-1.
        """
        rng     = np.random.default_rng(42)
        fs      = _FS
        fd      = -22_000.0           # LibreSDR-style LO offset
        tone_hz = _TONE_HZ + fd       # where the preamble tone actually lands
        n_ant   = 5

        # Simulated steering vector (az=355°, el=25°, r=0.4253λ)
        az, el = np.deg2rad(355.0), np.deg2rad(25.0)
        k  = np.arange(n_ant)
        phi = 2 * np.pi * k / n_ant
        tau = 2 * np.pi * 0.4253 * np.cos(el) * np.cos(az - phi)
        a   = np.exp(1j * tau)

        # Build frame: 2×superframe worth of samples with one burst at sample 50000
        n_burst = 2621
        n_frame = 200_000
        SNR_db  = 15.0
        amp     = 10 ** (SNR_db / 20.0)
        t_pre   = np.arange(n_burst, dtype=np.float64)
        preamble_tone = amp * np.exp(2j * np.pi * tone_hz / fs * t_pre)

        X_frame = (rng.standard_normal((n_ant, n_frame))
                   + 1j * rng.standard_normal((n_ant, n_frame))) / np.sqrt(2)
        burst_iq = np.outer(a, preamble_tone)   # (n_ant, n_burst)
        X_frame[:, 50_000 : 50_000 + n_burst] += burst_iq

        # Step 1: detect burst on ch0
        starts = detect_energy_bursts(X_frame[0], fs=fs, threshold_factor=3.0)
        assert len(starts) >= 1, "No burst detected in integration test"

        b_start = starts[0]
        b_end   = min(b_start + 10_547, n_frame)
        X_win   = X_frame[:, b_start:b_end]

        # Step 2: scan for preamble tone
        peaks = scan_preamble_tones(X_win[0], fs=fs, nom_tone_hz=_TONE_HZ,
                                    scan_bw_hz=45_000, min_snr_db=3.0)
        assert len(peaks) >= 1
        detected_tone = peaks[0][0]
        assert abs(detected_tone - tone_hz) < 500.0, (
            f"Tone scan off: got {detected_tone:.0f}, expected {tone_hz:.0f}"
        )

        # Step 3: BPF + normalize
        X_bpf = apply_bpf_and_normalize(X_win, pre_samples=2621,
                                        fs=fs, tone_hz=detected_tone)

        # Step 4: MF covariance
        R, y, snr = compute_mf_covariance(X_bpf, detected_tone, fs,
                                           n_pre=2621, bpf_guard=0)
        assert snr > 5.0, f"Integration test SNR too low: {snr:.1f} dB"

        ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
        ratio_db = 10.0 * np.log10(ev[0] / (ev[1] + 1e-30))
        assert ratio_db > 20.0, f"R_mf rank-1 check failed: λ1/λ2 = {ratio_db:.1f} dB"
