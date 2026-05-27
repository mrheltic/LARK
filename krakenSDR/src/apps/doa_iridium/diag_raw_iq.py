#!/usr/bin/env python3
"""
diag_raw_iq.py — Quick diagnostic: capture raw IQ from KrakenSDR and analyze.

Checks:
1. Signal power vs noise floor
2. FFT spectrum — is there a tone at +3125 Hz?
3. Inter-channel phase coherence
"""
import sys
import os
import time
import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from hardware.kraken_iq_source import KrakenIQSource
import config as C

FREQ  = int(getattr(C, "FREQ_HZ", 1_626_270_000))
GAIN  = getattr(C, "GAIN_DB", 49.0)
N_ANT = getattr(C, "N_ANTENNAS", 5)
FS    = float(getattr(C, "SAMPLE_RATE_HZ", 1_024_000))


def main() -> None:
    print(f"Connecting to Heimdall: {N_ANT} ant, {FREQ/1e6:.3f} MHz, gain={GAIN} dB")
    src = KrakenIQSource(
        host="127.0.0.1", port=5000, ctrl_port=5001,
        num_channels=N_ANT, freq_hz=FREQ, gain_db=GAIN,
        verbose=0,
    )
    src.start()
    try:
        time.sleep(1.0)

        N_FRAMES = 5
        frames = []
        for i in range(N_FRAMES):
            frame = src.get_frame(timeout=3.0)
            if frame is not None:
                frames.append(frame)
                print(f"  Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
            else:
                print(f"  Frame {i}: timeout")

        if not frames:
            print("No frames received!")
            return

        X = np.hstack(frames)
        print(f"\nTotal IQ: shape={X.shape}, {X.shape[1]/FS:.2f}s")

        ch0 = X[0]
        pwr = np.mean(np.abs(ch0)**2)
        pwr_db = 10 * np.log10(pwr + 1e-30)
        print(f"CH0 power: {pwr_db:.1f} dB")

        nfft = min(len(ch0), 65536)
        win = np.blackman(nfft)
        spec = np.abs(np.fft.fft(ch0[:nfft] * win, n=nfft))**2
        freqs = np.fft.fftfreq(nfft, 1.0 / FS)
        spec = np.fft.fftshift(spec)
        freqs = np.fft.fftshift(freqs)

        mask_pos = freqs > 0
        spec_db = 10 * np.log10(spec[mask_pos] + 1e-30)
        freqs_khz = freqs[mask_pos] / 1000

        peak_idx = np.argmax(spec_db)
        peak_freq = freqs_khz[peak_idx]
        peak_pwr = spec_db[peak_idx]
        noise_floor = np.median(spec_db)
        peak_snr = peak_pwr - noise_floor

        print(f"\nSpectrum (CH0, nfft={nfft}):")
        print(f"  Noise floor: {noise_floor:.1f} dB")
        print(f"  Peak: {peak_freq:.1f} kHz, power={peak_pwr:.1f} dB, SNR={peak_snr:.1f} dB")

        top5 = np.argsort(spec_db)[-5:][::-1]
        print("  Top 5 peaks:")
        for idx in top5:
            print(f"    {freqs_khz[idx]:.1f} kHz  {spec_db[idx]:.1f} dB  "
                  f"(SNR={spec_db[idx]-noise_floor:.1f} dB)")

        mask_3k = np.abs(freqs_khz - 3.125) < 0.5
        if np.any(mask_3k):
            print(f"  Power at 3125 Hz: {spec_db[mask_3k].max():.1f} dB  "
                  f"(SNR={spec_db[mask_3k].max()-noise_floor:.1f} dB)")
        mask_dc = np.abs(freqs_khz) < 0.5
        if np.any(mask_dc):
            print(f"  Power at DC: {spec_db[mask_dc].max():.1f} dB")

        print("\nInter-channel analysis:")
        for ch in range(1, X.shape[0]):
            xcorr     = np.abs(np.mean(X[ch] * np.conj(X[0])))
            phase_deg = np.degrees(np.angle(np.mean(X[ch] * np.conj(X[0]))))
            print(f"  CH{ch}: |xcorr with CH0|={xcorr:.4f}  phase_diff={phase_deg:.1f}°")

        R  = (X @ X.conj().T) / X.shape[1]
        ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
        print(f"\nCovariance eigenvalues: {np.round(ev, 4)}")
        print(f"  λ1/λ5 = {ev[0]/(ev[-1]+1e-30):.1f}  (rank-1 expected for single source)")
        print(f"  SINR  = {10*np.log10((ev[0]-np.mean(ev[1:]))/(np.mean(ev[1:])+1e-30)):.1f} dB")

    finally:
        src.stop()


if __name__ == "__main__":
    main()
