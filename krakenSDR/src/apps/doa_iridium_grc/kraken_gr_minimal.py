#!/usr/bin/env python3
"""
kraken_gr_minimal.py — Minimal GNU Radio pipeline for KrakenSDR data collection.

Connects to Heimdall DAQ via krakensdr_source (gr-krakensdr OOT module),
displays live FFT spectrum for all 5 channels and computes cross-correlation.

Usage:
    python3 kraken_gr_minimal.py [options]

    Options:
        --freq 1626.27e6    Center frequency [Hz]
        --gain 40.2          IF gain [dB]
        --host 127.0.0.1     Heimdall IP
        --port 5000           IQ data port
        --ctrl 5001          Control port
        --cpi-size 131072    CPI length (from Heimdall config)
        --save FILENAME      Save raw IQ to .npy file
        --save-frames 10     Number of frames to save (default: 10)
        --no-gui             Run without GUI (save only)
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_GR_KRAKEN = os.path.abspath(os.path.join(_SRC, "..", "..", "external", "gr-krakensdr", "python"))
if _GR_KRAKEN not in sys.path:
    sys.path.insert(0, _GR_KRAKEN)


def parse_args():
    p = argparse.ArgumentParser(description="Minimal KrakenSDR GNU Radio pipeline")
    p.add_argument("--freq", type=float, default=1626.27e6)
    p.add_argument("--gain", type=float, default=40.2)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--ctrl", type=int, default=5001)
    p.add_argument("--cpi-size", type=int, default=131072)
    p.add_argument("--save", type=str, default=None, help="Save IQ frames to .npy")
    p.add_argument("--save-frames", type=int, default=10)
    p.add_argument("--no-gui", action="store_true", help="No GUI, save and exit")
    return p.parse_args()


def run_standalone_save(args):
    """Save frames using KrakenIQSource (no GNU Radio dependency)."""
    from hardware.kraken_iq_source import KrakenIQSource

    src = KrakenIQSource(
        host=args.host, port=args.port, ctrl_port=args.ctrl,
        num_channels=5, freq_hz=args.freq, gain_db=args.gain,
        queue_size=4, verbose=2,
    )
    src.start()

    frames = []
    print(f"Collecting {args.save_frames} frames...")
    for i in range(args.save_frames):
        frame = src.get_frame(timeout=10.0)
        if frame is None:
            print(f"  Frame {i}: timeout")
            continue
        frames.append(frame)
        ch0_pwr = np.mean(np.abs(frame[0, :]) ** 2)
        print(f"  Frame {i}: shape={frame.shape}, power(ant0)={ch0_pwr:.6f}")

    src.stop()

    if frames and args.save:
        arr = np.stack(frames)
        np.save(args.save, arr)
        print(f"Saved {arr.shape[0]} frames to {args.save} (shape={arr.shape})")
    elif not frames:
        print("No frames collected!")


def run_gnuradio_gui(args):
    """GNU Radio flowgraph with live FFT display."""
    from gnuradio import gr, qtgui
    from krakensdr import krakensdr_source

    tb = gr.top_block("KrakenSDR Minimal")

    print(f"\n{'='*60}")
    print(f"  KrakenSDR Minimal Pipeline (GNU Radio)")
    print(f"  Freq: {args.freq/1e6:.3f} MHz | Gain: {args.gain:.1f} dB")
    print(f"  Source: {args.host}:{args.port}")
    print(f"  CPI size: {args.cpi_size}")
    print(f"{'='*60}\n")

    src = krakensdr_source(
        ipAddr=args.host, port=args.port, ctrlPort=args.ctrl,
        numChannels=5, freq=args.freq / 1e6,
        gain=[args.gain] * 5, debug=True,
    )

    fft_size = 4096

    sinks = []
    for ch in range(5):
        s2v = gr.stream_to_vector(gr.sizeof_gr_complex, fft_size)
        fft = qtgui.freq_sink_c(
            fft_size, 5, 0, args.freq,
            f"Antenna {ch}",
        )
        fft.set_update_time(0.1)
        fft.set_y_axis(-120, 0)
        tb.connect((src, ch), (s2v, 0))
        tb.connect((s2v, 0), (fft, 0))
        sinks.append(fft)

    head = gr.head(gr.sizeof_gr_complex, args.cpi_size * 5)
    save_vec = gr.stream_to_vector(gr.sizeof_gr_complex, args.cpi_size)
    nah = blocks.file_sink(gr.sizeof_gr_complex, "/dev/null")

    tb.start()
    try:
        input("Press Enter to stop...\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        tb.stop()
        tb.wait()
        print("\nStopped.")


def run_gnuradio_headless(args):
    """GNU Radio flowgraph without GUI — collects and prints frame info."""
    from gnuradio import gr
    from krakensdr import krakensdr_source

    tb = gr.top_block("KrakenSDR Headless")

    print(f"\n{'='*60}")
    print(f"  KrakenSDR Headless Pipeline (GNU Radio)")
    print(f"  Freq: {args.freq/1e6:.3f} MHz | Gain: {args.gain:.1f} dB")
    print(f"  Source: {args.host}:{args.port}")
    print(f"  CPI size: {args.cpi_size}")
    print(f"{'='*60}\n")

    src = krakensdr_source(
        ipAddr=args.host, port=args.port, ctrlPort=args.ctrl,
        numChannels=5, freq=args.freq / 1e6,
        gain=[args.gain] * 5, debug=True,
    )

    class frame_collector(gr.sync_block):
        def __init__(self, n_frames=10, save_path=None):
            gr.sync_block.__init__(self, name="frame_collector",
                                   in_sig=[np.complex64] * 5,
                                   out_sig=None)
            self.n_frames = n_frames
            self.save_path = save_path
            self.collected = 0
            self.current_frame = []
            self.frames = []
            self.samples_since_last = 0

        def work(self, input_items, output_items):
            n = len(input_items[0])
            for ch in range(5):
                pass
            self.samples_since_last += n
            if self.samples_since_last >= args.cpi_size:
                self.samples_since_last = 0
                self.collected += 1

                pwr = [np.mean(np.abs(input_items[ch][:args.cpi_size])**2) for ch in range(5)]
                corr = np.abs(np.corrcoef(
                    np.array([input_items[0][:args.cpi_size],
                               input_items[1][:args.cpi_size]])
                )[0, 1])

                print(f"  Frame {self.collected}: "
                      f"pwr=[{pwr[0]:.5f},{pwr[1]:.5f},{pwr[2]:.5f},{pwr[3]:.5f},{pwr[4]:.5f}] "
                      f"corr(0,1)={corr:.4f}")

                if self.collected >= self.n_frames:
                    tb.stop()
            return n

    collector = frame_collector(n_frames=args.save_frames, save_path=args.save)
    for ch in range(5):
        tb.connect((src, ch), (collector, ch))

    tb.start()
    try:
        tb.wait()
    except KeyboardInterrupt:
        pass
    finally:
        tb.stop()
        tb.wait()
        print("\nDone.")


def main():
    args = parse_args()

    if args.no_gui:
        print("Running in save-only mode (KrakenIQSource, no GNU Radio)")
        run_standalone_save(args)
    else:
        try:
            from gnuradio import gr
            HAS_GR = True
        except ImportError:
            HAS_GR = False

        try:
            from gnuradio import qtgui
            HAS_QTGUI = True
        except ImportError:
            HAS_QTGUI = False

        if HAS_GR and HAS_QTGUI:
            run_gnuradio_gui(args)
        elif HAS_GR:
            run_gnuradio_headless(args)
        else:
            print("GNU Radio not available, falling back to KrakenIQSource save mode")
            run_standalone_save(args)


if __name__ == "__main__":
    main()