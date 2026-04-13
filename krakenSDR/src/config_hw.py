# =============================================================================
#  KrakenSDR — Hardware & RF configuration
#  Edit this file to change hardware, antenna, and radio parameters.
# =============================================================================

# ── Heimdall DAQ connection ───────────────────────────────────────────────────
#  Heimdall runs inside the Docker container with network_mode: host,
#  so ports 5000/5001 are directly reachable at 127.0.0.1.
HEIMDALL_HOST  = "127.0.0.1"   # Heimdall server IP
HEIMDALL_PORT  = 5000           # IQ data port
HEIMDALL_CTRL  = 5001           # control/status port

# ── Antenna array geometry ────────────────────────────────────────────────────
N_ANTENNAS     = 5              # number of KrakenSDR antennas

GEOMETRY       = "UCA"          # "UCA" = uniform circular array  (recommended, full 360°)
#                                # "ULA" = uniform linear  array

RADIUS_LAMBDA  = 0.358          # [UCA] array radius in fractions of λ
#                                #  KrakenSDR 5-ant @ 1626 MHz → physical radius ~6.63 cm
#                                #  λ @ 1626 MHz ≈ 18.5 cm  →  6.63/18.5 ≈ 0.358λ

D_LAMBDA       = 0.5            # [ULA] inter-element spacing in fractions of λ
#                                #  Only used when GEOMETRY = "ULA"

# ── RF / Radio ────────────────────────────────────────────────────────────────
FREQ_HZ        = 1_626_270_000  # carrier frequency [Hz] — Iridium simplex ring alerts
#                                #  Standard Iridium downlink channel (TDMA)

SAMPLE_RATE_HZ = 1.024e6        # ADC sample rate [Hz]
#                                #  Must match daq_chain_config.ini: sample_rate = 1024000

GAIN_DB        = 15             # IF gain [dB]  (same value applied to all channels)
#                                #  Can be overridden per-channel as a list, e.g.: [30, 30, 30, 30, 30]
#                                #  Valid RTL-SDR discrete gain steps (dB):
#                                #  0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4,
#                                #  15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7,
#                                #  32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
#                                #  43.9, 44.5, 48.0, 49.6

# ── Sample budget per frame ───────────────────────────────────────────────────
HW_NUM_SAMPLES  = 0             # IQ samples consumed per Heimdall frame
#                                #  0 = use all samples sent by Heimdall (auto)
#                                #  N > 0 = truncate to first N samples
#                                #  Heimdall typically sends 512–2048 samples/frame
#                                #  Recommended fixed values: 512, 1024, 2048
