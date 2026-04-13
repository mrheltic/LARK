# =============================================================================
#  KrakenSDR — Hardware configuration (network & DAQ layer only)
#
#  This file contains ONLY hardware-level constants that are independent of
#  the application scenario and frequency plan:
#    • Heimdall DAQ network addresses / ports
#    • Physical hardware constraints (number of channels, ADC rate)
#
#  Everything else — frequency, gain, array geometry, DoA algorithm
#  parameters — lives in the config.py of the relevant application folder:
#    apps/doa/config.py      ← ISM band direction-finding (868 / 433 MHz …)
#    apps/iridium/config.py  ← Iridium L-band burst receive & decode
#    apps/space/config.py    ← 3-D satellite DoA with cross array
#
#  The top-level config.py is a backward-compat skeleton that re-exports
#  these hardware constants for code that has not yet been migrated.
# =============================================================================

# ── Heimdall DAQ connection ───────────────────────────────────────────────────
#  Heimdall runs on the same host (Docker network_mode: host).
#  Ports 5000 / 5001 are bound by daq_start_sm.sh and must match
#  daq_chain_config.ini → [network] / data_ip / ctrl_ip.
HEIMDALL_HOST  = "127.0.0.1"   # Heimdall server address
HEIMDALL_PORT  = 5000           # IQ data stream port
HEIMDALL_CTRL  = 5001           # command / status port

# ── SDR hardware constants ────────────────────────────────────────────────────
N_ANTENNAS     = 5              # number of populated KrakenSDR channels
#                                #  Change only if using a 3-channel Kerberos or custom HW.

SAMPLE_RATE_HZ = 1.024e6        # ADC sample rate [Hz]
#                                #  Must match daq_chain_config.ini: sample_rate = 1024000
#                                #  Valid values for RTL-SDR: 0.25, 0.5, 1.024, 1.4, 1.8,
#                                #  2.048, 2.4, 2.56, 3.2 (MS/s).  1.024 is the most stable.

# ── Frame budget ──────────────────────────────────────────────────────────────
HW_NUM_SAMPLES  = 0             # IQ samples consumed per Heimdall frame
#                                #  0 = use all samples Heimdall sends (automatic, recommended)
#                                #  N > 0 = truncate each frame to the first N samples
#                                #  Typical Heimdall frame sizes: 512 / 1024 / 2048 samples
