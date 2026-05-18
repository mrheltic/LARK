# =============================================================================
#  LibreSDR Central Configuration
#  Zynq7020 + AD9363 (PlutoSDR-compatible firmware)
#
#  Single source of truth for hardware parameters.
#  Import from here instead of hardcoding values in scripts.
#
#  LEGAL NOTE: transmitting in the Iridium band (1616-1626.5 MHz) without a
#  licence is ILLEGAL.  Use one of these safe configurations:
#    1. Wired:     TX -> SMA cable -> >=30 dB attenuator -> RX
#    2. ISM band:  868.1 MHz or 433.92 MHz (no licence required)
#    3. Near-field lab test with authorised supervision only.
# =============================================================================

# -- Hardware connection -------------------------------------------------------
#  pyadi-iio / libiio URI
#    Ethernet (default): "ip:192.168.1.10"
#    USB-OTG:            "usb:"
#    On-board ARM:       "local:"
DEVICE_URI: str = "ip:192.168.1.10"

# -- TX / RX sample rate -------------------------------------------------------
#  Practical minimum for AD9363 over Ethernet: 1 MSPS (TCP overhead).
#  USB-OTG allows up to 61.44 MSPS.
SAMPLE_RATE: int = 1_000_000          # 1 MSPS

# -- RF bandwidth --------------------------------------------------------------
#  Must be >= signal bandwidth.
#  Iridium IRA at 25 ksps + RRC beta=0.4 -> occupied BW ~35 kHz.
#  200 kHz is conservative and covers all common ISM signals.
RF_BANDWIDTH: int = 200_000           # 200 kHz

# -- Iridium frequencies -------------------------------------------------------
IRIDIUM_RING_ALERT_HZ: int = 1_626_270_000   # 1626.270 MHz  (Ring Alert / Simplex)
IRIDIUM_BAND_LOW_HZ:   int = 1_616_000_000   # 1616 MHz (band start -- licensed)
IRIDIUM_BAND_HIGH_HZ:  int = 1_626_500_000   # 1626.5 MHz (band end -- licensed)

# -- ISM test frequencies (no licence required) --------------------------------
ISM_868_HZ:  int = 868_100_000        # 868.1 MHz  (EU ISM, 25 mW ERP limit)
ISM_433_HZ:  int = 433_920_000        # 433.92 MHz (EU ISM, 10 mW ERP limit)

# -- TX gain / attenuation -----------------------------------------------------
#  AD9363 uses attenuation: 0 dB = max power, -89.75 dB = minimum.
#
#  Recommended starting points:
#    -60 dB  : very low power (first connection, sanity check)
#    -30 dB  : bench test (cable + 20 dB attenuator)
#    -20 dB  : indoor ISM test at 1-3 m with KrakenSDR
#    -10 dB  : outdoor ISM test at > 10 m
TX_GAIN_DB: float = -60.0             # start conservative!

# -- RX parameters -------------------------------------------------------------
RX_GAIN_DB:   float = 30.0
RX_GAIN_MODE: str   = "slow_attack"  # "manual" | "slow_attack" | "fast_attack"

# -- Iridium IRA signal parameters ---------------------------------------------
#  Derived from public sources (gr-iridium, iridium-toolkit).
#  Must match iridium/realistic_sim.py exactly.
IRIDIUM_SYMBOL_RATE:   int   = 25_000
IRIDIUM_RRC_BETA:      float = 0.4
IRIDIUM_SPS:           int   = 10      # samples/symbol at 250 kHz base rate
IRIDIUM_PREAMBLE_SYMS: int   = 64      # all-zero dibits -> tone at fc + Rs/8 = fc + 3125 Hz
IRIDIUM_BURST_SYMS:    int   = 245     # preamble + UW + data + tail
IRIDIUM_SUPERFRAME_S:  float = 0.090   # 90 ms TDMA slot period

# -- Pilot-tone parameters (for CW DoA tests) ----------------------------------
#  Offset of the CW pilot tone above the TX LO:
#    0        = plain carrier at DC
#    100_000  = tone at LO + 100 kHz (KrakenSDR default)
#               Falls on FFT bin 12800 at 1.024 MSPS / 131072-pt CPI
PILOT_TONE_OFFSET_HZ: int = 100_000

# -- Connection reliability ----------------------------------------------------
CONNECT_RETRIES:   int   = 3
CONNECT_TIMEOUT_S: float = 5.0
