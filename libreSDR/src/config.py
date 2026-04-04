# =============================================================================
#  LibreSDR Central Configuration
#  Zynq7020 + AD9363 (firmware PlutoSDR-compatible)
#
#  This file is the single point of truth for all LibreSDR hardware parameters.
#  All TX scripts import from here.
#
#  Edit ALLOWED_FREQS before transmitting: do NOT use the real Iridium band
#  (1616–1626.5 MHz) without a licence. Use authorised ISM frequencies
#  or a wired TX→attenuator→RX connection for lab tests.
# =============================================================================

# ── Hardware connection ────────────────────────────────────────────
#  pyadi-iio / libiio URI
#    Ethernet (default): "ip:192.168.2.1"
#    Direct USB-OTG connection: "usb:"
#    On-board (ARM core): "local:"
DEVICE_URI: str = "ip:192.168.2.1"

# ── TX parameters ──────────────────────────────────────────────────────────────
#  TX centre frequency [Hz]
#  LEGAL NOTE: transmitting in the Iridium band (1616–1626.5 MHz) without
#  a licence is illegal. For testing, use an RF cable + attenuator ≥ 30 dB
#  between TX and RX, or use an authorised ISM frequency.
TX_FREQ_HZ: int = 1_626_270_000     # Iridium Ring Alert – cable/shielded testing ONLY

# Iridium channel for wired tests (= channel 8 from iridium-toolkit BASE_FREQ)
# Uncomment the next line to use the 433 MHz ISM band instead:
# TX_FREQ_HZ = 433_920_000

# TX sample rate [Hz].
# Practical minimum for AD9363 over Ethernet: 1 MSPS (TCP overhead).
# USB-OTG allows up to 61.44 MSPS.
TX_SAMPLE_RATE: int = 1_000_000     # 1 MSPS

# TX RF bandwidth [Hz]
# Must be ≥ signal bandwidth (250 kHz for Iridium-like at 25 ksps + RRC 0.4)
TX_RF_BW: int = 250_000             # 250 kHz

# TX attenuation [dB]   range: 0 (max power) … 89.75 dB (min)
# Start at -60 dB and increase cautiously!
TX_GAIN_ATTENUATION_DB: float = 60.0   # 60 dB attenuation

# ── RX parameters (for loopback and check_device) ───────────────────────────────
RX_FREQ_HZ:      int   = TX_FREQ_HZ
RX_SAMPLE_RATE:  int   = TX_SAMPLE_RATE
RX_RF_BW:        int   = TX_RF_BW
RX_GAIN_MODE:    str   = "slow_attack"   # "manual", "slow_attack", "fast_attack"
RX_GAIN_DB:      float = 30.0

# ── Iridium-like burst parameters ──────────────────────────────────
#  These values MUST match shared.iridium and those used by the KrakenSDR
#  receiver. Do not edit here: import from shared.iridium directly if needed.
IRIDIUM_SYMBOL_RATE: int   = 25_000
IRIDIUM_RRC_BETA:    float = 0.4
IRIDIUM_SPS:         int   = 10     # samples/symbol in the realistic simulator

# ── Bursts per TX frame ────────────────────────────────────────────
#  Iridium superframe = 90 ms → 8 slots → at most 8 bursts per frame
NUM_BURSTS_PER_FRAME: int = 1       # 1 burst per transmission (loopback)

# ── Cyclic TX (continuous loop) ────────────────────────────────────
#  If True: the IQ buffer is transmitted in continuous loop (AD9363 DMA cyclic)
#  If False: one-shot transmission per call
CYCLIC_TX: bool = False

# ── Maximum connection retries ─────────────────────────────────────────
CONNECT_RETRIES: int = 3
CONNECT_TIMEOUT_S: float = 5.0
