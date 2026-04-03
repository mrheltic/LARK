# =============================================================================
#  LibreSDR Central Configuration
#  Zynq7020 + AD9363 (firmware PlutoSDR-compatible)
#
#  Questo file è l'unico punto dove modificare i parametri hardware del
#  LibreSDR. Tutti gli script TX importano da qui.
#
#  Modifica ALLOWED_FREQS prima di trasmettere: NON usare la banda Iridium
#  reale (1616–1626.5 MHz) senza licenza. Usa frequenze ISM autorizzate
#  o una connessione cablata TX→attenuatore→RX per i test di laboratorio.
# =============================================================================

# ── Connessione hardware ──────────────────────────────────────────────────────
#  URI pyadi-iio / libiio
#    Ethernet (default): "ip:192.168.2.1"
#    Connessione diretta USB-OTG: "usb:"
#    On-board (ARM core): "local:"
DEVICE_URI: str = "ip:192.168.2.1"

# ── Parametri TX ──────────────────────────────────────────────────────────────
#  Frequenza centrale TX [Hz]
#  NOTA LEGALE: trasmettere nella banda Iridium (1616–1626.5 MHz) senza
#  licenza è illegale. Per test usare un cavo RF + attenuatore ≥ 30 dB
#  tra TX e RX, oppure usare una frequenza ISM autorizzata.
TX_FREQ_HZ: int = 1_615_937_500     # banda Iridium – SOLO cablo/schermato

# Canale Iridium per test cablati (= canale 8 dal BASE_FREQ di iridium-toolkit)
# Decommentare la riga seguente per usare la banda ISM 433 MHz:
# TX_FREQ_HZ = 433_920_000

# Frequenza di campionamento TX [Hz].
# Minimo pratico AD9363 via Ethernet: 1 MSPS (TCP overhead).
# Per USB-OTG è possibile usare fino a 61.44 MSPS.
TX_SAMPLE_RATE: int = 1_000_000     # 1 MSPS

# Larghezza di banda RF TX [Hz]
# Deve essere ≥ banda del segnale (250 kHz per Iridium-like a 25 ksps + RRC 0.4)
TX_RF_BW: int = 250_000             # 250 kHz

# Attenuazione TX [dB]   range: 0 (massima potenza) … 89.75 dB (min)
# Iniziare da -60 dB e aumentare con cautela!
TX_GAIN_ATTENUATION_DB: float = 60.0   # 60 dB di attenuazione

# ── Parametri RX (per loopback e check_device) ───────────────────────────────
RX_FREQ_HZ:      int   = TX_FREQ_HZ
RX_SAMPLE_RATE:  int   = TX_SAMPLE_RATE
RX_RF_BW:        int   = TX_RF_BW
RX_GAIN_MODE:    str   = "slow_attack"   # "manual", "slow_attack", "fast_attack"
RX_GAIN_DB:      float = 30.0

# ── Parametri burst Iridium-like ─────────────────────────────────────────────
#  Questi valori DEVONO coincidere con shared.iridium e con quelli usati
#  dal ricevitore KrakenSDR.
#  Non modificarli qui: importarli da shared.iridium direttamente se serve.
IRIDIUM_SYMBOL_RATE: int   = 25_000
IRIDIUM_RRC_BETA:    float = 0.4
IRIDIUM_SPS:         int   = 10     # samples/symbol nel simulatore realistico

# ── Numero di burst per frame TX ─────────────────────────────────────────────
#  Superframe Iridium = 90 ms → 8 slot → al massimo 8 burst per frame
NUM_BURSTS_PER_FRAME: int = 1       # 1 burst per trasmissione (loopback)

# ── Ciclo continuo (cyclic TX) ────────────────────────────────────────────────
#  Se True: il buffer IQ viene trasmesso in loop continuo (AD9363 DMA cyclic)
#  Se False: trasmissione one-shot per ogni chiamata
CYCLIC_TX: bool = False

# ── Numero massimo di tentativi di connessione ────────────────────────────────
CONNECT_RETRIES: int = 3
CONNECT_TIMEOUT_S: float = 5.0
