/*
 * LARK — Iridium-like burst TX @ 868.1 MHz
 * =========================================
 * SX1272/SX1276 in FSK packet mode @ 25 kbps, fdev = 3 125 Hz
 *
 * Compatibilità pipeline KrakenSDR/LARK:
 *   ✓  detect_and_extract_burst()       potenza burst rilevabile
 *   ✓  _cfo_from_preamble()             tono a +3 125 Hz = +Rs/8
 *   ✓  narrowband_filter_burst()        segnale narrowband < 10 kHz
 *   ✓  MUSIC / Capon / Bartlett         segnale coerente su 5 canali
 *   ✗  validate_burst_uw()              FSK ≠ DQPSK → vedi nota LARK
 *
 * NOTA LARK: in _playback_analysis.py impostare
 *     UW_SCORE_MIN = 0.0     (riga ~74)
 * oppure lanciare con:
 *     python3 _playback_analysis.py ... --skip-uw   (se aggiungi il flag)
 *
 * Struttura burst (33 byte = 264 bit @ 25 kbps = 10.56 ms):
 *
 *   Byte  Contenuto     Bit      Effetto RF in FSK binario
 *   ----  ----------    -------  -----------------------------------------
 *   [0]   guard_pre     0x55     alternating → transitorio smooth ±fdev
 *   [1…8] preamble      0xFF×8   tutti 1 → CW a f_c + fdev = f_c + 3 125 Hz
 *                                ← _cfo_from_preamble() trova esattamente qui
 *   [9]   UW byte 1     0xEB     pattern fisso (non DQPSK standard)
 *   [10]  UW byte 2     0xE2     ← UW gate va disabilitato (vedi nota LARK)
 *   [11…32] payload     0xFF×22  CW coerente → covarianza DoA corretta
 *
 * Il ricevitore KrakenSDR è centrato a 868.000 MHz; TX a 868.100 MHz
 * (+100 kHz offset) evita il notch DC del SDR.
 * Pilot tone assoluto: 868.100 + 0.003125 = 868.103125 MHz.
 *
 * Hardware: Arduino + SX1272/SX1276
 *           (HopeRF RFM92W / RFM95W, Dragino LoRa Shield, …)
 * Libreria: https://github.com/CongducPham/LowCostLoRaGw  (SX1272.h)
 * Header  : sx1276Regs-Fsk.h + sx1276Regs-LoRa.h (dalla stessa libreria)
 *
 * ETSI 868 MHz: duty cycle ≤ 1 %.
 *   Test con cavo + attenuatore: BURST_PERIOD_MS 90   (≈ 11.7 % — solo bench)
 *   Test in aria:                BURST_PERIOD_MS 1100 (≈  0.96 % — conforme)
 */

#include <SPI.h>
#include "SX1272.h"
#include "sx1276Regs-Fsk.h"
#include "sx1276Regs-LoRa.h"

// ── Wrappers registro (identici all'esempio CW originale) ────────────────────
#define SX1276Write(reg, val)  sx1272.writeRegister((reg), (val))
#define SX1276Read(reg)        sx1272.readRegister(reg)

// ── Parametri RF ─────────────────────────────────────────────────────────────
#define XTAL_FREQ       32000000UL       // Hz — oscillatore SX127x
#define FREQ_STEP_F     61.03515625f     // Hz/LSB = Fxo / 2^19

// Frequenza TX: 868.100 MHz
// KrakenSDR centrato a 868.000 MHz → burst a +100 kHz dal DC
#define TX_FREQ_HZ      868100000UL

// Bit rate FSK = Symbol rate Iridium (25 ksps)
#define FSK_BITRATE_BPS 25000UL

// Fdev = Rs/8 = 3 125 Hz = esatta posizione del pilot tone Iridium
// Bit tutti-1 (0xFF) → frequenza f_c + fdev = f_c + 3 125 Hz
#define FSK_FDEV_HZ     3125UL

// ── Selezione PA ─────────────────────────────────────────────────────────────
// Decommentare SOLO la riga corrispondente al tuo modulo:
#define PABOOST                     // RFM92W / RFM95W / NiceRF1276 / inAir9B
// #undef  PABOOST                  // moduli con linea RFO

#ifdef PABOOST
  #define TX_POWER_DBM  10          // 14 dBm max ETSI — partire basso!
#else
  #define TX_POWER_DBM   7
#endif

// ── Temporizzazione ──────────────────────────────────────────────────────────
// Duty cycle burst: 10.56 ms / BURST_PERIOD_MS
// ETSI 868 MHz limite 1 % → periodo minimo ~1 060 ms per sicurezza.
//
//  Modalità              Periodo     Duty    Uso
//  ------------------    --------    -----   -----------------------------------
#define BURST_PERIOD_MS  1100        //  0.96%   test in aria (ETSI-safe)
// #define BURST_PERIOD_MS   90      // 11.7%   solo test in cavo con attenuatore

// ── Payload burst ─────────────────────────────────────────────────────────────
// 33 byte = 264 bit @ 25 kbps = 10.56 ms
// Mappatura simboli Iridium → byte FSK: 1 bite = 8 bit = 8 "simboli" a 25 kbps.
static const uint8_t BURST_PAYLOAD[] = {
  /* guard_pre  (8 simboli) */ 0x55,
  /* preamble   (64 simboli = 8 byte 0xFF) */
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
  /* UW approx  (12 s ≈ 2 byte) — non è DQPSK, UW gate va disabilitato */
  0xEB, 0xE2,
  /* payload    (177 bit ≈ 22 byte 0xFF — CW coerente sui 5 canali) */
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
};
#define BURST_LEN  ((uint8_t)(sizeof(BURST_PAYLOAD)))  // 33

// ── Prototipo ─────────────────────────────────────────────────────────────────
static void configFSK();
static void transmitBurst();

// ── configFSK() ───────────────────────────────────────────────────────────────
static void configFSK()
{
  // Step 1: Sleep — unico stato in cui si può cambiare LongRangeMode
  SX1276Write(REG_OPMODE,
    (SX1276Read(REG_OPMODE) & RF_OPMODE_MASK) | RF_OPMODE_SLEEP);
  delay(2);

  // Step 2: Seleziona FSK (clear bit LongRangeMode)
  SX1276Write(REG_OPMODE,
    (SX1276Read(REG_OPMODE) & RFLR_OPMODE_LONGRANGEMODE_MASK)
    | RFLR_OPMODE_LONGRANGEMODE_OFF);
  delay(2);

  // Step 3: Standby per la configurazione
  SX1276Write(REG_OPMODE,
    (SX1276Read(REG_OPMODE) & RF_OPMODE_MASK) | RF_OPMODE_STANDBY);
  delay(2);

  // Step 4: Frequenza TX = 868.100 MHz
  uint32_t frf = (uint32_t)((double)TX_FREQ_HZ / (double)FREQ_STEP_F);
  SX1276Write(REG_FRFMSB, (uint8_t)(frf >> 16));
  SX1276Write(REG_FRFMID, (uint8_t)(frf >>  8));
  SX1276Write(REG_FRFLSB, (uint8_t)(frf      ));

  // Step 5: Bit rate = 25 000 bps  →  BR[15:0] = Fxo / BR = 1280 = 0x0500
  uint16_t br = (uint16_t)((double)XTAL_FREQ / (double)FSK_BITRATE_BPS);
  SX1276Write(REG_BITRATEMSB, (uint8_t)(br >> 8));   // 0x05
  SX1276Write(REG_BITRATELSB, (uint8_t)(br     ));   // 0x00

  // Step 6: Fdev = 3 125 Hz  →  Fdev[13:0] = Fdev / FREQ_STEP ≈ 51 = 0x0033
  uint16_t fd = (uint16_t)((double)FSK_FDEV_HZ / (double)FREQ_STEP_F);
  SX1276Write(REG_FDEVMSB, (uint8_t)(fd >> 8));      // 0x00
  SX1276Write(REG_FDEVLSB, (uint8_t)(fd     ));      // 0x33

  // Step 7: Potenza PA (via libreria, dopo aver settato il flag PA_BOOST/RFO)
#ifdef PABOOST
  sx1272._needPABOOST = true;
#else
  sx1272._needPABOOST = false;
#endif
  sx1272.setPowerDBM((uint8_t)TX_POWER_DBM);

  // Step 8: Hardware preamble = 0 byte
  //         (gestiamo noi il guard_pre dentro il payload)
  SX1276Write(REG_PREAMBLEMSB, 0x00);
  SX1276Write(REG_PREAMBLELSB, 0x00);

  // Step 9: Sync word disabilitato (SyncOn = 0)
  SX1276Write(REG_SYNCCONFIG, 0x00);

  // Step 10: Pacchetto a lunghezza fissa, no CRC, no Manchester, no addr
  SX1276Write(REG_PACKETCONFIG1, 0x00);

  // Step 11: Packet mode (non continuous CW)
  SX1276Write(REG_PACKETCONFIG2,
    (SX1276Read(REG_PACKETCONFIG2) & RF_PACKETCONFIG2_DATAMODE_MASK)
    | RF_PACKETCONFIG2_DATAMODE_PACKET);

  // Step 12: Payload length = BURST_LEN byte (fixed packet)
  SX1276Write(REG_PAYLOADLENGTH, BURST_LEN);

  // Step 13: DIO mapping — DIO0 = TxDone in FSK packet mode
  SX1276Write(REG_DIOMAPPING1, 0x00);
  SX1276Write(REG_DIOMAPPING2, 0x30);
}

// ── transmitBurst() ───────────────────────────────────────────────────────────
static void transmitBurst()
{
  // Standby → si può scrivere nel FIFO solo in Standby/Sleep
  SX1276Write(REG_OPMODE,
    (SX1276Read(REG_OPMODE) & RF_OPMODE_MASK) | RF_OPMODE_STANDBY);
  delayMicroseconds(300);

  // Carica il payload nel FIFO (33 byte < 64 byte FIFO capacity)
  for (uint8_t i = 0; i < BURST_LEN; i++) {
    SX1276Write(REG_FIFO, BURST_PAYLOAD[i]);
  }

  // Avvia TX — il chip trasmette il FIFO e poi asserisce TxDone
  SX1276Write(REG_OPMODE,
    (SX1276Read(REG_OPMODE) & RF_OPMODE_MASK) | RF_OPMODE_TRANSMITTER);

  // Attesa TxDone: REG_IRQFLAGS2 bit3 (0x08).  Timeout = 25 ms.
  // Durata burst teorica: 33×8/25000 = 10,56 ms → il timeout non scatta mai
  // in condizioni normali; è un guard contro hang in caso di errore HW.
  uint32_t t0 = millis();
  while (!(SX1276Read(REG_IRQFLAGS2) & RF_IRQFLAGS2_TXDONE)) {
    if ((millis() - t0) >= 25UL) {
      Serial.println(F("[WARN] TX timeout — riconfigurazione"));
      configFSK();    // ripristina stato del chip prima di riprovare
      return;
    }
  }

  // Torna in standby (PA spento, consumo minimo tra un burst e il prossimo)
  SX1276Write(REG_OPMODE,
    (SX1276Read(REG_OPMODE) & RF_OPMODE_MASK) | RF_OPMODE_STANDBY);
}

// ── setup() ───────────────────────────────────────────────────────────────────
void setup()
{
  Serial.begin(115200);
  while (!Serial && millis() < 2000);

  Serial.println(F("\n=== LARK Iridium-like burst TX @ 868.1 MHz ==="));
  Serial.print(F("  TX freq    : ")); Serial.print(TX_FREQ_HZ / 1e6f, 3);
  Serial.println(F(" MHz"));
  Serial.print(F("  Bit rate   : ")); Serial.print(FSK_BITRATE_BPS);
  Serial.println(F(" bps (= Iridium symbol rate)"));
  Serial.print(F("  Fdev       : ")); Serial.print(FSK_FDEV_HZ);
  Serial.println(F(" Hz = Rs/8  (pilot tone a +3125 Hz)"));
  Serial.print(F("  Burst      : ")); Serial.print(BURST_LEN);
  Serial.print(F(" byte = "));
  Serial.print((float)BURST_LEN * 8000.0f / (float)FSK_BITRATE_BPS, 2);
  Serial.println(F(" ms"));
  Serial.print(F("  Periodo    : ")); Serial.print(BURST_PERIOD_MS);
  Serial.print(F(" ms  (duty cycle = "));
  Serial.print(100.0f * BURST_LEN * 8.0f / (FSK_BITRATE_BPS * BURST_PERIOD_MS / 1000.0f), 1);
  Serial.println(F(" %)"));
  Serial.println(F("  LARK      : UW_SCORE_MIN = 0.0  in _playback_analysis.py"));

  // Accende e inizializza la libreria (LoRa mode 1 = solo per init interno)
  sx1272.ON();
  delay(100);
  sx1272.setMode(1);
  delay(50);

  // Override con FSK packet mode
  configFSK();

  Serial.println(F("Radio OK — TX avviato\n"));
}

// ── loop() ────────────────────────────────────────────────────────────────────
void loop()
{
  static uint32_t lastTx = 0;

  if ((millis() - lastTx) >= (uint32_t)BURST_PERIOD_MS) {
    lastTx = millis();
    transmitBurst();

    Serial.print(F("TX burst @ t="));
    Serial.print(lastTx);
    Serial.println(F(" ms"));
  }
}
