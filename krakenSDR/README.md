# KrakenSDR — Direction-of-Arrival su segnali reali: da zero a satellite 🛰️

> **Tutto il codice è Python puro. Zero GNU Radio. Zero segreti.**
>
> Questo documento racconta ogni passo del percorso: dall'hardware, al DSP,
> ai risultati reali sugli Iridium NEXT in orbita.

---

## Indice

1. [Cos'è KrakenSDR e perché cinque antenne](#1-cosè-krakensdr-e-perché-cinque-antenne)
2. [Architettura hardware](#2-architettura-hardware)
3. [La catena software: dal sample al bearing](#3-la-catena-software-dal-sample-al-bearing)
4. [Il cuore DSP: MUSIC su UCA](#4-il-cuore-dsp-music-su-uca)
5. [Rilevazione burst Iridium](#5-rilevazione-burst-iridium)
6. [Applicazioni: guida pratica](#6-applicazioni-guida-pratica)
7. [Sistema di configurazione](#7-sistema-di-configurazione)
8. [Calibrazione hardware](#8-calibrazione-hardware)
9. [Neural calibration: dall'algebra al neurone](#9-neural-calibration-dallalgebra-al-neurone)
10. [Risultati sperimentali](#10-risultati-sperimentali)
11. [Struttura del repository](#11-struttura-del-repository)
12. [Riferimenti](#12-riferimenti)

---

## 1. Cos'è KrakenSDR e perché cinque antenne

Il **KrakenSDR** è un ricevitore SDR a 5 canali coerenti basato su RTL-SDR.
"Coerenti" significa che tutti e cinque i canali condividono lo stesso
oscillatore locale di riferimento (TCXO) e vengono campionati in sincrono.
Questa proprietà — che un normale RTL-SDR non ha — è indispensabile per
misurare le **differenze di fase tra antenne** che codificano la direzione
di arrivo di un segnale.

### Perché la fase porta la direzione

Considera un'onda piana proveniente da azimuth `φ`, elevazione `θ`.
Il fronte d'onda raggiunge ciascuna antenna con un ritardo proporzionale
alla proiezione della posizione di quell'antenna sul vettore di propagazione:

```
τ_k = (p_k · û(φ,θ)) / c
```

dove `p_k` è la posizione fisica dell'antenna `k` e `c` è la velocità della luce.
Nel dominio della frequenza, questo ritardo si traduce in uno **sfasamento**:

```
Δφ_k = 2π f · τ_k = 2π/λ · (p_{k,E} · cosθ·sinφ  +  p_{k,N} · cosθ·cosφ)
```

Con 5 antenne otteniamo 4 differenze di fase indipendenti → 4 equazioni con
2 incognite (az, el) → il problema è sovradeterminato e risolvibile con
algoritmi di stima subspace come MUSIC.

### Perché l'UCA (Uniform Circular Array)?

L'array UCA a 5 elementi su corona circolare è la scelta ottimale per:
- **(360° di copertura)** senza ambiguità azimuth.
- **(Simmetria)** → guadagno uniforme in tutte le direzioni.
- **(Compattezza)** → il raggio `r = λ/2 / (2·sin(π/N))` ≈ 0.4253λ
  è l'unico parametro geometrico.

```
            ant0 (Nord)
           ↑
  ant4 ····●···· ant1
            |
           ant3   ant2
```

Il KrakenSDR fisico ha antenna 0 in posizione hardware; l'allineamento
geografico (Nord) si ottiene ruotando `ANT0_OFFSET_DEG` in config.

---

## 2. Architettura hardware

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRANSMITTER (opzionale — test / calibrazione)                              │
│                                                                             │
│  LibreSDR (AD9363) ─── tx_868_gui.py  →  segnale 868 MHz burst IRA         │
│  Arduino SX1276    ─── lark_burst_tx_868.ino  (stand-alone, no PC)         │
└─────────────────────────────────────────────────────────────────────────────┘
                          │  RF  (cable / aria)
                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  RECEIVER — KrakenSDR 5-ch                                                  │
│                                                                             │
│  5× RTL-SDR (coerenti)  →  USB  →  Raspberry Pi / PC                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Heimdall DAQ firmware                                               │   │
│  │  (daq_start_sm.sh)                                                   │   │
│  │                                                                      │   │
│  │  ADC sampling: 1.024 MHz                                             │   │
│  │  CPI (Coherent Processing Interval): 131 072 campioni ≈ 128 ms      │   │
│  │  Output: frames (5, 131 072) complex64                               │   │
│  │  Transport: TCP stream, porta 5000 (dati) + 5001 (controllo)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                          │  TCP 127.0.0.1:5000
                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  PYTHON STACK (krakenSDR/src)                                               │
│                                                                             │
│  hardware.KrakenIQSource  →  get_frame() → (5, N) complex64                │
│         ↓                                                                   │
│  core.burst.BurstDetector / core.iridium_doa_burst                         │
│         ↓                                                                   │
│  core.doa_uca_2d  (2D-MUSIC / 2D-Capon / 2D-Bartlett)   ← UCA 868 MHz     │
│  core.doa_algorithms_3d  (2D-MUSIC su cross-array)       ← Satellite L-band│
│         ↓                                                                   │
│  matplotlib GUI  (8 pannelli)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Header del frame Heimdall

Heimdall antepon un header di 1024 byte a ogni frame IQ.  `IQHeader.decode()`
legge campi cruciali come `rf_center_freq`, `sampling_freq`, `cpi_length`,
`iq_sync_flag` (sincronizzazione inter-canale) e `adc_overdrive_flags`
(sovraccarico ADC = guadagno troppo alto).

---

## 3. La catena software: dal sample al bearing

### 3.1 `KrakenIQSource` — il lettore di frame

```python
from hardware.kraken_iq_source import KrakenIQSource

src = KrakenIQSource(
    host="127.0.0.1",
    port=5000,
    ctrl_port=5001,
    num_channels=5,
    freq_hz=868e6,
    gain_db=30.0,
)
src.start()
frame = src.get_frame(timeout=2.0)  # ndarray (5, 131_072) complex64
```

Internamente:
1. Si connette a Heimdall con il protocollo `b"streaming"` + `b"IQDownload"`.
2. Per ogni request legge 1024 byte di header + `N × Nr × 8` byte di payload
   (complex64 = float32 reale + float32 immaginario per canale maggiore).
3. Il frame viene de-interleaved: 5 canali × N campioni.
4. Se `frame_type != DATA` (dummy, ramp, cal) o `iq_sync_flag == 0`, il frame
   viene scartato.

### 3.2 Estrazione del tono pilota (per segnali burst)

Iridium IRA: i 64 simboli di preambolo con dibit "00" ruotano di +π/4 ciascuno
→ sequenza costante → tono CW puro a `f_carrier + Fs/8 = f_carrier + 3125 Hz`.

```python
from core.doa_uca_2d import extract_pilot_tone, amplitude_normalize_channels

# X: (5, N_burst) — burst grezzo multi-canale
X_nb = extract_pilot_tone(X, sample_rate=1_024_000, tone_hz=3125.0, bw_hz=5000.0)
# X_nb: (5, N_pilot) — componente narrowband al tono pilota

X_nb = amplitude_normalize_channels(X_nb)
# Normalizza ogni canale a potenza unitaria — elimina imbalance di guadagno HW
```

`extract_pilot_tone()` filtra con una finestra FFT attorno alla frequenza del
tono, restituendo solo la componente spazialmente coerente. Il filtraggio
aumenta l'SNR effettivo di ~20 dB rigettando il rumore broadband e la
componente dati.

### 3.3 Matrice di covarianza istantanea

```python
R_inst = (X_nb @ X_nb.conj().T) / X_nb.shape[1]   # (5, 5) complex128
```

`R_inst[i,j]` = correlazione statistica tra l'antenna `i` e l'antenna `j`.
Gli elementi off-diagonal di `R_inst` codificano le **differenze di fase**
inter-antenna: `∠R_inst[0,k]` ≈ `Δφ_k` dal modello di steering.

### 3.4 EMA — Exponential Moving Average sulla covarianza

```python
from core.doa_uca_2d import CovarianceAccumulatorUca

acc = CovarianceAccumulatorUca(alpha=0.90)

# Per ogni burst valido:
R_ema = acc.update(X_nb)   # accumula solo campioni con preambolo valido
```

L'EMA con `α = 0.90` effettua una media temporale che:
- Riduce la varianza di stima di `R` (più campioni → meno rumore)
- Decorrela parzialmente sorgenti multipath coerenti (effetto forward-backward)
- Mantiene la memoria invertendo il peso: `R_new = α·R_old + (1−α)·X@X†/N`

**Nell'architettura corrente**, l'EMA viene aggiornata solo sui burst con
`papr_inst ≥ 12 dB` (preambolo confermato); la stima DoA usa però `R_inst`
per ogni burst, non `R_ema`, per evitare il "congelamento" dell'angolo.

### 3.5 PAPR istantaneo come rilevatore di preambolo

```python
spec2d = doa_music_uca_2d(X_nb, cfg, R_in=R_inst)
az, el, papr_inst = find_peak_uca_2d(spec2d, cfg)

is_preamble = (papr_inst >= 12.0)  # dB
```

Il **Peak-to-Average Power Ratio** dello spettro MUSIC 2D è un classificatore
naturale:

| Tipo di finestra       | `papr_inst` tipico | Ragione                         |
|------------------------|--------------------|---------------------------------|
| Preambolo IRA          | ≥ 14–34 dB         | Segnale rank-1, peak acuto      |
| Sezione dati / rumore  | ≤ 12 dB            | Covarianza quasi-isotropica     |

Soglia 12 dB: ≥ 2.5 dB di margine sopra il massimo DATA, accetta preamboli
con errori hardware di fase fino a ±20°.

### 3.6 Correzione hardware di fase

I cavi che collegano le antenne al KrakenSDR hanno lunghezze leggermente
diverse → offset di fase per canale che spostano il vettore di steering
rispetto al modello teorico → MUSIC fornisce angolo errato.

```python
from core.doa_algorithms import apply_phase_correction

# phase_offsets_deg: (5,) lista da config.CHANNEL_PHASE_OFFSETS_DEG
X_cal = apply_phase_correction(X_nb, phase_offsets_deg)
```

Internamente moltiplica ogni canale `k` per `exp(-j·Δφ_k)`:

```python
phasors = np.exp(-1j * np.deg2rad(offsets))   # (N_ant,)
return X * phasors[:, np.newaxis]              # broadcast su tutti i campioni
```

Dopo questa correzione le fasi di `R_cal[0,k]` corrispondono esattamente
alle fasi teoriche del vettore di steering → MUSIC dà il bearing corretto.

**Impatto sull'hardware reale** (simulazione con errori ±20°):

| Errori HW         | `papr_inst` preamble | Supera soglia 12 dB? |
|-------------------|----------------------|----------------------|
| ±0° (calibrato)   | 31–34 dB             | ✓ sempre             |
| ±10°              | 17–18 dB             | ✓ sempre             |
| ±15°              | 14.6–15.0 dB         | ✓ con margine 2.6 dB |
| ±20°              | 12.4 dB              | ✓ con margine 0.4 dB |
| ±45°              | 6.2 dB               | ✗ rifiutato          |

### 3.7 MUSIC 2D → spettro azimuth-elevazione

```python
from core.doa_uca_2d import UcaConfig, doa_music_uca_2d, find_peak_uca_2d

cfg = UcaConfig(
    n_ant=5,
    radius_lambda=0.4253,   # λ/2 spacing, 5 ant
    n_az=180,               # 2° step
    n_el=36,                # 5° step, 5°–90°
)

spec2d = doa_music_uca_2d(X_cal, cfg, R_in=R_inst)
# spec2d: (n_el, n_az) float → spettro in dB, picco=0, pavimento=−40 dB

az_deg, el_deg, papr_db = find_peak_uca_2d(spec2d, cfg)
# az_deg: azimuth stimato [0°, 360°)
# el_deg: elevazione stimata [5°, 90°]
# papr_db: qualità della stima
```

### 3.8 EMA circolare sull'angolo (smoothing responsivo)

```python
# State: az_phasor = np.exp(0j)  (inizialmente Nord)
AZ_SMOOTH_ALPHA = 0.50

az_ph = np.exp(1j * np.deg2rad(az_inst))
S.az_phasor = AZ_SMOOTH_ALPHA * S.az_phasor + (1.0 - AZ_SMOOTH_ALPHA) * az_ph
az_smooth = float(np.degrees(np.angle(S.az_phasor)) % 360.0)
```

Lavorare su **fasori complessi** invece di angoli scalari evita l'artefatto
di wrap 0°/360°: la media di 359° e 1° dà correttamente 0°, non 180°.

Con `α = 0.50`, la costante di tempo è ≈ 2 burst validi (≈ 0.76 s a
11 Hz burst rate), quindi l'angolo si aggiorna in meno di un secondo.

---

## 4. Il cuore DSP: MUSIC su UCA

### 4.1 Teoria di MUSIC

Dati `N` campioni, `K` antenne, `D` sorgenti (`D < K`):

```
R = E[x x†]  =  A·S·A†  +  σ²·I
```

La decomposizione agli autovalori di `R`:

```
R = E · Λ · E†
```

Split in *signal subspace* (D autovettori più grandi) e
*noise subspace* (K−D autovettori più piccoli):

```
E = [E_s | E_n]    con   E_n^† · A·steering = 0 nel caso ideale
```

Pseudospettro MUSIC:

```
P_MUSIC(φ,θ) = 1 / ||E_n† · a(φ,θ)||²
```

Il picco di `P_MUSIC` si trova dove il vettore di steering `a(φ,θ)` è
ortogonale al noise subspace → corrispondente alla direzione della sorgente.

### 4.2 Matrice di steering pre-calcolata

`UcaConfig.get_steering_matrix()` calcola la matrice `(N_ant, N_el × N_az)`
una volta sola e la mette in cache per parametri fissati.
Il calcolo usa broadcasting NumPy per velocità:

```python
# Posizioni antenne in lunghezze d'onda: (5, 2) [Est, Nord]
pos = cfg.positions           # φ_k = 2π·k/N clockwise from North

# Griglia (n_el, n_az)
AZ, EL = np.meshgrid(az_rad, el_rad)
u_e = np.cos(EL) * np.sin(AZ)   # cosθ·sinφ
u_n = np.cos(EL) * np.cos(AZ)   # cosθ·cosφ

# Ritardi di fase: (5, n_el*n_az)
tau = 2π · (pos[:,0:1] · u_e.ravel() + pos[:,1:2] · u_n.ravel())
A   = exp(j·tau)   # matrice di steering
```

### 4.3 Esempio numerico completo

```python
import numpy as np
from krakenSDR.src.core.doa_uca_2d import (
    UcaConfig, doa_music_uca_2d, find_peak_uca_2d, CovarianceAccumulatorUca
)

# --- Parametri ---
cfg = UcaConfig(n_ant=5, radius_lambda=0.4253, n_az=180, n_el=36)
FS = 1_024_000      # Heimdall sample rate
N  = 2621           # pilot window samples

# --- Segnale sintetico da az=45°, el=20°, SNR=8 dB ---
rng  = np.random.default_rng(0)
az0, el0 = np.deg2rad(45), np.deg2rad(20)
pos  = cfg.positions
tau  = 2*np.pi * (pos[:,0]*np.cos(el0)*np.sin(az0)
                + pos[:,1]*np.cos(el0)*np.cos(az0))
tone = np.exp(2j*np.pi * 3125 / FS * np.arange(N))
snr  = 10**(8/10)
X    = rng.standard_normal((5, N)) + 1j*rng.standard_normal((5, N))
X   += np.sqrt(snr) * np.exp(1j*tau[:,None]) * tone[None,:]

# --- DoA ---
R    = X @ X.conj().T / N           # covarianza istantanea
spec = doa_music_uca_2d(X, cfg, R_in=R)
az_est, el_est, papr = find_peak_uca_2d(spec, cfg)

print(f"Vero:  az={45:.1f}°  el={20:.1f}°")
print(f"Stima: az={az_est:.1f}°  el={el_est:.1f}°  PAPR={papr:.1f} dB")
# Output: Vero: az=45.0°  el=20.0°
#         Stima: az=44.9°  el=20.2°  PAPR=28.4 dB
```

### 4.4 Algoritmi disponibili su UCA 2D

| Algoritmo             | Funzione                    | Caratteristiche                              |
|-----------------------|-----------------------------|----------------------------------------------|
| 2D-MUSIC              | `doa_music_uca_2d()`        | Migliore risoluzione, richiede stima D        |
| 2D-Capon (MVDR)       | `doa_capon_uca_2d()`        | Più robusto su D incerto, meno nitido        |
| 2D-Bartlett (CBF)     | `doa_bartlett_uca_2d()`     | Fallback: nessun null subspace, sempre stabile|
| Root-MUSIC UCA        | `doa_root_music_uca_2d()`   | Precisione sub-griglia, senza scan 2D        |
| Unitary-ESPRIT UCA    | `doa_unitary_esprit_uca_2d()`| Rotational invariance, velocissimo          |
| MFBA-MUSIC            | `doa_mfba_music_uca_2d()`   | Forward-backward UCA, de-correlazione multipath |

### 4.5 Eigenvalue spread come indicatore di qualità

```python
from core.doa_uca_2d import eigenvalue_spread_uca_db

eig = eigenvalue_spread_uca_db(R)
# eig[0]: spread massimo (dB) — indica presenza di segnale
# Soglia tipica EIG_SPREAD_MIN_DB = 2.5 dB
# Con segnale: eig[0] >> 2.5 dB   (sorgente rank-1 → λ_max >> λ_min)
# Solo rumore: eig[0] ≈ 0–2 dB   (R ≈ σ²·I → tutti autovalori uguali)
```

---

## 5. Rilevazione burst Iridium

### 5.1 Struttura del burst IRA

```
Durata attiva burst: 261 simboli / 25 000 sps = 10.44 ms
Periodo slot TDMA:   281 simboli / 25 000 sps = 11.25 ms
Super-frame period:  90 ms (un burst ogni 90 ms dallo stesso satellite)

Struttura burst:
  [Preambolo 64 sym][Unique Word 12 sym][Dati 165 sym][Tail 4 sym]
       ↑
     Tono puro a carrier + 3125 Hz
     (tutti dibit 00 → rotazione fissa +π/4 / simbolo)
```

```
   Spettro FFT durante il preambolo (idealizzato):
   
   |
   |              ████
   |    baseline  ████  ← picco a +3125 Hz
   |   ──────────░░░░──────────────────
   +--+--+--+--+--+--+--+--+--+--+-→  freq
   -40k                +3.125k   +40k
```

### 5.2 Pipeline di rilevazione (`core.iridium_doa_burst`)

```python
from core.iridium_doa_burst import (
    detect_and_extract_burst,    # energy gate su singolo canale
    detect_and_extract_all_bursts, # rileva tutti i burst nel frame
    compensate_doppler,          # rimozione offset Doppler
    compute_single_shot_covariance,  # R da finestra burst
    validate_burst_uw,               # verifica Unique Word
    narrowband_filter_burst,         # BPF su finestra burst
)

# frame: (5, 131_072) complex64 da Heimdall
bursts = detect_and_extract_all_bursts(frame, threshold_db=8.0)

for burst, doppler_hz in bursts:
    # burst: (5, ~10_690) — finestra burst multi-canale
    compensated = compensate_doppler(burst, sample_rate=1_024_000,
                                     doppler_hz=doppler_hz)
    R = compute_single_shot_covariance(compensated)
    # R: (5,5) — matrice di covarianza solo sulla finestra burst
```

### 5.3 Perché NON usare la covarianza EMA su burst

L'EMA standard media N frame consecutivi di 128 ms ciascuno.
Un burst Iridium dura solo ≈10.44 ms su 90 ms di super-frame:
il frame Heimdall di 128 ms **contiene al massimo un burst** (probabilmente).

```
EMA su 5 frame: 128ms × 5 = 640 ms di campioni
  →  di cui ~10.44 ms contengono il burst (segnale)
  →  i restanti 630 ms sono rumore puro
  →  R_EMA ≈ 98.4% rumore → MUSIC flat → nessun bearing
```

La soluzione: estrarre la finestra burst esatta (BurstDetector) e calcolare
`R_inst` **solo su quella finestra**. L'SNR effettivo passa da ≈−8 dB a ≈+8 dB.

### 5.4 Compensazione Doppler

Un satellite Iridium a 780 km di quota si muove a ~7.5 km/s.
Visto dal suolo, il Doppler massimo è:

```
Δf_max = (v_sat / c) × f_carrier = (7500 / 3e8) × 1626e6 ≈ 40.7 kHz
```

Senza compensazione, il downmix parziale lascia la sorgente a frequenza
variabile nel corso del pass (da +40 kHz a −40 kHz in ~10 min) → le fasi
inter-canale cambiano tra un burst e l'altro → R_EMA media fasori non coerenti.

`compensate_doppler()` applica lo stesso fasore `exp(-j·2πΔf·t/Fs)` a
tutti e 5 i canali simultaneamente, preservando le **differenze di fase**
inter-canale (invarianti alla frequenza portante):

```
Δφ_k_dopo = (Δφ_k_prima + Δφ_CFO) − (Δφ_0_prima + Δφ_CFO) = Δφ_k_prima ✓
```

---

## 6. Applicazioni: guida pratica

### 6.1 `apps/iridium/iridium_detector.py` — Rilevatore burst single-antenna

Il punto di partenza: verifica che la catena hardware veda i burst Iridium
prima di dispiegare il multi-antenna DoA.

```bash
cd krakenSDR/src
python3 apps/iridium/iridium_detector.py
python3 apps/iridium/iridium_detector.py --freq 1626270000 --gain 20
python3 apps/iridium/iridium_detector.py --snr 5 --papr 3.5
```

**Layout GUI (5 pannelli):**
```
┌─────────────────────────────────┬────────────────────────────┐
│  A: Spettro IQ  (±40 kHz)       │  B: Curva S Doppler        │
│     banda Iridium in evidenza   │     scatter SNR-colored     │
├─────────────────────────────────┴────────────────────────────┤
│  C: Spettrogram waterfall — zoomed ±64 kHz (newest = top)    │
├─────────────────────────────────┬────────────────────────────┤
│  D: SNR + PAPR per burst        │  E: Timeline burst binaria  │
└─────────────────────────────────┴────────────────────────────┘
```

**Cosa guardare:**
- Pannello C: strisce verticali luminose ogni ~11 ms → burst TDMA
- Pannello B: curva S (da +40 kHz a −40 kHz) → traccia Doppler di un pass satellite
- Pannello D: SNR ≥ 15 dB con PAPR ≥ 5 dB → burst decodificabili

---

### 6.2 `apps/iridium/iridium_live.py` — Decoder RAW verso iridium-parser

```bash
# Pipe diretto in iridium-parser per messaggi decodificati
python3 apps/iridium/iridium_live.py | \
    python3 ../../external/iridium-toolkit/iridium-parser.py -p

# Cattura RAW per analisi offline
python3 apps/iridium/iridium_live.py --freq 1626270000 > session.bits
```

`iridium_live.py` usa `BurstPipeline` (detector + IridiumDemod) per:
1. Rilevare burst → `BurstResult`
2. Demodulare DQPSK → bit string
3. Emettere linee `RAW:` su stdout (compatibili iridium-toolkit)

**Formato RAW:**
```
RAW: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
     |—freq_info—||—timestamp—||————————raw bits (hexa)————————————|
```

---

### 6.3 `apps/iridium/iridium_offline.py` — Playback di registrazioni WAV

```bash
python3 apps/iridium/iridium_offline.py pass_20260415.wav
python3 apps/iridium/iridium_offline.py pass.wav --demod --raw-out bursts.txt
python3 apps/iridium/iridium_offline.py pass.cf32 --fs 2048000 --freq 1626270000
```

Stessa GUI di `iridium_detector.py` ma su file registrati.
Supporta WAV (SC16 da SDR++), CF32 (float-32 interleaved), U8 (rtl_sdr).

**Workflow tipico:**
1. Uscire con laptop + RTL-SDR singolo, registrare con SDR++ in WAV
2. Tornare al PC, analizzare con `iridium_offline.py`
3. Usare lo scrub slider per navigare il pass satellite

---

### 6.4 `apps/iridium/iridium_analyzer.py` — Anatomia del burst

```bash
python3 apps/iridium/iridium_analyzer.py               # auto-detect WAV
python3 apps/iridium/iridium_analyzer.py pass.wav --snr 10
```

Interattivo: naviga i singoli burst e visualizza:
- Inviluppo IQ con regioni preambolo/UW/dati colorate
- Costellazione DQPSK (IQ scatter)
- Anatomia bit colorata
- Metadati (SNR, Doppler, decodifica protocollo)

---

### 6.5 `apps/iridium/iridium_pilot.py` — Estrazione tono pilota

```bash
python3 apps/iridium/iridium_pilot.py                  # auto-detect WAV
python3 apps/iridium/iridium_pilot.py --all            # tutti i burst
python3 apps/iridium/iridium_pilot.py --out report.txt
python3 apps/iridium/iridium_pilot.py --limit 300      # primi 300 s
```

Analisi dettagliata per ogni burst del tono pilota (i 64 simboli del preambolo):
- Fase e ampiezza per ogni simbolo pilota
- SNR del pilota vs rumore
- Residuo di frequenza dopo downmix
- Utile per verificare la qualità DoA **prima** di eseguire MUSIC

---

### 6.6 `apps/doa_test_868/doa_test_868_burst.py` — DoA burst a 868 MHz

Il **principale banco di test** per validare il pipeline DoA con hardware
controllato: LibreSDR (TX) + KrakenSDR (RX) in laboratorio o in campo.

```bash
cd krakenSDR/src
python3 apps/doa_test_868/doa_test_868_burst.py            # hardware reale
python3 apps/doa_test_868/doa_test_868_burst.py --demo     # sintetico @ 45°
python3 apps/doa_test_868/doa_test_868_burst.py --algo capon
python3 apps/doa_test_868/doa_test_868_burst.py --out-dir /tmp/doa_burst

# Auto-calibrazione con TX a posizione nota (es. 20° Nord)
python3 apps/doa_test_868/doa_test_868_burst.py --calibrate 20.0
```

**Layout GUI (8 pannelli in 2 righe):**
```
┌─────────────┬──────────────┬────────────────┬────────────────┐
│  Compass    │  Heatmap 2D  │  Az/El history │  Eigenvalues   │
│  MUSIC polar│  (az × el)   │  rolling window│  spread dB     │
├─────────────┼──────────────┼────────────────┼────────────────┤
│  Coherence  │  PAPR + SNR  │  IQ FFT ch0    │  Phase diffs   │
│  matrix|ρ|  │  history     │  pilot tone    │  ch1..4 vs ch0 │
└─────────────┴──────────────┴────────────────┴────────────────┘
```

**Pipeline interna per ogni frame Heimdall:**

```
Frame (5, 131072)
   │
   ├─ Ricerca burst (energy gate + PAPR check)
   │     se no burst → skip
   │
   ├─ Estrai finestra burst (5, ~10690)
   │
   ├─ extract_pilot_tone(X, 1_024_000, 3125.0, 5000.0)
   │     → X_nb (5, ~2621) — solo il tono pilota
   │
   ├─ amplitude_normalize_channels(X_nb)
   │
   ├─ apply_phase_correction(X_nb, CHANNEL_PHASE_OFFSETS_DEG)
   │     → X_cal — steering vectors corretti per HW
   │
   ├─ R_inst = X_cal @ X_cal† / N
   │
   ├─ eigenvalue_spread(R_inst)
   │     < EIG_SPREAD_MIN_DB → fast-reject (skip MUSIC)
   │
   ├─ doa_music_uca_2d(X_cal, cfg, R_in=R_inst)
   │     → spec2d (n_el, n_az)
   │
   ├─ find_peak_uca_2d(spec2d, cfg)
   │     → az_inst, el_inst, papr_inst
   │
   ├─ is_preamble = papr_inst >= 12.0 dB
   │     if False → stato minimo, continue
   │
   ├─ acc.update(X_cal)   — EMA solo su burst validi
   │
   └─ Circular EMA: az_phasor = α·az_phasor + (1-α)·exp(j·az_inst)
         az_smooth = angle(az_phasor) mod 360°
```

**Parametri chiave (config.py):**

```python
RADIUS_LAMBDA           = 0.4253   # geometria UCA 5 ant @ 868 MHz
COV_ALPHA               = 0.90     # EMA covarianza
AZ_SMOOTH_ALPHA         = 0.50     # EMA angolo (≈2 burst validi)
EIG_SPREAD_MIN_DB       = 2.5      # fast-reject Level 1
CHANNEL_PHASE_OFFSETS_DEG = [0,0,0,0,0]  # da --calibrate
```

---

### 6.7 `apps/doa_test_868/doa_test_868_realtime.py` — DoA CW a 868 MHz

Modalità CW (continuous wave): nessun gate burst, EMA standard su tutti i frame.
Utile con beacon LoRa o segnali continui.

```bash
python3 apps/doa_test_868/doa_test_868_realtime.py
python3 apps/doa_test_868/doa_test_868_realtime.py --algo capon
python3 apps/doa_test_868/doa_test_868_realtime.py --offset 45  # calibrazione manuale
```

---

### 6.8 `apps/doa/doa_runner.py` — DoA ISM generico (ULA/UCA 1D)

Il backend generico per ISM 433/868 MHz con array qualsiasi (ULA o UCA 1D).
Algoritmi: MUSIC, Root-MUSIC, Capon, ML, ESPRIT. Display 8 pannelli.

```bash
python3 apps/doa/doa_runner.py
LARK_PROFILE=outdoor_5ant python3 apps/doa/doa_runner.py
NUM_CHANNELS=5 ARRAY_TYPE=UCA CENTER_FREQ=868 python3 apps/doa/doa_runner.py
```

**Differenza rispetto a `doa_test_868_burst.py`:**
- `doa_runner.py`: EMA su tutti i frame (CW), algoritmi 1D, array ULA/UCA
- `doa_test_868_burst.py`: burst-gated, 2D (az+el), solo UCA 5-ant

---

### 6.9 `apps/space/space_doa_realtime.py` — DoA 3D satelliti (realtime)

Il pezzo più avanzato: DoA 3D (azimuth + elevazione) su burst Iridium reali
da satelliti in orbita. Usa un **cross-array** ("+") invece dell'UCA.

```bash
python3 apps/space/space_doa_realtime.py
python3 apps/space/space_doa_realtime.py --freq 1626.27 --gain 20 --algo music
python3 apps/space/space_doa_realtime.py --algo capon --mode cw --n_az 90 --n_el 27
```

**Layout GUI (8 pannelli = 2 righe × 4 colonne):**
```
Row 0: [Sky plot polar]  [Heatmap rect]  [Az+El history]  [Eigenvalues]
Row 1: [Coherence matrix] [PAPR+SNR hist] [IQ FFT ch0]    [Phase stability]
```

**Cross-array geometry:**
```
              ant1 (North)
                |
  ant3 ──── ant0 ──── ant2
  (West)    (ctr)     (East)
                |
              ant4 (South)
```
Con 4 bracci ortogonali di lunghezza `D_LAMBDA`:
- Braccio E-W risolve la componente Est del vettore di arrivo → contribuisce ad azimuth
- Braccio N-S risolve la componente Nord → contribuisce ad azimuth + elevazione
- Standard `D_LAMBDA = 0.5λ` → no grating lobes (sicuro ovunque)
- Sparse `D_LAMBDA = 1.0λ` → 2× risoluzione, grating lobes solo sotto l'orizzonte

---

### 6.10 `apps/space/space_collector.py` — Raccolta dati multiantenna

Raccoglie burst raw su tutti e 5 i canali senza fare DoA → salva su disco per
analisi offline. Ideale per sessioni di campo dove si vuole raccogliere dati
e analizzarli dopo (magari con la calibrazione TLE).

```bash
python3 apps/space/space_collector.py
python3 apps/space/space_collector.py --freq 1626.27 --gain 20 --limit 200
python3 apps/space/space_collector.py --out /mnt/ssd/captures --threshold 8
```

**Formato output NPZ:**
```
kraken_space_raw_YYYYMMDD_HHMMSS.npz
  bursts     : (N, 5, N_burst)  complex64   burst IQ per antenna
  timestamps : (N,)             float64     [ms] dall'inizio sessione
  doppler_hz : (N,)             float64     stima Doppler per burst
  snr_db     : (N,)             float32     SNR burst [dB]

kraken_space_raw_YYYYMMDD_HHMMSS.json
  freq_hz, sample_rate_hz, gain_db, n_antennas,
  burst_threshold_db, n_bursts, duration_s, timestamp_utc
```

---

### 6.11 `apps/space/space_doa_playback.py` — Playback offline 3D satellite

```bash
python3 apps/space/space_doa_playback.py
python3 apps/space/space_doa_playback.py /path/to/capture.npz
python3 apps/space/space_doa_playback.py capture.npz --algo capon --n_az 90
```

Stessa GUI di `space_doa_realtime.py` ma su file NPZ registrati.
Controlli: ⏸/▶, ⏮ rewind, frame slider, ×0.25/×0.5/×1/×2/×4 velocità.

---

### 6.12 `apps/space/iridium_pass_predict.py` — Confronto DoA vs TLE

Lo strumento di **validazione**: confronta il bearing misurato con la posizione
predetta dei satelliti Iridium NEXT calcolata dai TLE NORAD.

```bash
python3 apps/space/iridium_pass_predict.py                        # file picker
python3 apps/space/iridium_pass_predict.py recording.npz
python3 apps/space/iridium_pass_predict.py recording.npz --lat 45.07 --lon 7.69
python3 apps/space/iridium_pass_predict.py --predict-now           # mostra visibili ora
python3 apps/space/iridium_pass_predict.py --predict-window 2      # prossime 2h
python3 apps/space/iridium_pass_predict.py recording.npz --export out.json
```

**Display (4 pannelli):**
```
[0] Sky plot: DoA misurata (scatter) + traiettorie TLE (archi)
[1] Istogramma errore azimuth: DoA_az − TLE_az [°]
[2] Istogramma errore elevazione: DoA_el − TLE_el [°]
[3] Scatter Doppler osservato vs predetto
```

**Come funziona il matching TLE:**
1. Scarica catalogo Iridium NEXT da CelesTrak (cache 24h)
2. Per ogni burst accettato, calcola la posizione di tutti i satelliti visibili
   al timestamp del burst usando SGP4 propagation
3. Identifica il satellite più vicino in bearing az/el
4. `GT_MATCH_THRESHOLD_KHZ = 8 kHz` su Doppler per confermare il match
5. Riporta: az_err, el_err, Doppler_err per burst

---

### 6.13 `apps/space/calibration_run.py` — Calibrazione NN (MLP)

Pipeline di neural calibration per trasformare le differenze di fase
inter-antenna in bearing (az, el) senza dipendere dal modello geometrico.

```bash
# Raccolta dataset da registrazione
python3 apps/space/calibration_run.py collect recording.npz

# Training
python3 apps/space/calibration_run.py train calibration_dataset.npz

# Pipeline completa (collect + train + validate)
python3 apps/space/calibration_run.py auto recording.npz --epochs 300

# Validazione modello esistente
python3 apps/space/calibration_run.py validate model.npz --data new_rec.npz

# Confronto MLP vs MUSIC
python3 apps/space/calibration_run.py compare model.npz recording.npz
```

Vedere [§9. Neural calibration](#9-neural-calibration-dallalgebra-al-neurone).

---

## 7. Sistema di configurazione

Il sistema ha 3 livelli:

```
config_hw.py              ← Livello 1: costanti hardware invarianti
    │                        (indirizzo Heimdall, ADC rate, N canali)
    │ re-exported from
    ▼
apps/<group>/config.py    ← Livello 2: configurazione per scenario
    │                        (frequenza, guadagno, geometria array, algoritmo)
    │ override via
    ▼
profiles.py               ← Livello 3: preset named per scenari comuni
```

### Utilizzo

```python
# In ogni app script:
import config as C    # risolve a apps/<group>/config.py (locale)

C.FREQ_HZ            # frequenza carrier
C.GAIN_DB            # guadagno IF
C.N_ANTENNAS         # numero canali
C.HEIMDALL_HOST      # indirizzo Heimdall
C.RADIUS_LAMBDA      # raggio UCA in λ
C.COV_ALPHA          # alpha EMA covarianza
```

### Profili

```bash
# Preset al volo senza modificare config.py
LARK_PROFILE=iridium_1626         python3 apps/space/space_doa_realtime.py
LARK_PROFILE=outdoor_5ant_fast    python3 apps/doa/doa_runner.py
LARK_PROFILE=outdoor_5ant_precision python3 apps/doa/doa_runner.py
```

Profili disponibili in `src/profiles.py`:

| Profilo                   | Scenario                      | α_cov | α_ang | Algoritmo   |
|---------------------------|-------------------------------|-------|-------|-------------|
| `outdoor_5ant`            | Esterno, sorgente fissa       | 0.95  | 0.80  | MUSIC       |
| `outdoor_5ant_fast`       | Sorgente mobile               | 0.70  | 0.40  | MUSIC       |
| `outdoor_5ant_precision`  | Alta precisione, statica      | 0.96  | 0.90  | Root-MUSIC  |
| `indoor_5ant`             | Interno, multipath pesante    | 0.80  | 0.60  | MUSIC       |
| `iridium_1626`            | Satellite Iridium 1626.27 MHz | 0.90  | –     | 2D-MUSIC    |

---

## 8. Calibrazione hardware

### 8.1 Perché è necessaria

I cavi coassiali del KrakenSDR hanno piccole differenze di lunghezza fisica
(anche solo 1 cm a 868 MHz = 360° × 1cm / 34.5cm λ ≈ 10° di offset).
Effetto misurabile sulle prestazioni MUSIC:

```
Errore HW    papr_inst (preamble)   Δbeaming   Ring Rejection
±0°          31–34 dB              ≈0°         < 0.5°
±10°         17–18 dB              ≈3°         < 2°
±15°         14.6 dB               ≈8°         < 5°
±20°         12.4 dB               ≈15°        ≈ limite accettabilità
±45°         6.2 dB  (rifiutato)   –           –
```

### 8.2 Auto-calibrazione con `--calibrate`

```bash
# TX a posizione nota (es. 20° az, 30 m di distanza)
python3 apps/doa_test_868/doa_test_868_burst.py --calibrate 20.0
```

**Algoritmo:**
1. Accumula R_EMA su burst validi finché `acc.is_warm`
2. Estrae l'autovettore dominante `v = eigh(R_ema)[-1]` ≈ risposta array reale
3. Calcola le fasi geometriche teoriche per `az=known_az`, `el=0°`:
   `τ_k = 2π(p_{k,E}·sin(az) + p_{k,N}·cos(az))`
4. `hw_offset_k = ∠v_k − τ_k` (normalizzato su ch0)
5. Aggiorna automaticamente `CHANNEL_PHASE_OFFSETS_DEG` in `config.py`

### 8.3 Calibrazione NN sulla cross-array (satelliti)

Per la cross-array a L-band:
```bash
# Raccolta dataset (usa TLE per ground-truth az/el)
python3 apps/space/calibration_run.py collect krakenSDR/recordings/*.npz

# Training MLP (14 features → az/el)
python3 apps/space/calibration_run.py train krakenSDR/calibration/calibration_dataset.npz

# Il modello viene salvato come:
# krakenSDR/calibration/calib_model_latest.npz
```

---

## 9. Neural calibration: dall'algebra al neurone

### 9.1 Feature extraction (`core/calibration_features.py`)

Per ogni burst con ground-truth TLE, si estraggono 14 feature:

```
F = [cos(Δφ₁), sin(Δφ₁),     ← fase inter-antenna ch0→1 (2)
     cos(Δφ₂), sin(Δφ₂),     ← fase inter-antenna ch0→2 (2)
     cos(Δφ₃), sin(Δφ₃),     ← fase inter-antenna ch0→3 (2)
     cos(Δφ₄), sin(Δφ₄),     ← fase inter-antenna ch0→4 (2)
     |ρ₀₁|, |ρ₀₂|, |ρ₀₃|, |ρ₀₄|,  ← coerenze (4)
     λ_spread_dB,             ← eigenvalue spread (1)
     doppler_khz]             ← Doppler normalizzato (1)
                              ←                   Totale: 14
```

La codifica (cos, sin) per le fasi evita la discontinuità a ±180°.

### 9.2 Architettura MLP (`core/calibration_model.py`)

```
Input:  14
Hidden: 64  (ReLU)
Hidden: 64  (ReLU)
Hidden: 32  (ReLU)
Output:  3  → (cos_az, sin_az, el_norm)
```

Output: `az = atan2(sin_az, cos_az)`, `el = el_norm × 90°`

**Loss function:**
```python
L = w_az · (1 − cos(az_pred − az_true))   # invariante a wrap 0°/360°
  + w_el · ((el_pred − el_true) / 90)²    # MSE normalizzata
```

**Zero dipendenze ML:** solo NumPy + Scipy. Pesi salvati come `.npz` (no pickle).

---

## 10. Risultati sperimentali

### 10.1 Test 868 MHz con LibreSDR (aprile 2026)

| Sessione        | SNR medio | has_signal | Az mean ± std | Note                           |
|-----------------|-----------|------------|---------------|--------------------------------|
| 20260428_182743 | 4 dB      | 7%         | 184.7° ± 104° | Prima del fix calibrazione     |
| Post-fix        | –         | ~26%       | 21.5° ± 16°   | Dopo R_inst + soglia 12 dB     |
| Simulazione     | 4 dB SNR  | 26.0%      | 21.5° ± 0.7°  | Convergenza in ≤10 burst       |

**Root causes del 7% iniziale (identificate dall'analisi del recording):**
1. Nessuna calibrazione HW → errori fase ±15° → papr_inst ≈ 14.6 dB < soglia 15 dB
2. DoA su R_EMA congelato (stesso risultato tra update) → angolo fisso 76.1% del tempo
3. Pausa massima tra update: 151 secondi (!)

**Fix applicati:**
1. `apply_phase_correction()` con `CHANNEL_PHASE_OFFSETS_DEG`
2. DoA su `R_inst` per-burst → angolo aggiornato a ~11 Hz
3. Soglia `papr_inst` da 15 → 12 dB (margin analysis)
4. EMA circolare sull'angolo (α=0.5) → convergenza in ≈2 burst (0.76 s)

### 10.2 Test Iridium satellite (aprile 2026)

| Metrica              | Risultato           |
|----------------------|---------------------|
| Burst rate rilevati  | ~11 Hz (su ring-ch) |
| GT match rate        | 65–78% dei burst    |
| Az residual σ        | ±12–18°             |
| El residual σ        | ±8–12°              |
| Curva S Doppler      | −38 kHz → +40 kHz   |
| Duration pass tipico | 8–12 min            |

---

## 11. Struttura del repository

```
krakenSDR/
├── README.md                  ← questo file
├── arduino/                   ← firmware Arduino (TX standalone)
│   └── lark_burst_tx_868/
│       └── lark_burst_tx_868.ino
├── calibration/               ← dati di calibrazione (generati a runtime)
│   ├── calib_model_latest.npz
│   ├── calibration_dataset.npz
│   ├── phase_offsets_latest.json
│   └── *.json / *.npz / *.png  (storico sessioni)
├── data/                      ← registrazioni NPZ (generati a runtime)
│   ├── doa_868/
│   └── *.npz
└── src/
    ├── config_hw.py           ← costanti hardware (Heimdall addr, N_ch, Fs)
    ├── config.py              ← re-export config_hw per compatibilità
    ├── profiles.py            ← preset named (outdoor_5ant, iridium_1626…)
    │
    ├── hardware/              ← driver IQ source
    │   ├── kraken_iq_source.py  TCP client Heimdall (protocollo IQDownload)
    │   ├── heimdall_manager.py  Process manager per daq_start_sm.sh
    │   └── file_iq_source.py    Sorgente IQ da file (WAV, CF32, U8)
    │
    ├── core/                  ← algoritmi DSP puri (no I/O, testabili)
    │   ├── doa_algorithms.py     Array 1D: MUSIC / Root-MUSIC / Capon / ML / ESPRIT
    │   ├── doa_uca_2d.py         Array 2D UCA: algoritmi avanzati (incluso Root-MUSIC UCA,
    │   │                          ESPRIT UCA, MFBA, enhanced preprocessing)
    │   ├── doa_algorithms_3d.py  Cross-array 2D: MUSIC / Capon / IAA
    │   ├── burst.py              Rilevatore burst Iridium (FFT, PAPR, PassTracker)
    │   ├── burst_pipeline.py     Connette BurstDetector + IridiumDemod
    │   ├── iridium_demod.py      Decoder DQPSK → bit string → RAW: line
    │   ├── iridium_doa_burst.py  Estrazione burst multi-ch, Doppler, covarianza
    │   ├── calibration_features.py  Feature extraction (14 features / burst)
    │   ├── calibration_model.py     MLP NumPy-puro (14→64→64→32→3)
    │   └── signal_quality.py        Metriche qualità (power balance, PAPR, coherence)
    │
    ├── apps/                  ← applicazioni eseguibili
    │   ├── doa/               ← DoA ISM generico (433/868 MHz, ULA/UCA 1D)
    │   │   ├── config.py
    │   │   ├── doa_runner.py    Entry point puro Python (8 pannelli)
    │   │   ├── doa_realtime.py  GRC wrapper (GNU Radio opzionale)
    │   │   └── doa_playback.py  Playback da NPZ
    │   │
    │   ├── doa_test_868/      ← DoA 2D UCA @ 868 MHz (burst IRA)
    │   │   ├── config.py
    │   │   ├── doa_test_868_burst.py    Burst-gated 2D DoA (principale)
    │   │   └── doa_test_868_realtime.py CW 2D DoA (segnali continui)
    │   │
    │   ├── iridium/           ← Ricezione / decode Iridium L-band
    │   │   ├── config.py
    │   │   ├── iridium_detector.py  Single-antenna burst detector (GUI)
    │   │   ├── iridium_live.py      Multi-antenna → RAW: lines su stdout
    │   │   ├── iridium_offline.py   Playback WAV/CF32/U8
    │   │   ├── iridium_analyzer.py  Anatomia burst interattiva
    │   │   └── iridium_pilot.py     Analisi tono pilota
    │   │
    │   └── space/             ← DoA 3D satelliti (cross-array)
    │       ├── config.py
    │       ├── space_doa_realtime.py  DoA real-time satellite
    │       ├── space_doa_playback.py  Playback offline 3D DoA
    │       ├── space_collector.py     Raccolta burst raw multi-ch
    │       ├── iridium_pass_predict.py Confronto DoA vs TLE NORAD
    │       └── calibration_run.py     Pipeline calibrazione NN
    │
    ├── scripts/               ← strumenti diagnostici CLI
    │   ├── start_heimdall.sh         Avvio Heimdall DAQ
    │   ├── test_gt_matching.py       Test matching Doppler vs TLE (60 s)
    │   ├── deep_multichannel_analysis.py Analisi profonda multi-FDMA (tutti i canali)
    │   └── playback_analysis.py      Analisi MUSIC batch su NPZ archivio
    │
    ├── config/                ← file INI per Heimdall DAQ
    │   ├── daq_chain_config.ini      Iridium 1626.270 MHz
    │   └── daq_chain_config_868.ini  ISM 868 MHz
    │
    ├── ui/                    ← widget GUI riutilizzabili
    │   ├── dialogs.py         Startup dialog Iridium
    │   └── theme.py           Palette colori oscura
    │
    └── tests/                 ← test suite (pytest, 118 test)
        ├── test_doa.py         DoA 1D: sweep 360°, multipath, SNR, geometria
        ├── test_doa_3d.py      DoA 2D cross-array
        ├── test_calibration.py Pipeline calibrazione NN
        └── test_satellite.py   Observer, TLE, qualità canali
```

### Cosa fa ogni modulo core in una riga

| Modulo                   | Cosa fa                                                  |
|--------------------------|----------------------------------------------------------|
| `doa_algorithms.py`      | MUSIC/Capon/ESPRIT/ML per UCA 1D (`ArrayConfig`)         |
| `doa_uca_2d.py`          | MUSIC/Capon/Root-MUSIC/ESPRIT/MFBA per UCA 2D (`UcaConfig`) |
| `doa_algorithms_3d.py`   | MUSIC/Capon/IAA per cross-array 2D (`CrossArrayConfig`)  |
| `burst.py`               | Rilevatore FFT burst Iridium (`BurstDetector`)           |
| `burst_pipeline.py`      | Detector + Demod → `RAW:` line (`BurstPipeline`)         |
| `iridium_demod.py`       | Resample → downmix → LPF → RRC → sync → DQPSK → bits    |
| `iridium_doa_burst.py`   | Estrazione burst multi-ch, Doppler, R single-shot        |
| `calibration_features.py`| 14 feature da R (fase, coerenza, eigspread, Doppler)     |
| `calibration_model.py`   | MLP NumPy puro: train / predict / serialize              |
| `signal_quality.py`      | Power balance, PAPR, coherence per health-check          |

---

## 12. Riferimenti

### Algoritmi DoA

- **Schmidt R.O.** (1986). *Multiple emitter location and signal parameter
  estimation.* IEEE Trans. Antennas Propagat. **34**(3), 276–280. — MUSIC
- **Barabell A.** (1983). *Improving the resolution performance of eigenstructure-based
  direction-finding algorithms.* ICASSP. — Root-MUSIC
- **Capon J.** (1969). *High-resolution frequency-wavenumber spectrum analysis.*
  Proc. IEEE **57**(8), 1408–1418. — MVDR/Capon
- **Pillai S.U. & Kwon B.H.** (1989). *Forward/backward spatial smoothing
  techniques for coherent signal identification.* IEEE Trans. ASSP **37**(4). — FBA
- **Mathews C.P. & Zoltowski M.D.** (1994). *Eigenstructure techniques for
  2D angle estimation with uniform circular arrays.* IEEE Trans. SP **42**(9). — UCA phase modes
- **Van Trees H.L.** (2002). *Optimum Array Processing.* Wiley, §9.2. — UCA steering
- **Wax M. & Kailath T.** (1985). *Detection of signals by information theoretic
  criteria.* IEEE Trans. ASSP **33**(2). — AIC/MDL

### Iridium Protocol

- **ITU-R M.1031** — Iridium radio interface characteristics
- **ETSI EN 300 461** — Iridium TDMA frame structure
- **iridium-toolkit** (muccc) — https://github.com/muccc/iridium-toolkit

### Hardware

- **KrakenSDR** — https://www.krakenrf.com
- **Heimdall DAQ firmware** — https://github.com/krakenrf/heimdall_daq_fw
- **LibreSDR / AD9363** — https://wiki.analog.com/resources/tools-software/linux-drivers/iio-transceiver/ad9361

### Propagazione orbitale

- **Vallado D.A.** (2013). *Fundamentals of Astrodynamics and Applications.* 4th ed. — SGP4
- **CelesTrak** — https://celestrak.org (TLE catalogue)

---

*Ultima revisione: Aprile 2026 — commit 599ca6f*
