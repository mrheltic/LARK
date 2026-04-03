# KrakenSDR DoA — Documentazione Tecnica

> **Stima della Direzione di Arrivo (DoA) per KrakenSDR con array 3/5 antenne UCA/ULA.**  
> Algoritmi: MUSIC / Root-MUSIC / Capon / ML / ESPRIT  
> Decorrelazione: FBA, Toeplitz, FB+Toeplitz  
> Covarianza temporale EMA · Display real-time 8 pannelli · Registrazione e playback

---

## Indice

1. [Struttura del progetto](#1-struttura-del-progetto)
2. [Hardware: cosa manda il KrakenSDR](#2-hardware-cosa-manda-il-krakensdr)
   - 2.1 [Architettura hardware](#21-architettura-hardware)
   - 2.2 [Il pacchetto TCP: header + payload](#22-il-pacchetto-tcp-header--payload)
   - 2.3 [Struttura dell header 1024 byte](#23-struttura-dellheader-1024-byte)
   - 2.4 [Payload IQ: complex64 multiplex per canale](#24-payload-iq-complex64-multiplex-per-canale)
   - 2.5 [Protocollo di handshake](#25-protocollo-di-handshake)
   - 2.6 [Canale di controllo porta 5001](#26-canale-di-controllo-porta-5001)
   - 2.7 [Come si estrae la fase inter-canale](#27-come-si-estrae-la-fase-inter-canale)
   - 2.8 [Sincronizzazione di fase: cosa fa Heimdall](#28-sincronizzazione-di-fase-cosa-fa-heimdall)
3. [Architettura del sistema e flusso dei dati](#3-architettura-del-sistema-e-flusso-dei-dati)
4. [Modello di segnale](#4-modello-di-segnale)
5. [Geometria dell array e vettori di sterzatura](#5-geometria-dellarray-e-vettori-di-sterzatura)
6. [Matrice di covarianza campionaria](#6-matrice-di-covarianza-campionaria)
7. [Decorrelazione della covarianza](#7-decorrelazione-della-covarianza)
8. [Da UCA a VULA: Phase Mode Excitation](#8-da-uca-a-vula-phase-mode-excitation)
9. [Algoritmi DoA](#9-algoritmi-doa)
10. [Covarianza temporale EMA](#10-covarianza-temporale-ema)
11. [Smoothing angolare basato su fasori](#11-smoothing-angolare-basato-su-fasori)
12. [Metriche diagnostiche](#12-metriche-diagnostiche)
13. [Calibrazione di fase hardware](#13-calibrazione-di-fase-hardware)
14. [Registrazione e playback](#14-registrazione-e-playback)
15. [Risultati sperimentali](#15-risultati-sperimentali)
16. [Riferimento alla configurazione](#16-riferimento-alla-configurazione)
17. [Quick Start](#17-quick-start)
18. [Bibliografia](#18-bibliografia)

---

## 1. Struttura del progetto

```
workspace/pysdr_doa/
|-- config.py               <- Configurazione centrale — modifica solo questo file
|-- kraken_iq_source.py     <- Connessione TCP a Heimdall (protocollo IQ)
|-- doa_algorithms.py       <- Libreria algoritmi DoA (MUSIC, Capon, ML, ESPRIT...)
|-- pysdr_doa_realtime.py   <- Applicazione real-time: display 8 pannelli
|-- pysdr_doa_playback.py   <- Playback delle registrazioni .npz
|-- README.md               <- Questo file
```

Tutti i parametri configurabili si trovano esclusivamente in `config.py`.  
Le applicazioni leggono da `config.py` all avvio; non modificare gli altri file
a meno che non si voglia cambiare la logica di elaborazione.

---

## 2. Hardware: cosa manda il KrakenSDR

### 2.1 Architettura hardware

Il KrakenSDR e un campionatore coerente a 5 canali basato su RTL-SDR.
Un oscillatore locale (LO) comune e distribuito a tutti e 5 i ricevitori,
garantendo coerenza di fase tra i canali — requisito fondamentale per la DoA.

```
     Antenna 0         Antenna 1         Antenna 2
         |                 |                 |
    [RTL-SDR 0]       [RTL-SDR 1]       [RTL-SDR 2]
         |                 |                 |
         +--------+--------+
                  |
          [LO comune]        <- coerenza di fase
                  |
          [USB 3.0 hub]
                  |
                 PC
                  |
           [Heimdall DAQ]
          porta 5000 (IQ)
          porta 5001 (ctrl)
```

Il firmware **Heimdall** gira in background (nel progetto: Docker container),
riceve i campioni dai tre/cinque RTL-SDR, esegue la sincronizzazione di fase
via rumore impulsivo interno, e li espone via TCP porta 5000 in frames binari.

### 2.2 Il pacchetto TCP: header + payload

Ogni frame ricevuto da Heimdall ha la struttura:

```
+---------------------------+----------------------------------------------+
|  HEADER  (1024 byte)      |  PAYLOAD  (variabile)                        |
|  struct-packed, binario   |  Nr * K campioni complex64 (interleaved IQ)  |
+---------------------------+----------------------------------------------+
```

- `Nr` = numero di antenne attive (es. 3)
- `K`  = numero di campioni per canale per frame (es. 1024, 2048...)
- Dimensione payload = Nr * K * 2 * 4 byte  (due float32: I e Q)

### 2.3 Struttura dell header (1024 byte)

Il formato e definito dalla struct C (vedi `kraken_iq_source.py`, classe `IQHeader`).
Il formato di unpack Python e:

```python
fmt = "II16sIIIQQQIQIIQIII" + "I"*32 + "IIII" + "I"*192 + "I"
```

| Offset (B) | Tipo C      | Campo                  | Descrizione                                          |
|-----------|-------------|------------------------|------------------------------------------------------|
| 0         | uint32      | `sync_word`            | Parola di sincronizzazione = 0x2bf7b95a              |
| 4         | uint32      | `frame_type`           | 0=DATA, 1=DUMMY, 2=RAMP, 3=CAL, 4=TRIGW             |
| 8         | char[16]    | `hardware_id`          | Stringa hardware ID (es. "KrakenSDR")                |
| 24        | uint32      | `unit_id`              | Identificativo unita                                 |
| 28        | uint32      | `active_ant_chs`       | Numero canali attivi (es. 3 o 5)                     |
| 32        | uint32      | `ioo_type`             | Tipo I/O                                             |
| 36        | uint64      | `rf_center_freq`       | Frequenza portante [Hz] (es. 865210000)              |
| 44        | uint64      | `adc_sampling_freq`    | Frequenza di campionamento ADC (es. 2048000)         |
| 52        | uint64      | `sampling_freq`        | Frequenza di campionamento effettiva (es. 1024000)   |
| 60        | uint32      | `cpi_length`           | Lunghezza CPI (campioni per canale per frame)        |
| 64        | uint64      | `time_stamp`           | Timestamp Unix in microsecondi                       |
| 72        | uint32      | `daq_block_index`      | Indice progressivo blocco DAQ                        |
| 76        | uint32      | `cpi_index`            | Indice CPI (Coherent Processing Interval)            |
| 80        | uint64      | `ext_integration_cntr` | Contatore integrazione esteso                        |
| 88        | uint32      | `data_type`            | Tipo dati payload (0 = complex64)                    |
| 92        | uint32      | `sample_bit_depth`     | Profondita di bit (es. 32 = float32)                 |
| 96        | uint32      | `adc_overdrive_flags`  | Bit-mask canali in overdrive ADC                     |
| 100       | uint32[32]  | `if_gains`             | Gain IF per canale x10 (es. 300 = 30.0 dB)          |
| 228       | uint32      | `delay_sync_flag`      | 1 = ritardo risolto (sincronizzazione completata)    |
| 232       | uint32      | `iq_sync_flag`         | 1 = IQ sincronizzato (fase stabile)                  |
| 236       | uint32      | `sync_state`           | Stato macchina di sincronizzazione                   |
| 240       | uint32      | `noise_source_state`   | 1 = rumore interno attivo (calibrazione in corso)    |
| 244       | uint32[192] | `reserved`             | Riservato per uso futuro                             |
| ...       | uint32      | `header_version`       | Versione del formato header                          |

**Note importanti:**

- Il campo `if_gains` contiene i valori moltiplicati per 10 (interi):
  es. `300` corrisponde a 30.0 dB.  
- `payload_bytes` si calcola come:
  `active_ant_chs * cpi_length * 2 * (sample_bit_depth // 8)`

### 2.4 Payload IQ: complex64 multiplex per canale

Il payload contiene i campioni IQ di tutti i canali in formato **interleaved**
ordinati per riga (row-major / C order):

```
Layout in memoria:
  [I_ch0_s0, Q_ch0_s0, I_ch0_s1, Q_ch0_s1, ..., I_ch0_sK-1, Q_ch0_sK-1,
   I_ch1_s0, Q_ch1_s0, ...,
   I_ch2_s0, Q_ch2_s0, ...]

Reshape in Python:
  X = np.frombuffer(payload_bytes, dtype=np.complex64)
      .reshape(Nr, K)
```

Dove:
- `Nr` = `active_ant_chs`
- `K`  = `cpi_length`
- Ogni campione e un numero complesso `I + jQ` (float32 + float32 = 8 byte)

Il significato fisico di I e Q:
- **I (In-phase)**: parte reale del segnale in banda base
- **Q (Quadrature)**: parte immaginaria (sfasata di 90 gradi rispetto a I)
- Insieme formano il **segnale analitico**: `s(t) = I(t) + j*Q(t)`

La fase del segnale al tempo t e:

```
phi(t) = arctan2(Q(t), I(t)) = angle(s(t))
```

### 2.5 Protocollo di handshake

Il protocollo e ASCII a lunghezza fissa su TCP porta 5000:

```
CLIENT                          HEIMDALL (porta 5000)
  |                                   |
  |-- b"streaming" (9 byte) --------> |  apertura sessione
  |                                   |
  |<-- [HEADER 1024B + PAYLOAD] ------| bootstrap frame automatico
  |                                   |    (Heimdall lo invia subito)
  |                                   |
  |          loop principale:         |
  |-- b"IQDownload" (10 byte) ------> |  richiesta frame
  |<-- [HEADER 1024B + PAYLOAD] ------| risposta con dati IQ
  |-- b"IQDownload" ----------------> |
  |<-- [HEADER 1024B + PAYLOAD] ------|
  ...
  |-- b"q" (1 byte) ---------------> |  chiusura connessione
```

In Python (`kraken_iq_source.py`):

```python
# 1. apertura sessione
sock.sendall(b"streaming")
# 2. bootstrap (no IQDownload)
bootstrap = _recv_frame(request=False)
# 3. loop
while True:
    sock.sendall(b"IQDownload")
    raw_hdr = _recv_exact(sock, 1024)
    hdr.decode(raw_hdr)
    payload = _recv_exact(sock, hdr.payload_bytes)
    X = np.frombuffer(payload, dtype=np.complex64).reshape(Nr, K)
```

### 2.6 Canale di controllo (porta 5001)

Il canale di controllo usa messaggi di 128 byte su TCP porta 5001.

| Comando | Formato (128 B)                         | Descrizione                          |
|---------|-----------------------------------------|--------------------------------------|
| INIT    | `b"INIT" + bytes(124)`                  | Inizializza la sessione di controllo |
| FREQ    | `b"FREQ" + uint64(freq_hz) + pad`       | Imposta frequenza portante [Hz]      |
| GAIN    | `b"GAIN" + uint32[Nr](gain*10) + pad`   | Imposta gain IF (x10, interi)        |
| EXIT    | `b"EXIT" + bytes(124)`                  | Chiude la connessione                |

Esempio di invio frequenza in Python:

```python
import struct
freq_cmd = b"FREQ" + struct.pack("<Q", int(freq_hz)) + bytes(116)
ctrl_sock.send(freq_cmd)  # 4 + 8 + 116 = 128 byte
```

Esempio di invio gain (3 canali a 30 dB = 300 nel campo int):

```python
gain_int = [300, 300, 300]  # 30.0 dB * 10
gain_cmd = b"GAIN" + struct.pack("<" + "I"*3, *gain_int) + bytes(116)
ctrl_sock.send(gain_cmd)
```

### 2.7 Come si estrae la fase inter-canale

Questa e la parte fondamentale per la DoA: misurare la differenza di fase
tra il segnale ricevuto su ciascuna antenna rispetto all antenna di riferimento.

**Metodo diretto (temporale):**

```python
# X ha forma (Nr, K) con Nr=3, K=numero campioni
# Prodotto di correlazione incrociata istantanea
corr = X[0, :] * np.conj(X[k, :])   # per k=1,2,...,Nr-1
phi_k = np.angle(np.mean(corr))       # fase media inter-canale [rad]
```

**Metodo via matrice di covarianza (usato nel codice):**

La matrice di covarianza campionaria e:

```
R = X @ X^H / K    (forma: Nr x Nr)
```

Non si tratta di una moltiplicazione scalare: e il prodotto matriciale
tra X (Nr x K) e X coniugato-trasposto (K x Nr).

L elemento `R[0, k]` (riga 0, colonna k) contiene:

```
R[0,k] = (1/K) * sum_{i=0}^{K-1} x_0[i] * x_k*[i]
         = E[ x_0(t) * x_k*(t) ]   (correlazione incrociata media)
```

La fase di questo numero complesso e esattamente la differenza di fase
stimata tra il canale 0 e il canale k:

```python
R = X @ X.conj().T / K
phi_k = np.angle(R[0, k])    # sfasamento ch0->chk in radianti
```

Il valore `-pi <= phi_k <= pi` codifica il ritardo di fase del segnale
dovuto alla diversa posizione fisica delle antenne nell array.

**Conversione fase -> angolo di arrivo (DoA):**

Per una ULA con passo `d = lambda/2`:

```
phi = 2*pi * d/lambda * sin(theta)
=> theta = arcsin(phi / pi)    [per d = lambda/2]
```

Per una UCA il calcolo e piu complesso (vedi sezione 8).

**Estrazione diretta da R nel Pannello H** (display diagnostico):

```python
R = X @ X.conj().T / K
for k in range(1, Nr):
    phase_k = np.angle(R[0, k])   # in radianti
    # visualizzato nel pannello H come "ang R[0,k]"
```

Questo pannello mostra l evoluzione temporale della differenza di fase
tra ciascuna coppia di antenne. Una fase stabile indica un segnale coerente
e ben calibrato.

### 2.8 Sincronizzazione di fase: cosa fa Heimdall

Prima di poter usare le differenze di fase per la DoA, i canali devono essere
sincronizzati. Heimdall esegue la calibrazione automaticamente:

1. **Calibrazione delay:** Heimdall inietta un segnale impulsivo (rumore bianco
   interno) con `noise_source_state=1`. Misura il ritardo di gruppo tra i canali
   e lo compensa digitalmente. Al termine: `delay_sync_flag=1`.

2. **Calibrazione IQ:** Corregge la differenza di fase residua dopo la 
   compensazione delay. Al termine: `iq_sync_flag=1`.

3. **Stato macchina:** `sync_state` indica lo stato corrente:
   - 0: non sincronizzato
   - 1: delay sincronizzato
   - 2: IQ sincronizzato (pronto per DoA)
   - 3+: stati intermedi/errore

Il codice controlla questi flag in `pysdr_doa_realtime.py`:

```python
hdr = src.last_header
if hdr.sync_state < 2:
    # skip DoA, segnale non ancora sincronizzato
    continue
```

---

## 3. Architettura del sistema e flusso dei dati

```
KrakenSDR HW
    |
    | USB
    v
Heimdall DAQ (Docker)
port 5000 (IQ)  port 5001 (ctrl)
    |
    | TCP
    v
KrakenIQSource (kraken_iq_source.py)
  - handshake
  - recv thread in background
  - queue frames (max 4)
    |
    v
pysdr_doa_realtime.py
  |
  +-- get_frame() -> X (Nr, K) complex64
  |
  +-- apply_phase_correction(X, PHASE_OFFSETS_DEG)
  |
  +-- amplitude_normalize(X)  [se AMPLITUDE_NORMALIZE=True]
  |
  +-- squelch check: power(X) >= SQUELCH_THRESHOLD_DB
  |
  +-- R_new = covariance(X)
  +-- R_ema = alpha * R_ema + (1-alpha) * R_new   [EMA temporale]
  |
  +-- uca_to_vula(X, r_lambda)  [se UCA + non-Off decorrelation]
  +-- apply_decorrelation(R, method)
  |
  +-- doa_ALGORITHM(X, cfg, R_in=R_ema)
       |
       +-> theta_scan, spectrum -> stima angolo
  |
  +-- angle_smooth(theta_est)  [EMA su fasori]
  |
  +-- update() Matplotlib - 8 pannelli
```

**EMA (Exponential Moving Average) doppia:**

| Livello     | Parametro      | Oggetto        | Scopo                               |
|-------------|----------------|----------------|-------------------------------------|
| Covarianza  | COV_ALPHA=0.92 | matrice R NxN  | memoria temporale, anti-multipath   |
| Angolo      | ANGLE_SMOOTH=0.75 | un phasore  | riduce jitter visuale angolo finale |

---

## 4. Modello di segnale

Si considera un array di Nr antenne che ricevono D sorgenti in campo lontano.
Il vettore di osservazione al tempo t e:

```
x(t) = A(theta) * s(t) + n(t)
```

Dove:
- `x(t)` : vettore (Nr x 1) dei campioni IQ ricevuti alle antenne
- `A(theta)` : matrice (Nr x D) dei vettori di sterzatura per ogni sorgente
- `s(t)` : vettore (D x 1) dei segnali sorgente
- `n(t)` : rumore termico bianco (Nr x 1), sigma^2 * I

La matrice di covarianza teorica e:

```
R = E[x x^H] = A * P_s * A^H + sigma^2 * I
```

Con P_s = E[s s^H] matrice di potenza dei segnali.

**Sottospazio segnale/rumore:**

L eigendecomposizione di R produce:
- D autovettori con autovalori grandi: sottospazio segnale E_s
- (Nr-D) autovettori con autovalori sigma^2: sottospazio rumore E_n

La DoA sfrutta l ortogonalita tra a(theta) e E_n:

```
a^H(theta_true) * E_n = 0
```

---

## 5. Geometria dell array e vettori di sterzatura

### ULA (Uniform Linear Array)

Elementi allineati con spaziatura d = D_LAMBDA * lambda:

```
a_ULA[k](theta) = exp(j * 2*pi * d * k * sin(theta))   k = 0,...,Nr-1
```

### UCA (Uniform Circular Array)

Elementi distribuiti su cerchio di raggio r = RADIUS_LAMBDA * lambda:

```
phi_k = 2*pi*k/Nr                         (angolo fisico antenna k)
a_UCA[k](theta) = exp(j * 2*pi * r * cos(theta - phi_k))
```

La UCA ha copertura angolare completa 360 gradi, la ULA solo +/-90 rispetto
all asse di allineamento.

**Configurazione attiva:**

| Parametro        | Valore        | Significato                               |
|------------------|---------------|-------------------------------------------|
| N_ANTENNAS       | 3             | tre elementi per configurazione indoor    |
| GEOMETRY         | UCA           | array circolare, copertura 360 gradi      |
| RADIUS_LAMBDA    | 0.289         | raggio = 0.289 lambda                     |
| FREQ_HZ          | 865.21e6      | portante 865.21 MHz (banda ISM 868 MHz)   |
| lambda           | ~34.6 cm      | 3e8 / 865.21e6                            |
| raggio fisico    | ~10.0 cm      | 0.289 * 34.6 cm                           |

---

## 6. Matrice di covarianza campionaria

La matrice di covarianza campionaria si calcola come:

```
R_hat = X @ X^H / K      (forma Nr x Nr, complessa hermitiana)
```

Dove:
- `X` e la matrice IQ (Nr x K)
- `K` numero di campioni
- `@` e il prodotto matriciale, `^H` il coniugato trasposto

**Significato degli elementi:**

- Elementi diagonali `R[k,k]`: potenza media del canale k (reali, > 0)
- Off-diagonali `R[i,k]` con i != k: correlazione incrociata tra canale i e k
  - `|R[i,k]|` indica la forza della correlazione (0 = segnali ortogonali)
  - `angle(R[i,k])` = differenza di fase stimata ch_i -> ch_k

La **coerenza normalizzata** e:

```
C[i,k] = |R[i,k]| / sqrt(R[i,i] * R[k,k])    <- valore in [0,1]
```

Valore vicino a 1: i due canali ricevono la stessa sorgente (correlati).
Valore vicino a 0: sorgenti indipendenti o assenza di segnale.

---

## 7. Decorrelazione della covarianza

Con segnali coerenti (multipath, riflessioni) la matrice R puo diventare
singolare o mal condizionata, degradando la DoA. Tre metodi di decorrelazione:

### Forward-Backward Averaging (FBA)

```
R_fb = (R + J * R* * J) / 2
```

Dove J e la matrice di scambio (anti-diagonale identita).
L operazione media R con la sua versione "time-reversed coniugata",
raddoppiando gli snapshot effettivi e sopprimendo fonti coerenti.

Costo: O(Nr^2). Raccomandato di default per indoor con multipath.

### Toeplitz Rectification (TOEP)

Media R lungo ciascuna diagonale, imponendo struttura Toeplitz:

```
R_TOEP[i,j] = (1/(M - |i-j|)) * Tr(R, |i-j|)
```

Valido per array shift-invariant (ULA a spaziatura uniforme).

### FB + Toeplitz (FBTOEP)

Applica in sequenza: Toeplitz forward + Toeplitz backward, poi media:

```
R_f = toeplitz(R[:,0], R[0,:])
R_b = toeplitz(flip(R[:,-1]), flip(R[-1,:]))
R_fbtoep = 0.5 * (R_f + R_b*)
```

Massima decorrelazione: raccomandato per Nr >= 5.

---

## 8. Da UCA a VULA: Phase Mode Excitation

Per usare Root-MUSIC ed ESPRIT (algoritmi polinomiali per ULA shift-invariant)
su una UCA circolare, si trasforma la UCA in una **Virtual ULA** (VULA) usando
la **Phase Mode Excitation** (decomposizione di Jacobi-Anger).

**Teoria:**

```
exp(j*z*cos(theta - phi_k)) = sum_{m=-inf}^{+inf} j^m * J_m(z) * exp(j*m*(theta-phi_k))
```

Dove `J_m(z)` e la funzione di Bessel del primo tipo di ordine m,
e `z = 2*pi*r*cos(phi_k)`.

Per N elementi uniformemente spaziati, si usano modi `-L <= m <= L`
con `L = floor(2*pi*r)`.

La matrice di trasformazione T (codice: `uca_to_vula()`):

```python
L = floor(2*pi*r_lambda)
ms = range(-L, L+1)
F = exp(2j*pi * outer(ms, n_idx) / N)   # (2L+1, N)
diag_vals = [1 / (1j^m * J_m(2*pi*r) + eps) for m in ms]
T = diag(diag_vals) @ F / N
```

**Pre-whitening:**

Per preservare le statistiche del rumore dopo la trasformazione:

```
A = T @ T^H
A_inv_sqrt = U @ diag(1/sqrt(eigenvalues(A))) @ U^H
X_vula = (A_inv_sqrt @ T) @ X
```

Per N=3 con r=0.289*lambda: L=1, VULA ha 3 elementi virtuali (stesso N).

---

## 9. Algoritmi DoA

### 9.1 MUSIC

**MUltiple SIgnal Classification** - algoritmo subspace standard.

```
P_MUSIC(theta) = 1 / (a^H(theta) * E_n * E_n^H * a(theta))
```

I picchi di P_MUSIC identificano le direzioni di arrivo.

- Divide gli autovettori di R in sottospazio segnale (D piu grandi) e rumore (Nr-D piu piccoli)
- Sfrutta l ortogonalita di a(theta_true) con E_n
- Risoluzione theoretically unlimited con K -> inf
- Implementazione: `doa_music()` in `doa_algorithms.py`

**Vantaggi:** robustezza, applicabile a qualsiasi geometria array, nessuna assunzione sul modello rumore.  
**Svantaggi:** degradazione con segnali coerenti (servono FBA/TOEP/FBTOEP).

### 9.2 Root-MUSIC via VULA

Versione polinomiale di MUSIC su Virtual ULA.

Anziché scansionare theta su una griglia, si trova il polinomio:

```
C(z) = sum_{k=-(Nr-1)}^{Nr-1} c_k * z^{-k}
     = z^{Nr-1} * a^H(z) * E_n * E_n^H * a(z)
```

Dove `c_k = Tr(E_n * E_n^H, k)` (traccia sulla k-esima diagonale).

Le radici di C(z) dentro il cerchio unitario piu vicine a |z|=1 danno:

```
theta_k = angle(z_k)
```

**Vantaggi:** accuratezza sub-griglia (non dipende da SCAN_POINTS), piu efficiente.  
**Svantaggi:** richiede Prima la trasformazione VULA.  
**Implementazione:** `doa_root_music()`.

### 9.3 Capon / MVDR

**Minimum Variance Distortionless Response** - beamformer adattivo.

```
P_MVDR(theta) = 1 / (a^H(theta) * R^{-1} * a(theta))
```

Non usa la decomposizione in sottospazi, opera direttamente su R^{-1}.

- Minimizza la potenza in output soggetto al vincolo di guadagno unitario in theta
- Migliore reiezione dei lobi laterali rispetto al beamformer convenzionale
- Meno sensibile al rango del sottospazio rispetto a MUSIC
- Implementazione: `doa_capon()`.

**Vantaggi:** buon compromesso robustezza/risoluzione indoor.  
**Svantaggi:** sensibile al condizionamento di R (serve loading diagonale).

### 9.4 Stochastic Maximum Likelihood (ML)

Massimizzazione della log-likelihood per modello stocastico gaussiano.
Per una sorgente, il criterio semplificato e:

```
J(theta) = p_hat(theta) / sigma2_hat(theta)

p_hat(theta)  = (a^H R a) / (a^H a)     [stima potenza segnale]
sigma2_hat(theta) = (Tr(R) - p_hat) / (M-1)  [stima rumore residuo]
```

Il denominatore "pena" gli angoli dove il rumore rimane alto dopo
aver "rimosso" la sorgente stimata.

**Vantaggi:** picchi piu nitidi di MUSIC con SNR elevato (come nel caso indoor).  
**Svantaggi:** computazionalmente piu pesante, assume D=1.  
**Implementazione:** `doa_ml()`.  
**Riferimento:** Stoica & Nehorai, IEEE Trans. ASSP 38(1), 1990.

### 9.5 ESPRIT

**Estimation of Signal Parameters via Rotational Invariance Techniques**.

Sfrutta l invarianza per traslazione della VULA:

```
a_2(theta) = e^{j*theta} * a_1(theta)
```

Dove a_1 = subarray [0..M-2] e a_2 = subarray [1..M-1].

Procedura:
1. Trasformare UCA -> VULA (Phase Mode Excitation)
2. Applicare FBA su R
3. Eigendecomporre R -> sottospazio segnale E_s (D autovettori piu grandi)
4. Partizionare: E_s1 = E_s[:-1, :], E_s2 = E_s[1:, :]
5. Operatore rotazionale LS: Phi = pinv(E_s1) @ E_s2
6. Autovalori mu_k di Phi -> theta_k = angle(mu_k)

**Vantaggi:** nessuna scansione angolare (velocita), accuratezza sub-griglia.  
**Svantaggi:** richiede VULA (UCA con Phase Mode), sensibile a N basso.  
**Implementazione:** `doa_esprit()`.  
**Riferimento:** Roy & Kailath, IEEE Trans. ASSP 37(7), 1989.

---

## 10. Covarianza temporale EMA

Per ridurre la varianza della stima (multipath rapido, jitter di frame),
la covarianza e aggiornata con una Exponential Moving Average tra frame:

```
R_ema[t] = alpha * R_ema[t-1] + (1-alpha) * R_new[t]
```

Con `alpha = COV_ALPHA = 0.92` (configurabile in `config.py`).

La costante di tempo e circa `tau = 1 / (1 - alpha) = 12.5 frame`.
A 9 fps (INTERVAL_MS=80 + latenza), questo corrisponde a ~1.4 secondi
di "memoria" della stima di covarianza.

**Quando aumentare alpha (es. 0.95):** segnale con multipath forte,
angolo abbastanza stabile nel tempo -> serve piu memoria temporale.

**Quando diminuire alpha (es. 0.7):** sorgente in movimento rapido,
si sacrifica stabilita per la reattivita del tracking.

---

## 11. Smoothing angolare basato su fasori

L angolo finale e smussato usando una EMA su fasori complessi per gestire
correttamente il wraparound 0 gradi / 360 gradi:

```python
phasor = exp(j * theta_rad)
phasor_ema = alpha * phasor_ema + (1-alpha) * phasor
theta_smooth = angle(phasor_ema)  # in radianti
```

Un EMA diretta sull angolo in gradi fallirebbe attorno al confine 0/360:
la media tra 359 gradi e 1 grado sarebbe 180 gradi (sbagliato!).
Con i fasori: media tra `exp(j*359*pi/180)` e `exp(j*1*pi/180)` = `exp(j*0) = 1`,
corrispondente a 0 gradi (corretto).

`ANGLE_SMOOTH_ALPHA = 0.75` -> costante di tempo ~4 frame (~0.4 s) a 9 fps.

---

## 12. Metriche diagnostiche

Il display real-time mostra 8 pannelli. Ecco il significato delle metriche:

### Pannello A: Pseudospettro MUSIC/Capon/ML
Il plot polare/lineare del pseudospettro in dB. Il picco indica la DoA stimata.
L ampiezza relativa indica la qualita della stima (picco piu alto = piu sicuro).

### Pannello B: Bussola + angolo stimato
Visualizzazione compass dell angolo finale smussato (`theta_smooth`).
Il badge in alto mostra:
- Angolo in gradi
- FPS corrente
- Algoritmo e decorrelazione attivi

### Pannello C: Istogramma angoli
Distribuzione storica degli angoli stimati (ultimi ~200 frame).  
`sigma` = deviazione standard circolare:

```
sigma [deg] = (180/pi) * sqrt(-2 * log(|mean(exp(j*theta_k))|))
```

- `sigma < 5 deg` : stima eccellente (stabile)
- `sigma < 15 deg` : stima buona per indoor
- `sigma > 30 deg` : instabilita / multipath severo

### Pannello D: Autovalori di R
Plot degli autovalori di R_ema in ordine decrescente.
- Autovalore 1 (piu grande): potenza sorgente dominante
- Autovalori 2..Nr: rumore + eventuali sorgenti secondarie
- Gap netto tra ev1 e ev2: buona separazione segnale/rumore

### Pannello E: Matrice di coerenza
Heatmap del modulo normalizzato `|C[i,k]| = |R[i,k]| / sqrt(R[i,i]*R[k,k])`.

```
mean |mu_OD| = media dei valori off-diagonali della matrice coerenza
```

- Valore alto (>0.7): segnale coerente tra i canali (buona DoA)
- Valore basso (<0.3): canali quasi indipendenti (rumore, segnale assente)

### Pannello F: PAPR e SNR stimato
**PAPR (Peak-to-Average Power Ratio):** rapporto tra lo snapshot IQ di
massima potenza e la potenza media:

```
PAPR = max(|x_k|^2) / mean(|x_k|^2)   [per tutti i campioni nel frame]
```

**SNR stimato** dall eigenvalue dominante vs. rumore:

```
SNR_dB = 10*log10((ev_max - sigma^2_noise) / sigma^2_noise)
sigma^2_noise = mean(Nr-D min eigenvalues)
```

### Pannello G: Spettro FFT
FFT del canale 0 (antenna di riferimento). Identifica la frequenza del segnale
nel frame corrente. Il marker verticale indica la frequenza stimata del picco.

### Pannello H: Fasi inter-canale
Evoluzione temporale di `angle(R[0,k])` per k=1,...,Nr-1.

Una fase stabile nel tempo indica:
- Il segnale e coerente tra le antenne
- La calibrazione Heimdall e completata
- La DoA e affidabile

Oscillazioni rapide indicano multipath o segnale debole.

---

## 13. Calibrazione di fase hardware

Il KrakenSDR ha piccole derive di fase hardware tra i canali
(di solito 1-10 gradi). Questi offset si correggono in `config.py`:

```python
PHASE_OFFSETS_DEG = [0.0, 0.0, 0.0]
```

**Procedura di misurazione (metodo diretto):**

1. Dirigere tutte le antenne verso la stessa sorgente a distanza nota
   e grande (campo lontano, >10 wavelengths)
2. Eseguire pysdr_doa_realtime.py con algoritmo MUSIC
3. Osservare la fase nel pannello H per qualche secondo
4. Calcolare la fase media per ciascuna coppia con il seguente snippet:

```python
import numpy as np

# Esempio: X e un frame acquisito con sorgente in campo lontano
# X.shape = (3, K)
K = X.shape[1]
R = X @ X.conj().T / K

# Offset del canale k rispetto al canale 0
for k in range(1, X.shape[0]):
    phi = np.angle(R[0, k])
    print(f"PHASE_OFFSETS_DEG[{k}] = {-np.degrees(phi):.2f}")
```

5. Copiare i valori (negati) in PHASE_OFFSETS_DEG in config.py
6. Riavviare l applicazione

**Procedura "Set Zero" automatica** (tasto nel pannello B del display):
Cattura un singolo frame e imposta automaticamente i PHASE_OFFSETS_DEG
basandosi sulla fase corrente di R. Utile per calibrazione rapida sul campo.

---

## 14. Registrazione e playback

### Registrazione (pysdr_doa_realtime.py)

Durante l acquisizione real-time e possibile registrare i frame raw per
analisi offline. I frame vengono salvati in formato `.npz` (NumPy compressed):

```python
np.savez_compressed(
    filepath,
    X=frames_list,      # array (N_frames, Nr, K) complex64
    hdr_freq=freq_hz,
    hdr_fs=sample_rate,
    timestamp=time.time()
)
```

Esempio di utilizzo da riga di comando alla fine di una sessione real-time,
i file `.npz` vengono salvati in `recordings/`.

### Playback (pysdr_doa_playback.py)

Avviare con:

```bash
python3 pysdr_doa_playback.py recordings/session_YYYYMMDD_HHMMSS.npz
```

Il player mostra gli stessi 8 pannelli del real-time, con controlli:
- **Play/Pause**: avvio e pausa della riproduzione
- **Rewind**: riavvolgimento all inizio
- **x 0.5 / x 1 / x 2 / x 4**: velocita di riproduzione
- **Slider**: navigazione manuale frame per frame

Tutti i parametri di configurazione di `config.py` vengono applicati anche
in playback (stesso algoritmo, stessa decorrelazione), permettendo di confrontare
diversi algoritmi sugli stessi dati registrati.

---

## 15. Risultati sperimentali

Dati misurati (sessione indoor, 24 marzo 2026):

| Metrica                   | Valore       | Interpretazione                              |
|---------------------------|--------------|----------------------------------------------|
| SNR stimato               | 22 dB        | Segnale forte, nessun problema di potenza    |
| cond(R)                   | 196          | Matrice mal condizionata (multipath severo)  |
| sigma angolare (raw)      | 28.3 deg     | Instabilita causata da multipath             |
| sigma angolare (EMA 0.92) | 1-3 deg      | Stabile con memoria temporale lunga          |
| fase off-diag sigma       | 1.6 rad      | Varianza di fase elevata frame-to-frame      |
| imbalance canali          | 4.6 dB       | Sbilanciamento guadagno -> AMPLITUDE_NORMALIZE|
| Algoritmo ottimale indoor | ML + FBA     | Migliore picco con SNR alto + multipath      |

**Raccomandazioni indoor:**
- Usare `DOA_ALGORITHM = "ML"` con `DECORRELATION = "FBA"`
- `COV_ALPHA = 0.92` (memoria ~1.4 s)
- `AMPLITUDE_NORMALIZE = True` per compensare squilibri di guadagno
- `ANGLE_SMOOTH_ALPHA = 0.75` per ridurre jitter visivo

---

## 16. Riferimento alla configurazione

Tutti i parametri sono in `config.py`:

| Parametro              | Default       | Descrizione                                        |
|------------------------|---------------|----------------------------------------------------|
| HEIMDALL_HOST          | "127.0.0.1"   | IP server Heimdall                                 |
| HEIMDALL_PORT          | 5000          | Porta IQ data                                      |
| HEIMDALL_CTRL          | 5001          | Porta controllo                                    |
| N_ANTENNAS             | 3             | Numero antenne                                     |
| GEOMETRY               | "UCA"         | "UCA" o "ULA"                                      |
| RADIUS_LAMBDA          | 0.289         | Raggio UCA in frazioni di lambda                   |
| D_LAMBDA               | 0.5           | Spaziatura ULA in frazioni di lambda               |
| FREQ_HZ                | 865.21e6      | Frequenza portante [Hz]                            |
| SAMPLE_RATE_HZ         | 1.024e6       | Frequenza di campionamento [Hz]                    |
| GAIN_DB                | 30.0          | Guadagno IF [dB]                                   |
| SCAN_POINTS            | 360           | Risoluzione griglia angolare                       |
| NUM_SIGNALS            | 1             | Numero sorgenti attese                             |
| HW_NUM_SAMPLES         | 0             | Campioni usati per frame (0=tutti)                 |
| INTERVAL_MS            | 80            | ms tra frame animazione                            |
| DOA_ALGORITHM          | "ML"          | "MUSIC","ROOT-MUSIC","CAPON","ML","ESPRIT"         |
| DECORRELATION          | "FBA"         | "Off","FBA","TOEP","FBTOEP"                        |
| COV_ALPHA              | 0.92          | EMA covarianza (0=no memoria, 1=no update)         |
| ANGLE_SMOOTH_ALPHA     | 0.75          | EMA angolo su fasori (0=disabilitato)              |
| AMPLITUDE_NORMALIZE    | True          | Normalizza ampiezza canali a potenza unitaria      |
| SQUELCH_ENABLED        | True          | Salta DoA se potenza < soglia                      |
| SQUELCH_THRESHOLD_DB   | -60.0         | Soglia squelch [dBW]                               |
| PHASE_OFFSETS_DEG      | [0,0,0]       | Correzione fase hardware per canale [gradi]        |

---

## 17. Quick Start

### Avvio in Docker (raccomandato)

```bash
# 1. Avvia Heimdall + GNU Radio Companion
# (usare il task VS Code "Open GRC + Heimdall")

# 2. Verifica che Heimdall sia in ascolto
nc -z 127.0.0.1 5000 && echo "Heimdall OK" || echo "Heimdall non raggiungibile"

# 3. Avvia il display real-time
cd workspace/pysdr_doa
python3 pysdr_doa_realtime.py

# 4. (oppure) Playback di una registrazione
python3 pysdr_doa_playback.py recordings/nome_file.npz
```

### Modifica dei parametri

```bash
# Apri config.py con il tuo editor
nano workspace/pysdr_doa/config.py

# Cambia frequenza portante (esempio: 433 MHz)
FREQ_HZ = 433e6

# Cambia algoritmo
DOA_ALGORITHM = "MUSIC"   # piu stabile in ambienti rumorosi

# Riavvia l applicazione (le modifiche vengono lette all avvio)
python3 pysdr_doa_realtime.py
```

### Verifica sync Heimdall

Se il display mostra "waiting for sync", Heimdall sta calibrando i canali.
La sincronizzazione richiede tipicamente 5-15 secondi all avvio.
Verificare i flag header:

```python
hdr = src.last_header
print(f"sync_state={hdr.sync_state}")      # deve essere >= 2
print(f"delay_sync={hdr.delay_sync_flag}") # deve essere 1
print(f"iq_sync={hdr.iq_sync_flag}")       # deve essere 1
```

---

## 18. Bibliografia

1. **Schmidt, R. O.** (1986). Multiple emitter location and signal parameter estimation. *IEEE Trans. Antennas Propagat.*, 34(3), 276-280.

2. **Barabell, A. J.** (1983). Improving the resolution performance of eigenstructure-based direction-finding algorithms. *Proc. ICASSP*, pp. 336-339.

3. **Roy, R. & Kailath, T.** (1989). ESPRIT — Estimation of signal parameters via rotational invariance techniques. *IEEE Trans. ASSP*, 37(7), 984-995.

4. **Pillai, S. U. & Kwon, B. H.** (1989). Forward/backward spatial smoothing techniques for coherent signal identification. *IEEE Trans. ASSP*, 37(4), 8-15.

5. **Tewfik, A. H. & Hong, M.** (1992). On the application of uniform linear array bearing estimation techniques to uniform circular arrays. *IEEE Trans. Signal Processing*, 40(4), 1008-1011.

6. **Capon, J.** (1969). High-resolution frequency-wavenumber spectrum analysis. *Proc. IEEE*, 57(8), 1408-1418.

7. **Stoica, P. & Nehorai, A.** (1990). Performance study of conditional and unconditional direction-of-arrival estimation. *IEEE Trans. ASSP*, 38(1), 133-149.

8. **Vallet, P. & Loubaton, P.** (2014). Toeplitz rectification and DoA estimation with MUSIC. *Proc. ICASSP*.

9. **McDonald, A. & van Wyk, M. A.** (2019). Direction of arrival estimation using modified FB Toeplitz matrix. *IEEE PrimeAsia*.

10. **Marsal, S. et al.** (2022). KrakenSDR project. *https://github.com/krakenrf/krakensdr_doa*
