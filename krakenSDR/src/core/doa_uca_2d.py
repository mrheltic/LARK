"""
doa_uca_2d — 2D (azimuth + elevation) DoA per Uniform Circular Array
======================================================================

Algoritmi di stima della Direzione di Arrivo in 2D (azimuth + elevazione)
per un array circolare uniforme (UCA) a N antenne.

Geometria UCA (piano Est-Nord, coordinate in λ):

              ant0 (Nord)
             /
    ant4 ···●··· ant1
            |
          ant3   ant2

    Antenna k al angolo  φ_k = 2π·k/N  dal Nord in senso orario.
    Coordinate East-North:
        p_k_E = r · sin(φ_k)
        p_k_N = r · cos(φ_k)

Convenzione angolare (uguale a doa_algorithms_3d per coerenza):
    azimuth  φ : gradi dal Nord, senso orario (0°=N, 90°=E, 180°=S, 270°=W)
    elevation θ : gradi sopra l'orizzonte     (0°=orizzonte, 90°=zenith)

Ritardo di fase sull'antenna k per una sorgente a (φ, θ):
    τ_k = 2π · ( p_k_E · cos(θ) · sin(φ) + p_k_N · cos(θ) · cos(φ) )

Steering vector:
    a(φ,θ) = [exp(j·τ_0), …, exp(j·τ_{N-1})]^T ∈ ℂ^N

Algoritmi implementati
----------------------
- 2D-MUSIC   : P = 1 / ‖E_n^H · a‖²   (super-risoluzione angolare)
- 2D-Capon   : P = 1 / (a^H · R^{-1} · a)  (MVDR)
- 2D-Bartlett: P = a^H · R · a         (beamformer convenzionale, più robusto)

Tutte le funzioni restituiscono uno spettro (n_el, n_az) in dB
(picco = 0 dB, floor = −40 dB), compatibile con `find_peak_2d` di
`doa_algorithms_3d.py`.

Riferimenti
-----------
* Schmidt R.O., IEEE Trans. Antennas Propagat. 34(3), 1986        — MUSIC
* Capon J., Proc. IEEE 57(8), 1969                                — MVDR
* Van Trees H.L., Optimum Array Processing, Wiley 2002, §9.2      — UCA steering
* Mathews C.P. & Zoltowski M.D., IEEE Trans. SP 42(9), 1994       — UCA phase modes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# =============================================================================
# UCA configuration
# =============================================================================

@dataclass
class UcaConfig:
    """
    Configurazione di un UCA a N antenne per DoA 2D (azimuth + elevazione).

    Parametri
    ---------
    n_ant                : numero di antenne          default 5
    radius_lambda        : raggio in frazioni di λ    default 0.5
                           A 868 MHz con r=17.3 cm → r=0.5λ
                           A 868 MHz con r=12.5 cm → r≈0.362λ (vecchio KrakenSDR)
    n_az                 : punti di scansione azimuth (0…360°)   default 72 → 5° step
    n_el                 : punti di scansione elevazione (el_min…90°) default 18 → 5°
    el_min_deg           : elevazione minima [°]      default 5°
    num_expected_signals : sorgenti attese (D nello split del sottospazio MUSIC)
                           0 = auto-detect via MDL
    ant0_offset_deg      : rotazione fisica dell'antenna 0 rispetto al Nord [°]
                           Usare calibrazione sul campo per correggerlo.
    ant_ccw              : True se le antenne fisiche sono poste in senso anti-orario
                           (CCW) guardando dall'alto; False = orario (CW, default).
                           Con matrice CW su array CCW: az_stimato = 360° − az_reale.
    """
    n_ant:                int   = 5
    radius_lambda:        float = 0.5
    n_az:                 int   = 72
    n_el:                 int   = 18
    el_min_deg:           float = 5.0
    num_expected_signals: int   = 1
    ant0_offset_deg:      float = 0.0   # rotazione fisica ant0 rispetto al Nord
    ant_ccw:              bool  = False  # True = antenne in senso anti-orario (CCW)

    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # ── Geometria ─────────────────────────────────────────────────────────────

    @property
    def positions(self) -> np.ndarray:
        """(n_ant, 2) array: coordinate [Est, Nord] per antenna in lunghezze d'onda."""
        k = np.arange(self.n_ant, dtype=np.float64)
        # senso positivo = orario (CW); ant_ccw=True inverte il segno dell'angolo
        sign  = -1.0 if self.ant_ccw else 1.0
        phi_k = np.deg2rad(self.ant0_offset_deg) + sign * 2.0 * np.pi * k / self.n_ant
        return np.column_stack([
            self.radius_lambda * np.sin(phi_k),   # Est
            self.radius_lambda * np.cos(phi_k),   # Nord
        ])

    def az_range_deg(self) -> np.ndarray:
        """Griglia di scansione azimuth in gradi: 0° … 360°."""
        return np.linspace(0.0, 360.0, self.n_az, endpoint=False)

    def el_range_deg(self) -> np.ndarray:
        """Griglia di scansione elevazione in gradi: el_min … 90°."""
        return np.linspace(self.el_min_deg, 90.0, self.n_el)

    # ── Matrice di steering (pre-calcolata e cachata) ─────────────────────────

    def get_steering_matrix(self) -> np.ndarray:
        """
        Matrice di steering (n_ant, n_el × n_az), pre-calcolata e messa in cache.

        Indice di colonna: i_el * n_az + i_az  →  punto di griglia (el[i_el], az[i_az]).

        Formula identica a CrossArrayConfig.get_steering_matrix():
            τ_k = 2π · (p_k_E · cosθ · sinφ + p_k_N · cosθ · cosφ)
            A_k = exp(j·τ_k)
        """
        key = ('uca2d',
               self.n_ant,
               round(self.radius_lambda, 8),
               self.n_az, self.n_el,
               round(self.el_min_deg, 4),
               round(self.ant0_offset_deg, 4),
               bool(self.ant_ccw))
        if key not in self._cache:
            az = np.deg2rad(self.az_range_deg())   # (n_az,)
            el = np.deg2rad(self.el_range_deg())   # (n_el,)

            AZ, EL = np.meshgrid(az, el)           # (n_el, n_az)

            # Coseni direttori nel piano Est-Nord
            u_east  = np.cos(EL) * np.sin(AZ)     # cosθ · sinφ
            u_north = np.cos(EL) * np.cos(AZ)     # cosθ · cosφ

            # Appiattire a (N_grid,) dove N_grid = n_el × n_az
            ue = u_east.ravel()
            un = u_north.ravel()

            # Ritardi di fase (n_ant, N_grid)
            p   = self.positions                   # (n_ant, 2)
            tau = 2.0 * np.pi * (
                p[:, 0:1] * ue[np.newaxis, :]
              + p[:, 1:2] * un[np.newaxis, :]
            )
            self._cache[key] = np.exp(1j * tau).astype(np.complex128)

        return self._cache[key]

    def invalidate_cache(self) -> None:
        """Forza il ricalcolo della matrice di steering al prossimo accesso."""
        self._cache.clear()

    # ── Compatibilità con find_peak_2d di doa_algorithms_3d ─────────────────

    @property
    def d_lambda(self) -> float:
        """Alias per compatibilità con CrossArrayConfig (non usato nel calcolo)."""
        return self.radius_lambda


# =============================================================================
# Funzione di picco (compatibile con doa_algorithms_3d.find_peak_2d)
# =============================================================================

def find_peak_uca_2d(
    spec: np.ndarray,
    cfg:  UcaConfig,
) -> Tuple[float, float, float]:
    """
    Trova (azimuth_deg, elevation_deg, papr_db) dallo spettro 2D.

    Usa l'interpolazione parabolica 2D attorno al bin di picco per
    accuratezza sub-griglia (±metà step di griglia).

    Restituisce
    -----------
    az_deg  : azimuth stimato  [°, 0…360]
    el_deg  : elevazione stimata [°, el_min…90]
    papr_db : Peak-to-Average Power Ratio dello spettro [dB]
    """
    idx       = np.unravel_index(np.argmax(spec), spec.shape)
    i_el, i_az = int(idx[0]), int(idx[1])
    n_el, n_az  = spec.shape

    # Azimuth — periodico (wrap-around)
    az_frac = 0.0
    ym = float(spec[i_el, (i_az - 1) % n_az])
    y0 = float(spec[i_el,  i_az])
    yp = float(spec[i_el, (i_az + 1) % n_az])
    denom_az = ym - 2.0 * y0 + yp
    if abs(denom_az) > 1e-6:
        az_frac = float(np.clip(0.5 * (ym - yp) / denom_az, -0.5, 0.5))

    # Elevazione — non periodica
    el_frac = 0.0
    if 0 < i_el < n_el - 1:
        ym = float(spec[i_el - 1, i_az])
        y0 = float(spec[i_el,     i_az])
        yp = float(spec[i_el + 1, i_az])
        denom_el = ym - 2.0 * y0 + yp
        if abs(denom_el) > 1e-6:
            el_frac = float(np.clip(0.5 * (ym - yp) / denom_el, -0.5, 0.5))

    az_step = 360.0 / n_az
    el_step = (90.0 - cfg.el_min_deg) / max(n_el - 1, 1)

    az_deg = (cfg.az_range_deg()[i_az] + az_frac * az_step) % 360.0
    el_deg = float(
        np.clip(cfg.el_range_deg()[i_el] + el_frac * el_step,
                cfg.el_min_deg, 90.0)
    )

    s_lin  = 10.0 ** (np.clip(spec, -200.0, 0.0) / 10.0)
    mean_v = float(np.mean(s_lin))
    papr   = (10.0 * np.log10(float(np.max(s_lin)) / (mean_v + 1e-15))
              if mean_v > 1e-15 else 0.0)

    return float(az_deg), float(el_deg), float(papr)


# =============================================================================
# Interno: calcolo covarianza
# =============================================================================

def _get_cov(X: np.ndarray, R_in: np.ndarray | None) -> np.ndarray:
    if R_in is not None:
        return np.asarray(R_in, dtype=complex)
    return (X @ X.conj().T) / X.shape[1]


# =============================================================================
# Decorrelazione della covarianza (anti-multipath)
# =============================================================================

def _circulant_smooth(R: np.ndarray) -> np.ndarray:
    """
    Circulant averaging (UCA spatial smoothing, Mathews & Zoltowski 1994).

    Per ogni lag circolare l calcola la media:
        d[l] = (1/N) * sum_{k=0}^{N-1}  R[k, (k+l) mod N]
    e ricostruisce la matrice circolante R_c[i,j] = d[(j-i) mod N].

    Equivalente a N sovrapposizione di sotto-array virtuali (rotazioni del UCA):
        R_c = (1/N) * sum_k  Π^k · R · (Π^k)^H
    dove Π è la matrice di permutazione ciclica.

    Benefici:
    - Decorrela sorgenti coerenti (multipath) → N/2 copie coerenti tollerabili
    - Forza la struttura circolante teorica della covarianza UCA
    - Fonte: Ita97/2D_MUSIC_DOA (spatial smoothing) adattato per UCA circolare
    """
    N = R.shape[0]
    k = np.arange(N)
    # Vettore del primo lag (d[0]..d[N-1]) — media su tutti gli starting point
    d = np.empty(N, dtype=complex)
    for lag in range(N):
        d[lag] = np.mean(R[k, (k + lag) % N])
    # Costruisci matrice circolante: R_c[i, j] = d[(j - i) % N]
    rows = [np.roll(d, i) for i in range(N)]
    R_c  = np.array(rows, dtype=complex)
    # Assicura simmetria Hermitiana (errori numerici)
    return (R_c + R_c.conj().T) * 0.5


def _fb_average_uca(R: np.ndarray) -> np.ndarray:
    """
    Forward-Backward averaging per UCA: R_fb = 0.5 * (R + J · R* · J).

    J è la matrice di scambio anti-diagonale (exchange matrix).
    Per ULA pari è esatta (a(-ψ) = J·a*(ψ) a meno di fase scalare).
    Per UCA N=5 è approssimata ma migliora la robustezza al multipath coerente
    riducendo il rango effettivo delle sorgenti coerenti.

    Implementazione analoga a Ita97/2D_MUSIC_DOA fb=True.
    """
    N = R.shape[0]
    J    = np.eye(N, dtype=complex)[::-1, :]   # antidiagonale identità
    R_fb = 0.5 * (R + J @ np.conj(R) @ J)
    return (R_fb + R_fb.conj().T) * 0.5  # forza simmetria Hermitiana


def _decor_cov(R: np.ndarray, mode: str) -> np.ndarray:
    """
    Applica decorrelazione alla covarianza.

    mode: 'none' | 'circulant' | 'fb' | 'both'
        'circulant' = solo circulant smoothing  (preferito per Capon)
        'fb'        = solo forward-backward     (limitato senza circulant)
        'both'      = circulant poi FB          (default per MUSIC indoor)
    """
    if mode in ("circulant", "both"):
        R = _circulant_smooth(R)
    if mode in ("fb", "both"):
        R = _fb_average_uca(R)
    return R


# =============================================================================
# 2D-MUSIC per UCA
# =============================================================================

def doa_music_uca_2d(
    X:           np.ndarray,
    cfg:         UcaConfig,
    R_in:        np.ndarray | None = None,
    n_snapshots: int | None = None,
    decorr:      str = "none",
) -> np.ndarray:
    """
    Pseudo-spettro 2D-MUSIC per UCA.

    P(φ, θ) = 1 / ‖E_n^H · a(φ, θ)‖²

    NOTA sulla decorrelazione per UCA:
    - Il circulant smoothing è corretto per URA (Ita97/2D_MUSIC_DOA, sps=True)
      ma NON per UCA: forza R ad essere ciclicamente simmetrica → autovettori
      = vettori DFT → spettro MUSIC con simmetria N-fold (stella a N punte).
    - Per UCA la decorrelazione anti-multipath corretta è la media temporale
      (EMA, gestita da CovarianceAccumulatorUca) — con alpha=0.97, 33 frame
      di integrazione decorrelano il multipath indoor.
    - decorr='none' è il default sicuro. Usare 'circulant'/'fb' solo in
      esperimenti offline con molti snapshot garantiti.

    Parametri
    ---------
    X           : (n_ant, N_campioni) complesso
    cfg         : UcaConfig
    R_in        : covarianza pre-calcolata (es. EMA); se fornita X non è usata
    n_snapshots : campioni IQ (per MDL auto-detect)
    decorr      : 'none' (default) | 'circulant' | 'fb' | 'both'

    Restituisce
    -----------
    spec : (n_el, n_az) float ndarray  [dB, picco = 0, floor = −40 dB]
    """
    R = _decor_cov(_get_cov(X, R_in), decorr)
    M = R.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(R)   # autovalori crescenti

    # Diagonal loading leggero: evita singolarità senza distorcere il sottospazio
    eps = 1e-6 * max(float(abs(eigenvalues[-1])), 1e-20)
    R   = R + eps * np.eye(M, dtype=complex)

    # Auto-detect sorgenti via MDL (Wax & Kailath 1985) se num_expected_signals == 0
    if cfg.num_expected_signals == 0:
        if n_snapshots is not None:
            _N = n_snapshots
        elif R_in is None:
            _N = X.shape[1]
        else:
            _N = 1024
        n_sig = _estimate_signal_count_mdl(R, _N, M - 1)
        if n_sig == 0:
            n_sig = 1
    else:
        n_sig = max(1, min(cfg.num_expected_signals, M - 1))

    En = eigenvectors[:, :-n_sig]           # (M, M-n_sig) sottospazio rumore

    A  = cfg.get_steering_matrix()          # (M, N_grid)
    Pa = En.conj().T @ A                   # (M-n_sig, N_grid)

    denom    = np.real(np.sum(np.abs(Pa) ** 2, axis=0))   # (N_grid,)
    pspec    = 1.0 / (denom + 1e-12)
    pspec_db = 10.0 * np.log10(pspec / (float(np.max(pspec)) + 1e-12) + 1e-12)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# 2D-Bartlett (CBF) per UCA
# =============================================================================

def doa_bartlett_uca_2d(
    X:    np.ndarray,
    cfg:  UcaConfig,
    R_in: np.ndarray | None = None,
) -> np.ndarray:
    """
    Beamformer convenzionale 2D (delay-and-sum) per UCA.

    P(φ, θ) = a^H(φ, θ) · R · a(φ, θ)

    Il più robusto quando un canale è degradato — degrada gracefully
    allargando il lobo principale invece di fallire silenziosamente.

    Restituisce
    -----------
    spec : (n_el, n_az) float ndarray  [dB, picco = 0, floor = −40 dB]
    """
    R = _get_cov(X, R_in)
    A = cfg.get_steering_matrix()              # (M, N_grid)

    pspec    = np.real(np.sum(A.conj() * (R @ A), axis=0))   # (N_grid,)
    pspec    = np.maximum(pspec, 1e-30)
    pspec_db = 10.0 * np.log10(pspec / (float(np.max(pspec)) + 1e-30) + 1e-30)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# 2D-Capon (MVDR) per UCA
# =============================================================================

def doa_capon_uca_2d(
    X:    np.ndarray,
    cfg:  UcaConfig,
    R_in: np.ndarray | None = None,
    decorr: str = "none",
) -> np.ndarray:
    """
    Beamformer 2D-Capon (MVDR) per UCA.

    P(φ, θ) = 1 / (a^H(φ, θ) · R^{-1} · a(φ, θ))

    decorr='none' di default (vedi nota in doa_music_uca_2d).
    Il circulant smoothing forza N-fold symmetry rendendo Capon
    equivalente a Bartlett su una covarianza degradata.

    Restituisce
    -----------
    spec : (n_el, n_az) float ndarray  [dB, picco = 0, floor = −40 dB]
    """
    R = _decor_cov(_get_cov(X, R_in), decorr)

    M   = R.shape[0]
    # Loading adattivo basato sul max autovalore
    ev_max = float(abs(np.linalg.eigvalsh(R)[-1]))
    eps    = 1e-4 * max(ev_max, 1e-20)
    R      = R + eps * np.eye(M, dtype=complex)

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.eye(M, dtype=complex)

    A  = cfg.get_steering_matrix()           # (M, N_grid)
    Qa = R_inv @ A                           # (M, N_grid)

    denom    = np.real(np.sum(A.conj() * Qa, axis=0))   # (N_grid,)
    pspec    = 1.0 / (np.maximum(denom, 1e-30))
    pspec_db = 10.0 * np.log10(pspec / (float(np.max(pspec)) + 1e-30) + 1e-30)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# Metriche di qualità del segnale
# =============================================================================

def eigenvalue_spread_uca_db(R: np.ndarray) -> np.ndarray:
    """
    Autovalori della covarianza in dB, ordinati decrescenti.
    Normalizzati rispetto al più piccolo (piano del rumore = 0 dB).
    """
    ev = np.sort(np.maximum(np.linalg.eigvalsh(R), 0.0))[::-1]
    return 10.0 * np.log10(ev / (ev[-1] + 1e-20) + 1e-20)


def snr_uca_db(R: np.ndarray) -> float:
    """Stima SNR [dB] dal rapporto autovalore max/min di R."""
    ev    = np.sort(np.maximum(np.linalg.eigvalsh(R), 0.0))
    ratio = (ev[-1] - ev[0]) / (ev[0] + 1e-20)
    return float(10.0 * np.log10(max(ratio, 1e-10)))


# =============================================================================
# EMA covariance accumulator per UCA
# =============================================================================

class CovarianceAccumulatorUca:
    """
    Accumulatore EMA della covarianza per UCA.

    R_new = α · R_old + (1 − α) · R_frame

    α = 0 → solo frame corrente;  α → 1 → memoria lunga.
    Usato in modalità CW continua; per burst usa la covarianza single-shot.
    """
    def __init__(self, alpha: float = 0.90):
        self.alpha = float(alpha)
        self._R: np.ndarray | None = None

    def update(self, X: np.ndarray) -> np.ndarray:
        """Aggiorna con il frame IQ (n_ant × N) e restituisce la covarianza EMA."""
        R_frame = (X @ X.conj().T) / X.shape[1]
        if self._R is None:
            self._R = R_frame.copy()
        else:
            self._R = self.alpha * self._R + (1.0 - self.alpha) * R_frame
        return self._R

    def reset(self) -> None:
        self._R = None

    @property
    def R(self) -> np.ndarray | None:
        return self._R


# =============================================================================
# Interno: stima del numero di sorgenti (MDL semplificato)
# =============================================================================

def _estimate_signal_count_mdl(
    R:          np.ndarray,
    n_snapshots: int,
    max_signals: int = 4,
) -> int:
    """MDL criterion for signal count estimation (Wax & Kailath 1985)."""
    M  = R.shape[0]
    ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
    ev = np.maximum(ev, 1e-30)
    D_max = min(max_signals, M - 1)
    N = float(n_snapshots)
    n = float(M)

    best_k, best_cost = 0, float("inf")
    for k in range(D_max + 1):
        noise_ev = ev[k:]
        m = float(len(noise_ev))
        g_k = float(np.exp(np.mean(np.log(noise_ev))))
        a_k = float(np.mean(noise_ev))
        if a_k < 1e-30 or g_k < 1e-30:
            break
        likelihood = N * m * float(np.log(a_k / g_k))
        penalty    = float(k) * (2.0 * n - float(k))
        cost = likelihood + 0.5 * penalty * np.log(N)
        if cost < best_cost:
            best_cost = cost
            best_k    = k

    return best_k
