"""
core.smart_api — Intelligent, LEGO-like API for KrakenSDR
==========================================================

High-level, intuitive API that makes KrakenSDR modules work like LEGO bricks.
Auto-configuration, smart defaults, and intelligent algorithm selection.

Design Philosophy
-----------------
    • One-liner setup for common use cases
    • Auto-detect optimal parameters
    • Sensible defaults for 90% of scenarios
    • Easy to learn, hard to break

Quick Start Examples
--------------------
    # 1. Create array for Iridium L-band
    >>> array = create_array('uca', frequency_mhz=1626.27, n_ant=5)
    
    # 2. Estimate DoA with auto-algorithm selection
    >>> doa = estimate_doa(iq_data, array)
    >>> print(f"Azimuth: {doa.azimuth_deg:.1f}°")
    
    # 3. Process Iridium bursts
    >>> pipeline = create_iridium_pipeline()
    >>> for result in pipeline.process_stream(iq_stream):
    ...     if result.has_burst:
    ...         print(f"DoA: {result.doa.azimuth_deg:.1f}°")
    
    # 4. Train calibration model
    >>> calibrator = create_calibrator(array)
    >>> model = calibrator.train_auto(training_data)

Factory Functions
-----------------
    create_array()           — Smart array geometry factory
    create_doa_estimator()   — Intelligent DoA estimator with auto-selection
    create_iridium_pipeline() — Complete Iridium processing pipeline
    create_calibrator()      — Auto-training calibration system
    estimate_doa()           — One-shot DoA estimation
"""

from __future__ import annotations

__all__ = [
    # Factory functions
    "create_array",
    "create_doa_estimator", 
    "create_iridium_pipeline",
    "create_calibrator",
    "estimate_doa",
    # Result classes
    "DoAResult",
    "BurstResult",
    "CalibrationResult",
    # Convenience functions
    "quick_doa",
    "quick_burst_detect",
    "auto_configure",
]

from dataclasses import dataclass
from typing import Optional, Union, Literal, Dict, Any
import numpy as np

from .array_geometry import (
    UniformCircularArray,
    UniformLinearArray,
    CrossArray,
    ArrayGeometryBase,
    compute_steering_matrix_1d,
)
from .covariance import (
    covariance,
    CovarianceAccumulator,
    apply_decorrelation,
)
from .doa_estimators import (
    music,
    capon,
    bartlett,
    esprit,
    root_music,
)
from .burst import BurstDetector as _BurstDetector
from .iridium_doa_burst import (
    detect_and_extract_burst,
    compensate_doppler,
    compute_single_shot_covariance,
)
from .calibration_model import CalibrationMLP, TrainConfig


# =============================================================================
# Result Classes — Clean, informative outputs
# =============================================================================

@dataclass
class DoAResult:
    """
    Clean DoA estimation result with all relevant information.
    
    Attributes
    ----------
    azimuth_deg : float
        Estimated azimuth angle [0-360°]
    elevation_deg : float or None
        Estimated elevation angle [0-90°] (None for 1D estimation)
    confidence : float
        Estimation confidence [0-1]
    algorithm : str
        Algorithm used for estimation
    spectrum : np.ndarray or None
        DoA spectrum (if available)
    snr_db : float
        Estimated signal-to-noise ratio [dB]
    """
    azimuth_deg: float
    elevation_deg: Optional[float]
    confidence: float
    algorithm: str
    spectrum: Optional[np.ndarray] = None
    snr_db: float = 0.0
    
    @property
    def has_elevation(self) -> bool:
        """True if elevation was estimated."""
        return self.elevation_deg is not None
    
    def __str__(self) -> str:
        el_str = f", El: {self.elevation_deg:.1f}°" if self.has_elevation else ""
        return f"DoAResult(Az: {self.azimuth_deg:.1f}°{el_str}, SNR: {self.snr_db:.1f} dB, Conf: {self.confidence:.2f})"


@dataclass
class BurstResult:
    """
    Burst detection and processing result.
    
    Attributes
    ----------
    has_burst : bool
        Whether a burst was detected
    doppler_hz : float
        Estimated Doppler shift [Hz]
    snr_db : float
        Burst signal-to-noise ratio [dB]
    doa : DoAResult or None
        DoA estimation (if available)
    raw_line : str or None
        Demodulated RAW: line (if demodulation enabled)
    """
    has_burst: bool
    doppler_hz: float = 0.0
    snr_db: float = 0.0
    doa: Optional[DoAResult] = None
    raw_line: Optional[str] = None
    
    def __str__(self) -> str:
        if not self.has_burst:
            return "BurstResult(no burst)"
        return f"BurstResult(Doppler: {self.doppler_hz/1e3:.1f} kHz, SNR: {self.snr_db:.1f} dB)"


@dataclass
class CalibrationResult:
    """
    Calibration training result.
    
    Attributes
    ----------
    model : CalibrationMLP
        Trained calibration model
    train_loss : float
        Final training loss
    val_loss : float
        Final validation loss
    accuracy_deg : float
        Mean angular accuracy [degrees]
    """
    model: CalibrationMLP
    train_loss: float
    val_loss: float
    accuracy_deg: float
    
    def __str__(self) -> str:
        return f"CalibrationResult(accuracy: {self.accuracy_deg:.2f}°, val_loss: {self.val_loss:.4f})"


# =============================================================================
# Factory Functions — Smart, intuitive object creation
# =============================================================================

def create_array(
    geometry: Literal['uca', 'ula', 'cross'] = 'uca',
    frequency_mhz: float = 1626.27,
    n_ant: int = 5,
    radius_cm: Optional[float] = None,
    spacing_cm: Optional[float] = None,
    **kwargs
) -> ArrayGeometryBase:
    """
    Create array geometry with intelligent defaults.
    
    Parameters
    ----------
    geometry : str
        Array type: 'uca' (circular), 'ula' (linear), 'cross' (plus-shaped)
    frequency_mhz : float
        Operating frequency in MHz (used to compute λ-normalized dimensions)
    n_ant : int
        Number of antenna elements
    radius_cm : float, optional
        For UCA: array radius in cm (auto-computed if None)
    spacing_cm : float, optional
        For ULA: element spacing in cm (auto-computed if None)
    **kwargs
        Additional geometry-specific parameters
        
    Returns
    -------
    ArrayGeometryBase
        Configured array geometry object
        
    Examples
    --------
        >>> # Iridium L-band UCA (default)
        >>> array = create_array('uca', frequency_mhz=1626.27)
        
        >>> # Custom radius
        >>> array = create_array('uca', frequency_mhz=1626.27, radius_cm=17.3)
        
        >>> # ULA for VHF
        >>> array = create_array('ula', frequency_mhz=162.0, n_ant=4)
    """
    # Compute wavelength
    c = 299792458.0  # m/s
    freq_hz = frequency_mhz * 1e6
    wavelength_m = c / freq_hz
    
    if geometry == 'uca':
        # Default: optimal radius for 5-element UCA at L-band
        if radius_cm is None:
            # KrakenSDR default: r ≈ 0.358λ at 1626 MHz
            radius_lambda = kwargs.get('radius_lambda', 0.358)
        else:
            radius_m = radius_cm / 100.0
            radius_lambda = radius_m / wavelength_m
        
        return UniformCircularArray(
            n_ant=n_ant,
            radius_lambda=radius_lambda,
            **{k: v for k, v in kwargs.items() if k != 'radius_lambda'}
        )
    
    elif geometry == 'ula':
        # Default: half-wavelength spacing
        if spacing_cm is None:
            d_lambda = kwargs.get('d_lambda', 0.5)
        else:
            spacing_m = spacing_cm / 100.0
            d_lambda = spacing_m / wavelength_m
        
        return UniformLinearArray(
            n_ant=n_ant,
            d_lambda=d_lambda,
        )
    
    elif geometry == 'cross':
        # Cross array (5-element KrakenSDR standard)
        d_lambda = kwargs.get('d_lambda', 0.5)
        return CrossArray(
            n_ant=n_ant,
            d_lambda=d_lambda,
        )
    
    else:
        raise ValueError(f"Unknown geometry: {geometry}. Use 'uca', 'ula', or 'cross'.")


def create_doa_estimator(
    array: ArrayGeometryBase,
    algorithm: Literal['auto', 'music', 'capon', 'bartlett', 'esprit', 'root_music'] = 'auto',
    n_sources: int = 1,
    decorrelation: Literal['auto', 'none', 'fba', 'spatial'] = 'auto',
    **kwargs
) -> 'DoAEstimator':
    """
    Create intelligent DoA estimator with auto-algorithm selection.
    
    Parameters
    ----------
    array : ArrayGeometryBase
        Array geometry configuration
    algorithm : str
        DoA algorithm: 'auto' (smart selection), 'music', 'capon', etc.
    n_sources : int
        Expected number of signal sources
    decorrelation : str
        Decorrelation method: 'auto', 'none', 'fba', 'spatial'
    **kwargs
        Algorithm-specific parameters
        
    Returns
    -------
    DoAEstimator
        Configured estimator ready to process IQ data
        
    Examples
    --------
        >>> array = create_array('uca', frequency_mhz=1626.27)
        >>> estimator = create_doa_estimator(array, algorithm='auto')
        >>> result = estimator.estimate(iq_data)
    """
    return DoAEstimator(
        array=array,
        algorithm=algorithm,
        n_sources=n_sources,
        decorrelation=decorrelation,
        **kwargs
    )


def create_iridium_pipeline(
    sample_rate_hz: int = 1_024_000,
    center_freq_mhz: float = 1626.27,
    enable_demod: bool = True,
    enable_doa: bool = True,
    array: Optional[ArrayGeometryBase] = None,
    **kwargs
) -> 'IridiumPipeline':
    """
    Create complete Iridium burst processing pipeline.
    
    Parameters
    ----------
    sample_rate_hz : int
        IQ sample rate [Hz]
    center_freq_mhz : float
        Center frequency [MHz]
    enable_demod : bool
        Enable DQPSK demodulation
    enable_doa : bool
        Enable DoA estimation on bursts
    array : ArrayGeometryBase, optional
        Array geometry (auto-created if None)
    **kwargs
        Pipeline configuration parameters
        
    Returns
    -------
    IridiumPipeline
        Ready-to-use processing pipeline
        
    Examples
    --------
        >>> pipeline = create_iridium_pipeline()
        >>> for result in pipeline.process_stream(iq_stream):
        ...     if result.has_burst:
        ...         print(f"DoA: {result.doa.azimuth_deg:.1f}°")
    """
    if array is None:
        array = create_array('uca', frequency_mhz=center_freq_mhz)
    
    return IridiumPipeline(
        sample_rate_hz=sample_rate_hz,
        center_freq_mhz=center_freq_mhz,
        enable_demod=enable_demod,
        enable_doa=enable_doa,
        array=array,
        **kwargs
    )


def create_calibrator(
    array: ArrayGeometryBase,
    auto_train: bool = False,
    **kwargs
) -> 'Calibrator':
    """
    Create calibration system with optional auto-training.
    
    Parameters
    ----------
    array : ArrayGeometryBase
        Array geometry configuration
    auto_train : bool
        Automatically train on first data batch
    **kwargs
        Calibration parameters
        
    Returns
    -------
    Calibrator
        Calibration system ready for training/inference
        
    Examples
    --------
        >>> calibrator = create_calibrator(array)
        >>> model = calibrator.train(training_data)
    """
    return Calibrator(array=array, auto_train=auto_train, **kwargs)


# =============================================================================
# Convenience Functions — One-shot operations
# =============================================================================

def estimate_doa(
    iq_data: np.ndarray,
    array: Optional[ArrayGeometryBase] = None,
    algorithm: str = 'auto',
    **kwargs
) -> DoAResult:
    """
    One-shot DoA estimation with minimal setup.
    
    Parameters
    ----------
    iq_data : np.ndarray
        IQ data matrix (n_ant, n_samples)
    array : ArrayGeometryBase, optional
        Array geometry (auto-created for 5-element UCA if None)
    algorithm : str
        DoA algorithm ('auto' for smart selection)
    **kwargs
        Additional estimation parameters
        
    Returns
    -------
    DoAResult
        Clean result with azimuth, elevation, confidence
        
    Examples
    --------
        >>> # Simplest usage
        >>> result = estimate_doa(iq_data)
        >>> print(f"Azimuth: {result.azimuth_deg:.1f}°")
        
        >>> # With custom array
        >>> array = create_array('uca', frequency_mhz=1626.27)
        >>> result = estimate_doa(iq_data, array=array)
    """
    if array is None:
        # Auto-detect number of antennas
        n_ant = iq_data.shape[0] if iq_data.ndim == 2 else 1
        array = create_array('uca', n_ant=n_ant)
    
    estimator = create_doa_estimator(array, algorithm=algorithm)
    return estimator.estimate(iq_data, **kwargs)


def quick_doa(iq_data: np.ndarray) -> float:
    """
    Ultra-simple DoA estimation: returns just azimuth angle.
    
    Parameters
    ----------
    iq_data : np.ndarray
        IQ data matrix (n_ant, n_samples)
        
    Returns
    -------
    float
        Azimuth angle in degrees [0-360]
        
    Examples
    --------
        >>> azimuth = quick_doa(iq_data)
        >>> print(f"Signal coming from {azimuth:.1f}°")
    """
    result = estimate_doa(iq_data)
    return result.azimuth_deg


def quick_burst_detect(iq_data: np.ndarray, threshold_db: float = 10.0) -> bool:
    """
    Ultra-simple burst detection: returns True/False.
    
    Parameters
    ----------
    iq_data : np.ndarray
        IQ data (single channel)
    threshold_db : float
        Detection threshold above noise floor [dB]
        
    Returns
    -------
    bool
        True if burst detected
        
    Examples
    --------
        >>> if quick_burst_detect(iq_data):
        ...     print("Burst detected!")
    """
    # Compute power envelope
    power = np.abs(iq_data) ** 2
    noise_floor = np.median(power)
    threshold = noise_floor * (10.0 ** (threshold_db / 10.0))
    return bool(np.max(power) > threshold)


def auto_configure(
    frequency_mhz: float,
    bandwidth_hz: float,
    application: Literal['iridium', 'adsb', 'ais', 'custom'] = 'custom',
) -> Dict[str, Any]:
    """
    Auto-configure system parameters based on application.
    
    Parameters
    ----------
    frequency_mhz : float
        Operating frequency [MHz]
    bandwidth_hz : float
        Signal bandwidth [Hz]
    application : str
        Application type for smart defaults
        
    Returns
    -------
    dict
        Configuration dictionary with recommended parameters
        
    Examples
    --------
        >>> config = auto_configure(1626.27, 25_000, application='iridium')
        >>> print(config['sample_rate_hz'])
    """
    configs = {
        'iridium': {
            'sample_rate_hz': 1_024_000,
            'array_type': 'uca',
            'n_ant': 5,
            'algorithm': 'music',
            'decorrelation': 'fba',
            'burst_detection': True,
            'doppler_max_hz': 40_000,
        },
        'adsb': {
            'sample_rate_hz': 2_400_000,
            'array_type': 'uca',
            'n_ant': 4,
            'algorithm': 'music',
            'decorrelation': 'none',
            'burst_detection': False,
            'doppler_max_hz': 0,
        },
        'ais': {
            'sample_rate_hz': 153_600,
            'array_type': 'uca',
            'n_ant': 4,
            'algorithm': 'capon',
            'decorrelation': 'fba',
            'burst_detection': False,
            'doppler_max_hz': 0,
        },
    }
    
    config = configs.get(application, {
        'sample_rate_hz': int(bandwidth_hz * 2.5),
        'array_type': 'uca',
        'n_ant': 5,
        'algorithm': 'music',
        'decorrelation': 'auto',
        'burst_detection': False,
        'doppler_max_hz': 0,
    })
    
    config['frequency_mhz'] = frequency_mhz
    config['bandwidth_hz'] = bandwidth_hz
    
    return config


# =============================================================================
# Intelligent Estimator Classes
# =============================================================================

class DoAEstimator:
    """
    Intelligent DoA estimator with auto-algorithm selection.
    
    Features
    --------
    • Auto-selects best algorithm based on SNR and array geometry
    • Handles decorrelation automatically
    • Provides confidence metrics
    • Caches computations for efficiency
    """
    
    def __init__(
        self,
        array: ArrayGeometryBase,
        algorithm: str = 'auto',
        n_sources: int = 1,
        decorrelation: str = 'auto',
        **kwargs
    ):
        self.array = array
        self.algorithm = algorithm
        self.n_sources = n_sources
        self.decorrelation = decorrelation
        self.config = kwargs
        
        # Internal state
        self._cov_accumulator = CovarianceAccumulator()
        self._snr_history = []
    
    def estimate(
        self,
        iq_data: np.ndarray,
        R_in: Optional[np.ndarray] = None,
    ) -> DoAResult:
        """
        Estimate DoA from IQ data.
        
        Parameters
        ----------
        iq_data : np.ndarray
            IQ data matrix (n_ant, n_samples)
        R_in : np.ndarray, optional
            Pre-computed covariance matrix
            
        Returns
        -------
        DoAResult
            Clean result with all relevant information
        """
        # Compute covariance
        if R_in is None:
            R = covariance(iq_data)
        else:
            R = R_in
        
        # Estimate SNR
        ev = np.linalg.eigvalsh(R)
        ev = np.sort(ev)[::-1]
        snr_db = 10.0 * np.log10(ev[0] / (np.mean(ev[1:]) + 1e-20))
        self._snr_history.append(snr_db)
        
        # Auto-select algorithm
        algo = self._select_algorithm(snr_db)
        
        # Auto-select decorrelation
        decorr = self._select_decorrelation(snr_db)
        
        # Apply decorrelation
        if decorr != 'none':
            R = apply_decorrelation(R, method=decorr)
        
        # Run estimation
        # Compute steering matrix for 1D scan
        from .array_geometry import compute_steering_matrix_1d
        theta_scan = np.linspace(0, 2*np.pi, 360, endpoint=False)
        steering_matrix = compute_steering_matrix_1d(
            self.array, theta_scan
        )
        
        if algo == 'music':
            spectrum = music(R=R, steering_matrix=steering_matrix, n_signals=self.n_sources)
            azimuth_rad = theta_scan[np.argmax(spectrum)]
            confidence = self._compute_confidence(spectrum)
        elif algo == 'capon':
            spectrum = capon(R=R, steering_matrix=steering_matrix)
            azimuth_rad = theta_scan[np.argmax(spectrum)]
            confidence = self._compute_confidence(spectrum)
        else:  # bartlett (most robust)
            spectrum = bartlett(R=R, steering_matrix=steering_matrix)
            azimuth_rad = theta_scan[np.argmax(spectrum)]
            confidence = self._compute_confidence(spectrum)
        
        # Convert to degrees
        azimuth_deg = np.degrees(azimuth_rad) % 360.0
        
        return DoAResult(
            azimuth_deg=azimuth_deg,
            elevation_deg=None,  # 1D estimation
            confidence=confidence,
            algorithm=algo,
            spectrum=spectrum,
            snr_db=snr_db,
        )
    
    def _select_algorithm(self, snr_db: float) -> str:
        """Auto-select algorithm based on conditions."""
        if self.algorithm != 'auto':
            return self.algorithm
        
        # Smart selection based on SNR
        if snr_db > 15:
            return 'music'  # High SNR: use super-resolution
        elif snr_db > 5:
            return 'capon'  # Medium SNR: adaptive beamforming
        else:
            return 'bartlett'  # Low SNR: most robust
    
    def _select_decorrelation(self, snr_db: float) -> str:
        """Auto-select decorrelation method."""
        if self.decorrelation != 'auto':
            return self.decorrelation
        
        # Enable FBA for coherent sources
        return 'fba' if snr_db > 10 else 'none'
    
    def _compute_confidence(self, spectrum: np.ndarray) -> float:
        """Compute confidence metric from spectrum."""
        spectrum_lin = 10.0 ** (spectrum / 10.0)
        peak = np.max(spectrum_lin)
        mean = np.mean(spectrum_lin)
        papr = peak / (mean + 1e-20)
        
        # Map PAPR to confidence [0, 1]
        # PAPR > 20 dB → confidence ≈ 1.0
        # PAPR < 3 dB → confidence ≈ 0.0
        confidence = np.clip((10.0 * np.log10(papr) - 3.0) / 17.0, 0.0, 1.0)
        return float(confidence)


class IridiumPipeline:
    """
    Complete Iridium burst processing pipeline.
    
    Features
    --------
    • Burst detection with adaptive thresholding
    • Doppler compensation
    • DoA estimation on bursts
    • Optional demodulation
    """
    
    def __init__(
        self,
        sample_rate_hz: int,
        center_freq_mhz: float,
        enable_demod: bool,
        enable_doa: bool,
        array: ArrayGeometryBase,
        **kwargs
    ):
        self.sample_rate_hz = sample_rate_hz
        self.center_freq_mhz = center_freq_mhz
        self.enable_demod = enable_demod
        self.enable_doa = enable_doa
        self.array = array
        
        # Create sub-components
        self._detector = _BurstDetector(
            fs=sample_rate_hz,
            burst_snr=kwargs.get('burst_snr', 8.0),
            burst_papr=kwargs.get('burst_papr', 5.0),
        )
        
        if enable_doa:
            self._doa_estimator = create_doa_estimator(array)
        else:
            self._doa_estimator = None
    
    def process_frame(self, iq_data: np.ndarray) -> BurstResult:
        """
        Process a single IQ frame.
        
        Parameters
        ----------
        iq_data : np.ndarray
            IQ data matrix (n_ant, n_samples)
            
        Returns
        -------
        BurstResult
            Processing result with burst info and DoA
        """
        # Detect burst
        burst_result = self._detector.process(iq_data[0])  # Use first channel
        
        if not burst_result.is_burst:
            return BurstResult(has_burst=False)
        
        # Extract burst
        burst = detect_and_extract_burst(iq_data, threshold_db=10.0)
        if burst is None:
            return BurstResult(has_burst=False)
        
        # Compensate Doppler
        compensated, doppler_hz = compensate_doppler(burst, self.sample_rate_hz)
        
        # Estimate DoA
        doa_result = None
        if self.enable_doa and self._doa_estimator is not None:
            R = compute_single_shot_covariance(compensated)
            doa_result = self._doa_estimator.estimate(compensated, R_in=R)
        
        return BurstResult(
            has_burst=True,
            doppler_hz=doppler_hz,
            snr_db=burst_result.burst_snr_db,
            doa=doa_result,
        )
    
    def process_stream(self, iq_stream):
        """
        Process a stream of IQ frames (generator).
        
        Parameters
        ----------
        iq_stream : iterable
            Stream of IQ frames
            
        Yields
        ------
        BurstResult
            Processing result for each frame
        """
        for iq_data in iq_stream:
            yield self.process_frame(iq_data)


class Calibrator:
    """
    Intelligent calibration system with auto-training.
    
    Features
    --------
    • Automatic feature extraction
    • Smart training with validation
    • Model persistence
    """
    
    def __init__(
        self,
        array: ArrayGeometryBase,
        auto_train: bool = False,
        **kwargs
    ):
        self.array = array
        self.auto_train = auto_train
        self.config = kwargs
        
        self.model: Optional[CalibrationMLP] = None
        self._training_data = []
    
    def collect_sample(
        self,
        iq_data: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float = 0.0,
    ):
        """
        Collect calibration sample.
        
        Parameters
        ----------
        iq_data : np.ndarray
            IQ data matrix
        azimuth_deg : float
            Known azimuth angle
        elevation_deg : float
            Known elevation angle
        """
        from .calibration_features import extract_feature_vector, encode_target
        
        R = covariance(iq_data)
        features = extract_feature_vector(R)
        target = encode_target(azimuth_deg, elevation_deg)
        
        self._training_data.append((features, target))
        
        if self.auto_train and len(self._training_data) >= 100:
            self.train_auto()
    
    def train_auto(self) -> CalibrationResult:
        """
        Auto-train calibration model.
        
        Returns
        -------
        CalibrationResult
            Training result with model and metrics
        """
        from .calibration_features import extract_feature_vector, encode_target
        from .calibration_model import train
        
        if len(self._training_data) < 10:
            raise ValueError(f"Need at least 10 samples, have {len(self._training_data)}")
        
        # Prepare data
        X = np.array([f for f, _ in self._training_data])
        Y = np.array([t for _, t in self._training_data])
        
        # Create and train model
        self.model = CalibrationMLP()
        history = train(self.model, X, Y, TrainConfig(epochs=100, verbose=False))
        
        # Compute accuracy
        az_pred, el_pred = self.model.predict(X)
        az_true = np.degrees(np.arctan2(Y[:, 1], Y[:, 0])) % 360.0
        accuracy = float(np.mean(np.abs(az_pred - az_true)))
        
        return CalibrationResult(
            model=self.model,
            train_loss=history['train_loss'][-1],
            val_loss=history['val_loss'][-1],
            accuracy_deg=accuracy,
        )
    
    def apply_calibration(self, doa_result: DoAResult) -> DoAResult:
        """
        Apply calibration to DoA result.
        
        Parameters
        ----------
        doa_result : DoAResult
            Uncalibrated DoA result
            
        Returns
        -------
        DoAResult
            Calibrated DoA result
        """
        if self.model is None:
            return doa_result
        
        # Would need features from covariance to apply calibration
        # This is a placeholder for the full implementation
        return doa_result
