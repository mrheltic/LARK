"""
Smart API Examples — LEGO-like KrakenSDR usage
==============================================

Complete examples showing how to use the new intelligent, modular API.

All examples are designed to be:
    • Copy-paste ready
    • Self-contained
    • Easy to understand
    • Hard to break
"""

# =============================================================================
# Example 1: Simplest DoA Estimation
# =============================================================================
"""
One-liner DoA estimation for Iridium signals.
"""

import numpy as np
from core import estimate_doa, quick_doa

# Generate simulated IQ data (5 antennas, 10000 samples)
np.random.seed(42)
iq_data = np.random.randn(5, 10000) + 1j * np.random.randn(5, 10000)

# Method 1: Full result with metadata
result = estimate_doa(iq_data)
print(f"Azimuth: {result.azimuth_deg:.1f}°")
print(f"SNR: {result.snr_db:.1f} dB")
print(f"Confidence: {result.confidence:.2f}")
print(f"Algorithm: {result.algorithm}")

# Method 2: Ultra-simple (just the angle)
azimuth = quick_doa(iq_data)
print(f"Signal coming from {azimuth:.1f}°")


# =============================================================================
# Example 2: Custom Array Configuration
# =============================================================================
"""
Create array geometry with custom parameters.
"""

from core import create_array, estimate_doa

# Iridium L-band UCA with default radius
array = create_array('uca', frequency_mhz=1626.27)
print(f"Created {array.n_ant}-element UCA")

# Custom radius (17.3 cm)
array_custom = create_array(
    'uca',
    frequency_mhz=1626.27,
    radius_cm=17.3
)

# ULA for VHF (4 elements, half-wavelength spacing)
array_ula = create_array(
    'ula',
    frequency_mhz=162.0,
    n_ant=4
)

# Estimate DoA with custom array
result = estimate_doa(iq_data, array=array_custom)


# =============================================================================
# Example 3: Intelligent DoA Estimator
# =============================================================================
"""
Create estimator with auto-algorithm selection.
"""

from core import create_array, create_doa_estimator

# Create array
array = create_array('uca', frequency_mhz=1626.27)

# Create estimator with auto-selection
estimator = create_doa_estimator(
    array,
    algorithm='auto',  # Auto-select best algorithm
    n_sources=1,
    decorrelation='auto',  # Auto-select decorrelation
)

# Process multiple frames
for frame_idx in range(10):
    # Simulate IQ data
    iq_data = np.random.randn(5, 10000) + 1j * np.random.randn(5, 10000)
    
    # Estimate DoA
    result = estimator.estimate(iq_data)
    
    print(f"Frame {frame_idx}: Az={result.azimuth_deg:.1f}°, "
          f"SNR={result.snr_db:.1f} dB, "
          f"Algo={result.algorithm}")


# =============================================================================
# Example 4: Complete Iridium Pipeline
# =============================================================================
"""
Full Iridium burst detection + DoA estimation pipeline.
"""

from core import create_iridium_pipeline

# Create pipeline with smart defaults
pipeline = create_iridium_pipeline(
    sample_rate_hz=1_024_000,
    center_freq_mhz=1626.27,
    enable_demod=True,
    enable_doa=True,
)

# Process IQ stream
def iq_stream():
    """Simulate IQ data stream."""
    for i in range(100):
        yield np.random.randn(5, 131072) + 1j * np.random.randn(5, 131072)

# Process frames
for result in pipeline.process_stream(iq_stream()):
    if result.has_burst:
        print(f"Burst detected!")
        print(f"  Doppler: {result.doppler_hz/1e3:.1f} kHz")
        print(f"  SNR: {result.snr_db:.1f} dB")
        
        if result.doa:
            print(f"  DoA: {result.doa.azimuth_deg:.1f}°")


# =============================================================================
# Example 5: Calibration System
# =============================================================================
"""
Train and use calibration model.
"""

from core import create_array, create_calibrator

# Create array
array = create_array('uca', frequency_mhz=1626.27)

# Create calibrator
calibrator = create_calibrator(array, auto_train=False)

# Collect training samples
for azimuth in range(0, 360, 10):
    # Simulate IQ data for known angle
    iq_data = np.random.randn(5, 10000) + 1j * np.random.randn(5, 10000)
    
    # Collect sample
    calibrator.collect_sample(iq_data, azimuth_deg=azimuth)

# Train model
result = calibrator.train_auto()
print(f"Training complete!")
print(f"  Accuracy: {result.accuracy_deg:.2f}°")
print(f"  Val loss: {result.val_loss:.4f}")


# =============================================================================
# Example 6: Auto-Configuration
# =============================================================================
"""
Auto-configure system for different applications.
"""

from core import auto_configure, create_array, create_doa_estimator

# Iridium configuration
iridium_config = auto_configure(
    frequency_mhz=1626.27,
    bandwidth_hz=25_000,
    application='iridium'
)
print("Iridium config:", iridium_config)

# Create components from config
array = create_array(
    iridium_config['array_type'],
    frequency_mhz=iridium_config['frequency_mhz'],
    n_ant=iridium_config['n_ant']
)

estimator = create_doa_estimator(
    array,
    algorithm=iridium_config['algorithm'],
    decorrelation=iridium_config['decorrelation']
)


# =============================================================================
# Example 7: Burst Detection Only
# =============================================================================
"""
Simple burst detection without DoA.
"""

from core import quick_burst_detect

# Simulate IQ data
iq_data = np.random.randn(10000) + 1j * np.random.randn(10000)

# Detect burst
if quick_burst_detect(iq_data, threshold_db=10.0):
    print("Burst detected!")
else:
    print("No burst")


# =============================================================================
# Example 8: Multi-Source DoA
# =============================================================================
"""
Estimate DoA for multiple sources.
"""

from core import create_array, create_doa_estimator

array = create_array('uca', frequency_mhz=1626.27)

# Create estimator for 2 sources
estimator = create_doa_estimator(
    array,
    algorithm='music',  # MUSIC handles multiple sources
    n_sources=2,
)

# Estimate
iq_data = np.random.randn(5, 10000) + 1j * np.random.randn(5, 10000)
result = estimator.estimate(iq_data)

print(f"Azimuth: {result.azimuth_deg:.1f}°")
print(f"Confidence: {result.confidence:.2f}")


# =============================================================================
# Example 9: Real-Time Processing Loop
# =============================================================================
"""
Real-time processing with adaptive parameters.
"""

from core import create_iridium_pipeline, create_array
import time

# Create pipeline
array = create_array('uca', frequency_mhz=1626.27)
pipeline = create_iridium_pipeline(
    sample_rate_hz=1_024_000,
    center_freq_mhz=1626.27,
    enable_doa=True,
    array=array,
)

# Simulate real-time stream
def realtime_stream():
    """Simulate real-time IQ data."""
    frame_idx = 0
    while True:
        # In real usage, this would read from SDR
        yield np.random.randn(5, 131072) + 1j * np.random.randn(5, 131072)
        frame_idx += 1
        time.sleep(0.1)  # 10 fps

# Process
for result in pipeline.process_stream(realtime_stream()):
    if result.has_burst:
        print(f"[{time.time():.1f}] Burst: "
              f"Dop={result.doppler_hz/1e3:.1f} kHz, "
              f"SNR={result.snr_db:.1f} dB")
        
        if result.doa:
            print(f"  → DoA: {result.doa.azimuth_deg:.1f}° "
                  f"(conf={result.doa.confidence:.2f})")


# =============================================================================
# Example 10: Complete Application
# =============================================================================
"""
Complete Iridium tracking application.
"""

from core import (
    create_array,
    create_iridium_pipeline,
    create_calibrator,
)
import numpy as np

class IridiumTracker:
    """Complete Iridium satellite tracker."""
    
    def __init__(self):
        # Create array
        self.array = create_array('uca', frequency_mhz=1626.27)
        
        # Create pipeline
        self.pipeline = create_iridium_pipeline(
            sample_rate_hz=1_024_000,
            center_freq_mhz=1626.27,
            enable_doa=True,
            array=self.array,
        )
        
        # Create calibrator
        self.calibrator = create_calibrator(self.array)
        
        # Statistics
        self.burst_count = 0
        self.doa_history = []
    
    def process_frame(self, iq_data):
        """Process one IQ frame."""
        result = self.pipeline.process_frame(iq_data)
        
        if result.has_burst:
            self.burst_count += 1
            
            if result.doa:
                # Apply calibration if available
                calibrated = self.calibrator.apply_calibration(result.doa)
                self.doa_history.append(calibrated.azimuth_deg)
                
                return {
                    'burst': True,
                    'doppler_hz': result.doppler_hz,
                    'snr_db': result.snr_db,
                    'azimuth_deg': calibrated.azimuth_deg,
                    'confidence': calibrated.confidence,
                }
        
        return {'burst': False}
    
    def get_statistics(self):
        """Get tracking statistics."""
        if not self.doa_history:
            return None
        
        return {
            'total_bursts': self.burst_count,
            'mean_azimuth': np.mean(self.doa_history),
            'std_azimuth': np.std(self.doa_history),
        }


# Use the tracker
tracker = IridiumTracker()

# Process some frames
for i in range(10):
    iq_data = np.random.randn(5, 131072) + 1j * np.random.randn(5, 131072)
    result = tracker.process_frame(iq_data)
    
    if result['burst']:
        print(f"Frame {i}: Az={result['azimuth_deg']:.1f}°")

# Get statistics
stats = tracker.get_statistics()
if stats:
    print(f"\nStatistics:")
    print(f"  Total bursts: {stats['total_bursts']}")
    print(f"  Mean azimuth: {stats['mean_azimuth']:.1f}°")
    print(f"  Std azimuth: {stats['std_azimuth']:.1f}°")


# =============================================================================
# Summary: API Design Principles
# =============================================================================
"""
The Smart API follows these principles:

1. ONE-LINER SETUP
   Most common operations require just one line:
   >>> result = estimate_doa(iq_data)

2. SENSIBLE DEFAULTS
   90% of use cases work with defaults:
   >>> array = create_array('uca', frequency_mhz=1626.27)

3. AUTO-CONFIGURATION
   System auto-selects optimal parameters:
   >>> estimator = create_doa_estimator(array, algorithm='auto')

4. COMPOSABILITY
   Components work like LEGO bricks:
   >>> pipeline = create_iridium_pipeline(array=array)

5. INTELLIGENT BEHAVIOR
   System adapts to conditions:
   - Auto-selects algorithm based on SNR
   - Auto-enables decorrelation when needed
   - Auto-adjusts thresholds

6. CLEAN OUTPUTS
   Results are informative and easy to use:
   >>> print(result.azimuth_deg)
   >>> print(result.confidence)

7. BACKWARD COMPATIBLE
   Old API still works:
   >>> from core import doa_music  # Legacy API
"""
