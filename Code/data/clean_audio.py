import numpy as np

try:
    from scipy import signal
except ImportError:
    signal = None


def butter_filter(
    waveform_np: np.ndarray, 
    sr: int,
    filter_type: str = "bandpass",
    low_hz: float = 300.0, 
    high_hz: float = 3400.0,
    order: int = 5
) -> np.ndarray:
    """
    Apply a digital Butterworth filter directly to a numpy waveform.
    filter_type: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
    """
    if signal is None:
        raise ImportError("Please install scipy: `pip install scipy` to use butter_filter.")

    nyq = sr / 2.0
    if filter_type == "lowpass":
        sos = signal.butter(order, high_hz / nyq, btype="low", output="sos")
    elif filter_type == "highpass":
        sos = signal.butter(order, low_hz / nyq, btype="high", output="sos")
    elif filter_type == "bandpass":
        sos = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    elif filter_type == "bandstop":
        sos = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="bandstop", output="sos")
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    
    return signal.sosfiltfilt(sos, waveform_np)   # zero-phase filtering


def noise_gate(waveform_np: np.ndarray, threshold_db: float = -40) -> np.ndarray:
    """Zero out samples below threshold_db relative to peak amplitude."""
    peak = np.max(np.abs(waveform_np))
    threshold = peak * (10 ** (threshold_db / 20))
    return np.where(np.abs(waveform_np) < threshold, 0.0, waveform_np)
