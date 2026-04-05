"""
Phase 4: Modality-Specific Audio Preprocessing.
Supports advanced torchaudio spectrogram extraction, resampling, and SpecAugment logic.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchaudio
    import torchaudio.transforms as AT
except Exception:
    torchaudio = None
    AT = None

try:
    import librosa
except Exception:
    librosa = None


# -----------------------------------------------------------------------------
# 1. WAVEFORM MANIPULATION & RESAMPLING
# -----------------------------------------------------------------------------

def load_canonical_audio(file_path: str, target_sr: int = 16000, max_duration_sec: float = 5.0) -> torch.Tensor:
    """
    Loads raw standard waveform from disk, forces downmix to Mono, resamples dynamically
    to `target_sr` via Native PyTorch FIR filters, and pads or truncates length.
    
    Returns:
        Tensor of shape (1, samples)
    """
    if torchaudio is None:
        raise ImportError("pip install torchaudio required for Audio Subsystem.")
        
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Force Mono Channel Collapse
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    # Standardised Canonical Frequency Resampling
    if sample_rate != target_sr:
        resampler = AT.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # Amplitude Standardisation Scaling [-1, 1]
    waveform = waveform / (waveform.abs().max() + 1e-8)
    
    # Constant Duration Crop/Pad Frame Limiting
    target_samples = int(target_sr * max_duration_sec)
    current_samples = waveform.shape[1]
    
    if current_samples < target_samples:
        pad_size = target_samples - current_samples
        waveform = F.pad(waveform, (0, pad_size))  # Zero-Pad right-axis
    else:
        waveform = waveform[:, :target_samples]
        
    return waveform


# -----------------------------------------------------------------------------
# 2. FEATURE EXTRACTION ALGORITHMS
# -----------------------------------------------------------------------------

def get_mfcc_extractor(sample_rate: int = 16000, n_mfcc: int = 40):
    """
    Returns configured Mel-Frequency Cepstral Coefficient Generator.
    """
    if AT is None:
        raise ImportError("torchaudio required.")
        
    return AT.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 80,
            "center": False,
        }
    )


def extract_delta_features(mfccs: torch.Tensor) -> torch.Tensor:
    """
    Computes Velocity (D1) and Acceleration (D2) kinematics of MFCCs.
    Returns: 
        Concat Tensor (1, n_mfcc*3, T)
    """
    delta_tf = AT.ComputeDeltas()
    d1 = delta_tf(mfccs)
    d2 = delta_tf(d1)
    
    return torch.cat([mfccs, d1, d2], dim=1)


def get_log_mel_spectrogram_extractor(sample_rate: int = 16000, n_mels: int = 128):
    """
    Returns sequential builder mapping (1, T) waveforms -> dB-Scale (Power) Mel-Spectrograms.
    """
    if AT is None:
        raise ImportError("torchaudio required.")
        
    return torch.nn.Sequential(
        AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0,
            window_fn=torch.hann_window,
        ),
        AT.AmplitudeToDB(stype="power", top_db=80.0)
    )


def get_stft_extractor():
    """Returns pure raw Phase-Inclusive Short-Time Fourier Transformer block."""
    if AT is None:
        raise ImportError("torchaudio required.")
        
    return AT.Spectrogram(
        n_fft=1024,
        hop_length=256,
        power=None
    )


# -----------------------------------------------------------------------------
# 3. ADVANCED SIGNAL AUGMENTATIONS (SPECAUGMENT + SCIPY WAVES)
# -----------------------------------------------------------------------------

def specaugment(log_mel: torch.Tensor, freq_mask_param: int = 15, time_mask_param: int = 35, passes: int = 2) -> torch.Tensor:
    """
    Mathematical SpecAugment execution directly on Spectrogram Tensor boundaries.
    Args:
        log_mel: (1, n_mels, T)
        passes: Dual-masking count execution loop.
    """
    f_mask = AT.FrequencyMasking(freq_mask_param=freq_mask_param)
    t_mask = AT.TimeMasking(time_mask_param=time_mask_param)
    
    aug_mel = log_mel.clone()
    for _ in range(passes):
        aug_mel = f_mask(aug_mel)
        aug_mel = t_mask(aug_mel)
        
    return aug_mel


def augment_audio_waveform(waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Applies aggressive additive White Noise, Librosa Shift, and Pitch mutations recursively over signals.
    Args:
        waveform: Float tensor (1, T)
    """
    if librosa is None:
        raise ImportError("pip install librosa to use waveform mutators.")
        
    wav_np = waveform.numpy()[0]
    
    if np.random.random() < 0.5:
        rate = np.random.uniform(0.9, 1.1)
        wav_np = librosa.effects.time_stretch(y=wav_np, rate=rate)
        
    if np.random.random() < 0.5:
        steps = np.random.uniform(-1.0, 1.0)
        wav_np = librosa.effects.pitch_shift(y=wav_np, sr=sr, n_steps=steps)
        
    if np.random.random() < 0.3:
        snr = np.random.uniform(15, 30)
        power = (wav_np ** 2).mean()
        noise = np.random.normal(0, np.sqrt(power / (10 ** (snr / 10))), wav_np.shape)
        wav_np = np.clip(wav_np + noise, -1, 1)
        
    return torch.from_numpy(wav_np.astype("float32")).unsqueeze(0)
