"""
Language System Utilities: Common utilities for language modules.
Provides error handling, dependency checking, and audio processing helpers.
"""

from typing import Optional, Callable, Any
import functools
import numpy as np
from config import SAMPLE_RATE


# Dependency availability flags
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None


def check_dependency(module_name: str, available: bool) -> bool:
    """
    Check if a dependency is available.
    
    Args:
        module_name: Name of the module (for error messages)
        available: Whether the module is available
        
    Returns:
        True if available, False otherwise
    """
    if not available:
        print(f"[LANGUAGE] {module_name} not available. Install with: pip install {module_name}")
    return available


def safe_initialize(init_func: Callable, fallback_value: Any = None, error_message: str = "") -> Any:
    """
    Safely initialize a component with error handling.
    
    Args:
        init_func: Function to call for initialization
        fallback_value: Value to return on failure
        error_message: Error message prefix
        
    Returns:
        Initialized object or fallback_value on failure
    """
    try:
        return init_func()
    except Exception as e:
        if error_message:
            print(f"[LANGUAGE] {error_message}: {e}")
        return fallback_value


def convert_audio_bytes_to_array(audio_bytes: bytes, sample_width: int = 2, 
                                 dtype: type = np.int16) -> np.ndarray:
    """
    Convert audio bytes to numpy array.
    
    Args:
        audio_bytes: Raw audio bytes
        sample_width: Bytes per sample (1=8-bit, 2=16-bit, 4=32-bit)
        dtype: NumPy dtype for conversion
        
    Returns:
        numpy array of audio samples (float32, normalized to [-1, 1])
    """
    if sample_width == 2:  # 16-bit
        audio = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32) / 32768.0
    elif sample_width == 4:  # 32-bit
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    elif sample_width == 1:  # 8-bit
        audio = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    return audio


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio samples (numpy array)
        source_rate: Source sample rate
        target_rate: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if source_rate == target_rate:
        return audio
    
    try:
        from scipy import signal
        num_samples = int(len(audio) * target_rate / source_rate)
        return signal.resample(audio, num_samples)
    except ImportError:
        # Simple linear interpolation fallback
        ratio = target_rate / source_rate
        indices = np.arange(0, len(audio), 1/ratio)
        indices = indices[indices < len(audio)]
        return np.interp(indices, np.arange(len(audio)), audio)


def create_audio_envelope(num_samples: int, sample_rate: int = SAMPLE_RATE,
                          attack_ms: float = 10, decay_ms: float = 20) -> np.ndarray:
    """
    Create amplitude envelope (attack, sustain, decay).
    
    Args:
        num_samples: Number of samples
        sample_rate: Sample rate
        attack_ms: Attack time in milliseconds
        decay_ms: Decay time in milliseconds
        
    Returns:
        Envelope array (0.0 to 1.0)
    """
    attack_samples = int(sample_rate * attack_ms / 1000.0)
    decay_samples = int(sample_rate * decay_ms / 1000.0)
    sustain_samples = num_samples - attack_samples - decay_samples
    
    if sustain_samples < 0:
        # Very short sound: just attack and decay
        attack_samples = num_samples // 2
        decay_samples = num_samples - attack_samples
        sustain_samples = 0
    
    envelope = np.ones(num_samples)
    
    # Attack (linear rise)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Sustain (constant)
    if sustain_samples > 0:
        envelope[attack_samples:attack_samples + sustain_samples] = 1.0
    
    # Decay (exponential fall)
    if decay_samples > 0:
        decay_start = attack_samples + sustain_samples
        envelope[decay_start:] = np.exp(-np.linspace(0, 5, decay_samples))
    
    return envelope


def generate_formant_resonance(t: np.ndarray, frequency: float, bandwidth: float, 
                               amplitude: float = 1.0) -> np.ndarray:
    """
    Generate a single formant resonance (damped sine wave).
    
    Args:
        t: Time array
        frequency: Formant frequency (Hz)
        bandwidth: Formant bandwidth (Hz)
        amplitude: Amplitude (0.0 to 1.0)
        
    Returns:
        Formant signal array
    """
    omega = 2 * np.pi * frequency
    alpha = bandwidth / 2.0  # Damping factor
    
    # Damped oscillation
    signal = amplitude * np.exp(-alpha * t) * np.sin(omega * t)
    
    return signal


def requires_dependency(dependency_name: str, available: bool):
    """
    Decorator to require a dependency for a method.
    
    Args:
        dependency_name: Name of required dependency
        available: Whether dependency is available
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not available:
                print(f"[LANGUAGE] {dependency_name} required for {func.__name__}")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

