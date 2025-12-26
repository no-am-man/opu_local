"""
Tests for core/language_utils.py - Language system utilities.
Targets 100% code coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from core.language_utils import (
    check_dependency,
    safe_initialize,
    convert_audio_bytes_to_array,
    resample_audio,
    create_audio_envelope,
    generate_formant_resonance,
    requires_dependency,
    TTS_AVAILABLE,
    SPEECH_RECOGNITION_AVAILABLE,
    WHISPER_AVAILABLE
)


class TestCheckDependency:
    """Tests for check_dependency function."""
    
    def test_check_dependency_available(self):
        """Test checking available dependency."""
        with patch('builtins.print') as mock_print:
            result = check_dependency('test_module', True)
            assert result is True
            mock_print.assert_not_called()
    
    def test_check_dependency_unavailable(self):
        """Test checking unavailable dependency."""
        with patch('builtins.print') as mock_print:
            result = check_dependency('test_module', False)
            assert result is False
            mock_print.assert_called_once()
            assert 'test_module' in str(mock_print.call_args)


class TestSafeInitialize:
    """Tests for safe_initialize function."""
    
    def test_safe_initialize_success(self):
        """Test successful initialization."""
        def init_func():
            return 'success'
        
        result = safe_initialize(init_func)
        assert result == 'success'
    
    def test_safe_initialize_with_fallback(self):
        """Test initialization with fallback on failure."""
        def init_func():
            raise ValueError("Test error")
        
        result = safe_initialize(init_func, fallback_value='fallback')
        assert result == 'fallback'
    
    def test_safe_initialize_with_error_message(self):
        """Test initialization with error message."""
        def init_func():
            raise RuntimeError("Test error")
        
        with patch('builtins.print') as mock_print:
            result = safe_initialize(
                init_func, 
                fallback_value=None,
                error_message="Initialization failed"
            )
            assert result is None
            mock_print.assert_called_once()
            assert 'Initialization failed' in str(mock_print.call_args)
    
    def test_safe_initialize_no_error_message(self):
        """Test initialization without error message."""
        def init_func():
            raise ValueError("Test error")
        
        with patch('builtins.print') as mock_print:
            result = safe_initialize(init_func, fallback_value='fallback')
            assert result == 'fallback'
            mock_print.assert_not_called()


class TestConvertAudioBytesToArray:
    """Tests for convert_audio_bytes_to_array function."""
    
    def test_convert_16bit_audio(self):
        """Test converting 16-bit audio bytes."""
        # Create 16-bit audio samples (2 bytes per sample)
        samples = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        audio_bytes = samples.tobytes()
        
        result = convert_audio_bytes_to_array(audio_bytes, sample_width=2)
        
        assert len(result) == 5
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(-1.0, abs=0.01)
        assert result[2] == pytest.approx(0.0, abs=0.01)
        assert result[4] == pytest.approx(1.0, abs=0.01)
    
    def test_convert_32bit_audio(self):
        """Test converting 32-bit audio bytes."""
        samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        audio_bytes = samples.tobytes()
        
        result = convert_audio_bytes_to_array(audio_bytes, sample_width=4)
        
        assert len(result) == 5
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(-1.0, abs=0.01)
        assert result[2] == pytest.approx(0.0, abs=0.01)
        assert result[4] == pytest.approx(1.0, abs=0.01)
    
    def test_convert_8bit_audio(self):
        """Test converting 8-bit audio bytes."""
        # Create 8-bit audio samples (0-255, centered at 128)
        samples = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        audio_bytes = samples.tobytes()
        
        result = convert_audio_bytes_to_array(audio_bytes, sample_width=1)
        
        assert len(result) == 5
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(-1.0, abs=0.1)
        assert result[2] == pytest.approx(0.0, abs=0.1)
        assert result[4] == pytest.approx(1.0, abs=0.1)
    
    def test_convert_unsupported_sample_width(self):
        """Test converting with unsupported sample width."""
        audio_bytes = b'\x00\x00\x00'
        with pytest.raises(ValueError, match="Unsupported sample width"):
            convert_audio_bytes_to_array(audio_bytes, sample_width=3)
    
    def test_convert_custom_dtype(self):
        """Test converting with custom dtype."""
        samples = np.array([-1000, 0, 1000], dtype=np.int16)
        audio_bytes = samples.tobytes()
        
        result = convert_audio_bytes_to_array(audio_bytes, sample_width=2, dtype=np.int16)
        assert len(result) == 3


class TestResampleAudio:
    """Tests for resample_audio function."""
    
    def test_resample_same_rate(self):
        """Test resampling with same rate (no-op)."""
        audio = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        result = resample_audio(audio, source_rate=44100, target_rate=44100)
        assert np.array_equal(result, audio)
    
    def test_resample_with_scipy(self):
        """Test resampling with scipy available (if installed)."""
        audio = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        
        # Try to use scipy if available, otherwise skip
        try:
            from scipy import signal
            result = resample_audio(audio, source_rate=44100, target_rate=22050)
            assert len(result) > 0
            assert len(result) < len(audio)  # Downsampled
        except ImportError:
            pytest.skip("scipy not available")
    
    def test_resample_without_scipy(self):
        """Test resampling without scipy (fallback)."""
        audio = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        
        # Mock ImportError for scipy import
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'scipy' or name.startswith('scipy.'):
                raise ImportError("No module named 'scipy'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = resample_audio(audio, source_rate=44100, target_rate=22050)
            assert len(result) > 0
            assert len(result) < len(audio)  # Downsampled
    
    def test_resample_downsample(self):
        """Test downsampling audio."""
        audio = np.linspace(0, 1, 100)
        result = resample_audio(audio, source_rate=44100, target_rate=22050)
        assert len(result) < len(audio)
    
    def test_resample_upsample(self):
        """Test upsampling audio."""
        audio = np.linspace(0, 1, 100)
        result = resample_audio(audio, source_rate=22050, target_rate=44100)
        assert len(result) > len(audio)


class TestCreateAudioEnvelope:
    """Tests for create_audio_envelope function."""
    
    def test_create_envelope_basic(self):
        """Test creating basic envelope."""
        envelope = create_audio_envelope(1000, sample_rate=44100)
        assert len(envelope) == 1000
        assert envelope[0] == pytest.approx(0.0, abs=0.1)
        assert envelope[-1] < 1.0  # Decay at end
    
    def test_create_envelope_custom_attack_decay(self):
        """Test creating envelope with custom attack/decay."""
        envelope = create_audio_envelope(
            1000, 
            sample_rate=44100,
            attack_ms=50,
            decay_ms=100
        )
        assert len(envelope) == 1000
        # Attack phase should be rising
        assert envelope[0] < envelope[100]
        # Decay phase should be falling
        assert envelope[-100] > envelope[-1]
    
    def test_create_envelope_very_short(self):
        """Test creating envelope for very short sound."""
        # Sound shorter than attack + decay
        envelope = create_audio_envelope(
            100,
            sample_rate=44100,
            attack_ms=50,
            decay_ms=100
        )
        assert len(envelope) == 100
        # Should still have attack and decay
        assert envelope[0] < envelope[50]
        assert envelope[50] > envelope[-1]
    
    def test_create_envelope_zero_attack(self):
        """Test creating envelope with zero attack."""
        envelope = create_audio_envelope(
            1000,
            sample_rate=44100,
            attack_ms=0,
            decay_ms=20
        )
        assert envelope[0] == pytest.approx(1.0, abs=0.1)
    
    def test_create_envelope_zero_decay(self):
        """Test creating envelope with zero decay."""
        envelope = create_audio_envelope(
            1000,
            sample_rate=44100,
            attack_ms=10,
            decay_ms=0
        )
        # Should maintain high value at end
        assert envelope[-1] > 0.5


class TestGenerateFormantResonance:
    """Tests for generate_formant_resonance function."""
    
    def test_generate_formant_basic(self):
        """Test generating basic formant."""
        t = np.linspace(0, 0.01, 441)  # 10ms at 44.1kHz
        result = generate_formant_resonance(t, frequency=1000.0, bandwidth=100.0)
        
        assert len(result) == 441
        assert result.dtype == np.float64 or result.dtype == np.float32
        # Should oscillate
        assert np.any(result > 0)
        assert np.any(result < 0)
    
    def test_generate_formant_custom_amplitude(self):
        """Test generating formant with custom amplitude."""
        t = np.linspace(0, 0.01, 441)
        result = generate_formant_resonance(
            t, 
            frequency=1000.0, 
            bandwidth=100.0,
            amplitude=0.5
        )
        
        # Amplitude should be reduced
        assert np.max(np.abs(result)) < 0.6
    
    def test_generate_formant_damped(self):
        """Test that formant is damped (decays over time)."""
        t = np.linspace(0, 0.1, 4410)  # 100ms
        result = generate_formant_resonance(t, frequency=1000.0, bandwidth=100.0)
        
        # Should decay over time
        early_max = np.max(np.abs(result[:1000]))
        late_max = np.max(np.abs(result[-1000:]))
        assert late_max < early_max
    
    def test_generate_formant_different_frequencies(self):
        """Test generating formants at different frequencies."""
        t = np.linspace(0, 0.01, 441)
        result1 = generate_formant_resonance(t, frequency=500.0, bandwidth=50.0)
        result2 = generate_formant_resonance(t, frequency=2000.0, bandwidth=200.0)
        
        # Different frequencies should produce different patterns
        assert not np.array_equal(result1, result2)


class TestRequiresDependency:
    """Tests for requires_dependency decorator."""
    
    def test_requires_dependency_available(self):
        """Test decorator when dependency is available."""
        @requires_dependency('test_module', True)
        def test_func():
            return 'success'
        
        result = test_func()
        assert result == 'success'
    
    def test_requires_dependency_unavailable(self):
        """Test decorator when dependency is unavailable."""
        with patch('builtins.print') as mock_print:
            @requires_dependency('test_module', False)
            def test_func():
                return 'should_not_run'
            
            result = test_func()
            assert result is None
            mock_print.assert_called_once()
            assert 'test_module' in str(mock_print.call_args)
            assert 'test_func' in str(mock_print.call_args)
    
    def test_requires_dependency_with_args(self):
        """Test decorator with function arguments."""
        @requires_dependency('test_module', True)
        def test_func(arg1, arg2, kwarg1=None):
            return arg1 + arg2 + (kwarg1 or 0)
        
        result = test_func(1, 2, kwarg1=3)
        assert result == 6
    
    def test_requires_dependency_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        @requires_dependency('test_module', True)
        def test_func():
            """Test function docstring."""
            return 'success'
        
        assert test_func.__name__ == 'test_func'
        assert 'Test function docstring' in test_func.__doc__

