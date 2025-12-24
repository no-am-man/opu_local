"""
Tests for AudioInputHandler class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import time

from core.audio_input_handler import AudioInputHandler
from config import (
    SAMPLE_RATE, CHUNK_SIZE,
    DITHERING_NOISE_SIGMA, BUFFER_DRAIN_MULTIPLIER, BUFFER_READ_MULTIPLIER,
    BUFFER_FULL_THRESHOLD_MULTIPLIER,
    SIMULATED_FREQ_BASE_1, SIMULATED_FREQ_BASE_2, SIMULATED_FREQ_BASE_3,
    SIMULATED_FREQ_MIN_1, SIMULATED_FREQ_MAX_1,
    SIMULATED_FREQ_MIN_2, SIMULATED_FREQ_MAX_2,
    SIMULATED_FREQ_MIN_3, SIMULATED_FREQ_MAX_3
)


class TestAudioInputHandler:
    """Test suite for AudioInputHandler."""
    
    @pytest.fixture
    def mock_afl(self):
        """Create a mock AestheticFeedbackLoop."""
        afl = Mock()
        afl.is_active.return_value = False
        return afl
    
    @pytest.fixture
    def handler(self, mock_afl):
        """Create an AudioInputHandler instance."""
        start_time = time.time()
        return AudioInputHandler(mock_afl, start_time)
    
    def test_init(self, mock_afl):
        """Test AudioInputHandler initialization."""
        start_time = time.time()
        handler = AudioInputHandler(mock_afl, start_time)
        
        assert handler.afl == mock_afl
        assert handler.start_time == start_time
        assert handler.use_microphone is False
        assert handler.audio_stream is None
        assert handler._last_overflow_warn is None
        assert handler.freq_state is None
    
    @patch('core.audio_input_handler.sd.InputStream')
    def test_setup_audio_input_success_low_latency(self, mock_input_stream, handler):
        """Test successful audio input setup with low latency."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream
        mock_stream.start.return_value = None
        
        handler.setup_audio_input()
        
        assert handler.use_microphone is True
        assert handler.audio_stream == mock_stream
        mock_input_stream.assert_called_once()
        mock_stream.start.assert_called_once()
    
    @patch('core.audio_input_handler.sd.InputStream')
    def test_setup_audio_input_fallback_no_latency(self, mock_input_stream, handler):
        """Test audio input setup falls back to no latency mode."""
        mock_stream = MagicMock()
        
        # First call raises exception, second succeeds
        mock_input_stream.side_effect = [
            Exception("Low latency not supported"),
            mock_stream
        ]
        mock_stream.start.return_value = None
        
        handler.setup_audio_input()
        
        assert handler.use_microphone is True
        assert handler.audio_stream == mock_stream
        assert mock_input_stream.call_count == 2
        mock_stream.start.assert_called_once()
    
    @patch('core.audio_input_handler.sd.InputStream')
    def test_setup_audio_input_fallback_to_simulation(self, mock_input_stream, handler):
        """Test audio input setup falls back to simulation when both fail."""
        mock_input_stream.side_effect = [
            Exception("Low latency failed"),
            Exception("Standard mode failed")
        ]
        
        handler.setup_audio_input()
        
        assert handler.use_microphone is False
        assert handler.audio_stream is None
        assert mock_input_stream.call_count == 2
    
    def test_should_use_microphone_false(self, handler):
        """Test _should_use_microphone returns False when microphone not available."""
        assert handler._should_use_microphone() is False
    
    def test_should_use_microphone_true(self, handler):
        """Test _should_use_microphone returns True when microphone available."""
        handler.use_microphone = True
        handler.audio_stream = MagicMock()
        
        assert handler._should_use_microphone() is True
    
    def test_is_opu_speaking(self, handler, mock_afl):
        """Test _is_opu_speaking delegates to AFL."""
        mock_afl.is_active.return_value = True
        assert handler._is_opu_speaking() is True
        
        mock_afl.is_active.return_value = False
        assert handler._is_opu_speaking() is False
    
    def test_generate_dithering_noise(self, handler):
        """Test dithering noise generation."""
        noise = handler._generate_dithering_noise()
        
        assert noise.shape == (CHUNK_SIZE,)
        assert noise.dtype == np.float32
        assert np.allclose(np.std(noise), DITHERING_NOISE_SIGMA, rtol=0.5)
    
    def test_get_audio_input_simulated_when_no_mic(self, handler):
        """Test get_audio_input returns simulated input when microphone not available."""
        handler.use_microphone = False
        
        audio = handler.get_audio_input()
        
        assert audio is not None
        assert len(audio) == CHUNK_SIZE
        assert audio.dtype == np.float32
    
    def test_get_audio_input_feedback_prevention(self, handler, mock_afl):
        """Test get_audio_input handles feedback prevention when OPU is speaking."""
        handler.use_microphone = True
        handler.audio_stream = MagicMock()
        mock_afl.is_active.return_value = True
        
        with patch.object(handler, '_drain_audio_buffer_aggressively') as mock_drain:
            audio = handler.get_audio_input()
            
            mock_drain.assert_called_once()
            assert audio is not None
            assert len(audio) == CHUNK_SIZE
    
    def test_get_audio_input_from_microphone(self, handler):
        """Test get_audio_input reads from microphone when available."""
        handler.use_microphone = True
        mock_stream = MagicMock()
        mock_stream.read_available = CHUNK_SIZE
        handler.audio_stream = mock_stream
        
        mock_data = np.random.randn(CHUNK_SIZE, 1).astype(np.float32)
        mock_stream.read.return_value = (mock_data, False)
        
        audio = handler.get_audio_input()
        
        assert audio is not None
        assert len(audio) == CHUNK_SIZE
    
    def test_is_buffer_full(self, handler):
        """Test _is_buffer_full detection."""
        threshold = CHUNK_SIZE * BUFFER_FULL_THRESHOLD_MULTIPLIER
        
        assert handler._is_buffer_full(threshold + 1) is True
        assert handler._is_buffer_full(threshold) is False
        assert handler._is_buffer_full(threshold - 1) is False
    
    def test_handle_buffer_overflow(self, handler):
        """Test buffer overflow handling."""
        handler.audio_stream = MagicMock()
        handler.audio_stream.read_available = CHUNK_SIZE * 10
        
        with patch.object(handler, '_warn_buffer_draining') as mock_warn, \
             patch.object(handler, '_drain_audio_buffer_aggressively') as mock_drain:
            handler._handle_buffer_overflow(1000)
            
            mock_warn.assert_called_once_with(1000)
            mock_drain.assert_called_once()
    
    def test_warn_buffer_draining(self, handler):
        """Test buffer draining warning."""
        handler._last_overflow_warn = None
        
        with patch('builtins.print') as mock_print:
            handler._warn_buffer_draining(1000)
            mock_print.assert_called_once()
            assert handler._last_overflow_warn is not None
    
    def test_warn_buffer_draining_throttled(self, handler):
        """Test buffer draining warning is throttled."""
        handler._last_overflow_warn = time.time()
        
        with patch('builtins.print') as mock_print:
            handler._warn_buffer_draining(1000)
            mock_print.assert_not_called()
    
    def test_drain_audio_buffer_aggressively(self, handler):
        """Test aggressive audio buffer draining."""
        mock_stream = MagicMock()
        mock_stream.read_available = CHUNK_SIZE * 5
        handler.audio_stream = mock_stream
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                handler.audio_stream.read_available = CHUNK_SIZE - 1
            return (np.zeros((CHUNK_SIZE, 1)), False)
        
        mock_stream.read.side_effect = side_effect
        
        handler._drain_audio_buffer_aggressively()
        
        assert mock_stream.read.called
    
    def test_read_audio_with_fallback_success(self, handler):
        """Test reading audio with successful first attempt."""
        mock_stream = MagicMock()
        mock_data = (np.random.randn(CHUNK_SIZE, 1), False)
        mock_stream.read.return_value = mock_data
        handler.audio_stream = mock_stream
        
        result = handler._read_audio_with_fallback(CHUNK_SIZE)
        
        assert result == mock_data
        mock_stream.read.assert_called_once_with(CHUNK_SIZE, blocking=False)
    
    def test_read_audio_with_fallback_retry(self, handler):
        """Test reading audio with fallback to smaller chunk."""
        mock_stream = MagicMock()
        mock_data = (np.random.randn(CHUNK_SIZE, 1), False)
        mock_stream.read.side_effect = [
            Exception("First attempt failed"),
            mock_data
        ]
        handler.audio_stream = mock_stream
        
        result = handler._read_audio_with_fallback(CHUNK_SIZE * 2)
        
        assert result == mock_data
        assert mock_stream.read.call_count == 2
    
    def test_read_audio_with_fallback_returns_none(self, handler):
        """Test reading audio returns None when all attempts fail."""
        mock_stream = MagicMock()
        mock_stream.read.side_effect = Exception("All attempts failed")
        handler.audio_stream = mock_stream
        
        result = handler._read_audio_with_fallback(CHUNK_SIZE)
        
        assert result is None
    
    def test_warn_if_overflowed(self, handler):
        """Test overflow warning."""
        handler._last_overflow_warn = None
        
        with patch('builtins.print') as mock_print:
            handler._warn_if_overflowed(True, 1000)
            mock_print.assert_called_once()
            assert handler._last_overflow_warn is not None
    
    def test_warn_if_overflowed_no_overflow(self, handler):
        """Test no warning when no overflow."""
        with patch('builtins.print') as mock_print:
            handler._warn_if_overflowed(False, 1000)
            mock_print.assert_not_called()
    
    def test_extract_latest_samples_full_chunk(self, handler):
        """Test extracting latest samples when data is full chunk."""
        data = np.random.randn(CHUNK_SIZE * 2, 1).astype(np.float32)
        
        result = handler._extract_latest_samples(data)
        
        assert len(result) == CHUNK_SIZE
        np.testing.assert_array_equal(result, data.flatten()[-CHUNK_SIZE:])
    
    def test_extract_latest_samples_padded(self, handler):
        """Test extracting latest samples with padding when data is short."""
        data = np.random.randn(CHUNK_SIZE // 2, 1).astype(np.float32)
        
        result = handler._extract_latest_samples(data)
        
        assert len(result) == CHUNK_SIZE
        assert np.allclose(result[:len(data.flatten())], data.flatten())
        assert np.allclose(result[len(data.flatten()):], 0)
    
    def test_pad_audio_data(self, handler):
        """Test audio data padding."""
        short_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        padded = handler._pad_audio_data(short_data)
        
        assert len(padded) == CHUNK_SIZE
        assert padded[:3].tolist() == [1.0, 2.0, 3.0]
        assert np.allclose(padded[3:], 0)
    
    def test_generate_simulated_input(self, handler):
        """Test simulated audio input generation."""
        audio = handler.generate_simulated_input()
        
        assert audio is not None
        assert len(audio) == CHUNK_SIZE
        assert audio.dtype == np.float32
    
    def test_create_time_vector(self, handler):
        """Test time vector creation."""
        time_vector = handler._create_time_vector()
        
        assert len(time_vector) == CHUNK_SIZE
        assert time_vector[0] == 0.0
        assert time_vector[-1] == CHUNK_SIZE / SAMPLE_RATE
    
    def test_initialize_frequency_state(self, handler):
        """Test frequency state initialization."""
        assert handler.freq_state is None
        
        handler._initialize_frequency_state()
        
        assert handler.freq_state is not None
        assert handler.freq_state['f1'] == SIMULATED_FREQ_BASE_1
        assert handler.freq_state['f2'] == SIMULATED_FREQ_BASE_2
        assert handler.freq_state['f3'] == SIMULATED_FREQ_BASE_3
        assert handler.freq_state['amp'] == 1.0
    
    def test_initialize_frequency_state_idempotent(self, handler):
        """Test frequency state initialization is idempotent."""
        handler._initialize_frequency_state()
        first_state = handler.freq_state.copy()
        
        handler._initialize_frequency_state()
        
        assert handler.freq_state == first_state
    
    def test_update_frequencies(self, handler):
        """Test frequency update with random walk."""
        frequencies = handler._update_frequencies(0.0)
        
        assert 'f1' in frequencies
        assert 'f2' in frequencies
        assert 'f3' in frequencies
        assert SIMULATED_FREQ_MIN_1 <= frequencies['f1'] <= SIMULATED_FREQ_MAX_1
        assert SIMULATED_FREQ_MIN_2 <= frequencies['f2'] <= SIMULATED_FREQ_MAX_2
        assert SIMULATED_FREQ_MIN_3 <= frequencies['f3'] <= SIMULATED_FREQ_MAX_3
    
    def test_calculate_amplitude(self, handler):
        """Test amplitude calculation."""
        amplitude = handler._calculate_amplitude(0.0)
        
        assert isinstance(amplitude, (int, float))
        assert amplitude > 0
    
    def test_generate_noise(self, handler):
        """Test noise generation."""
        noise = handler._generate_noise(0.0)
        
        assert len(noise) == CHUNK_SIZE
        assert isinstance(noise, np.ndarray)
    
    def test_combine_signals(self, handler):
        """Test signal combination."""
        time_vector = handler._create_time_vector()
        frequencies = {'f1': 440.0, 'f2': 880.0, 'f3': 1320.0}
        amplitude = 1.0
        noise = np.random.randn(CHUNK_SIZE)
        
        signal = handler._combine_signals(time_vector, frequencies, amplitude, noise)
        
        assert len(signal) == CHUNK_SIZE
        assert isinstance(signal, np.ndarray)
    
    def test_generate_spike(self, handler):
        """Test spike generation."""
        spike = handler._generate_spike()
        
        assert len(spike) <= CHUNK_SIZE
        assert len(spike) > 0
        assert isinstance(spike, np.ndarray)
    
    def test_calculate_silence_region(self, handler):
        """Test silence region calculation."""
        start, length = handler._calculate_silence_region()
        
        assert 0 <= start < CHUNK_SIZE
        assert 0 < length <= CHUNK_SIZE
        assert start + length <= CHUNK_SIZE
    
    def test_read_and_process_audio_data_success(self, handler):
        """Test reading and processing audio data successfully."""
        mock_stream = MagicMock()
        mock_data = (np.random.randn(CHUNK_SIZE, 1).astype(np.float32), False)
        mock_stream.read.return_value = mock_data
        handler.audio_stream = mock_stream
        
        result = handler._read_and_process_audio_data(CHUNK_SIZE)
        
        assert result is not None
        assert len(result) == CHUNK_SIZE
    
    def test_read_and_process_audio_data_fallback_to_noise(self, handler):
        """Test reading audio data falls back to noise on failure."""
        mock_stream = MagicMock()
        mock_stream.read.return_value = None
        handler.audio_stream = mock_stream
        
        result = handler._read_and_process_audio_data(CHUNK_SIZE)
        
        assert result is not None
        assert len(result) == CHUNK_SIZE
    
    def test_read_audio_from_microphone_success(self, handler):
        """Test reading from microphone successfully."""
        mock_stream = MagicMock()
        mock_stream.read_available = CHUNK_SIZE
        mock_data = (np.random.randn(CHUNK_SIZE, 1).astype(np.float32), False)
        mock_stream.read.return_value = mock_data
        handler.audio_stream = mock_stream
        
        result = handler._read_audio_from_microphone()
        
        assert result is not None
        assert len(result) == CHUNK_SIZE
    
    def test_read_audio_from_microphone_no_data(self, handler):
        """Test reading from microphone when no data available."""
        mock_stream = MagicMock()
        mock_stream.read_available = 0
        handler.audio_stream = mock_stream
        
        result = handler._read_audio_from_microphone()
        
        assert result is not None
        assert len(result) == CHUNK_SIZE
    
    def test_read_audio_from_microphone_exception(self, handler):
        """Test reading from microphone handles exceptions."""
        mock_stream = MagicMock()
        mock_stream.read_available = PropertyMock(side_effect=Exception("Stream error"))
        handler.audio_stream = mock_stream
        
        with patch.object(handler, 'generate_simulated_input', return_value=np.zeros(CHUNK_SIZE)):
            result = handler._read_audio_from_microphone()
            assert result is not None
    
    def test_handle_feedback_prevention_exception(self, handler):
        """Test feedback prevention handles exceptions."""
        handler.audio_stream = MagicMock()
        handler.audio_stream.read_available = PropertyMock(side_effect=Exception("Drain error"))
        
        result = handler._handle_feedback_prevention()
        
        assert result is not None
        assert len(result) == CHUNK_SIZE
    
    def test_drain_audio_buffer_no_mic(self, handler):
        """Test drain_audio_buffer does nothing when no microphone."""
        handler.use_microphone = False
        
        handler.drain_audio_buffer()
        
        # Should complete without error
    
    def test_drain_audio_buffer_success(self, handler):
        """Test drain_audio_buffer drains when buffer is full."""
        mock_stream = MagicMock()
        mock_stream.read_available = CHUNK_SIZE * 5
        handler.audio_stream = mock_stream
        handler.use_microphone = True
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                handler.audio_stream.read_available = CHUNK_SIZE - 1
            return (np.zeros((CHUNK_SIZE, 1)), False)
        
        mock_stream.read.side_effect = side_effect
        
        handler.drain_audio_buffer()
        
        assert mock_stream.read.called
    
    def test_drain_audio_buffer_exception(self, handler):
        """Test drain_audio_buffer handles exceptions gracefully."""
        mock_stream = MagicMock()
        mock_stream.read_available = PropertyMock(side_effect=Exception("Error"))
        handler.audio_stream = mock_stream
        handler.use_microphone = True
        
        handler.drain_audio_buffer()
        
        # Should complete without error
    
    def test_read_audio_from_microphone_buffer_overflow(self, handler):
        """Test reading from microphone handles buffer overflow."""
        mock_stream = MagicMock()
        mock_stream.read_available = CHUNK_SIZE * BUFFER_FULL_THRESHOLD_MULTIPLIER + 1
        mock_data = (np.random.randn(CHUNK_SIZE, 1).astype(np.float32), False)
        mock_stream.read.return_value = mock_data
        handler.audio_stream = mock_stream
        
        with patch.object(handler, '_handle_buffer_overflow') as mock_handle:
            result = handler._read_audio_from_microphone()
            mock_handle.assert_called_once()
            assert result is not None
    
    def test_drain_audio_buffer_aggressively_exception(self, handler):
        """Test aggressive draining handles exceptions."""
        mock_stream = MagicMock()
        mock_stream.read_available = CHUNK_SIZE * 5
        mock_stream.read.side_effect = Exception("Read error")
        handler.audio_stream = mock_stream
        
        handler._drain_audio_buffer_aggressively()
        
        # Should complete without error
    
    @patch('core.audio_input_handler.np.random.random')
    @patch('core.audio_input_handler.np.random.randint')
    @patch('core.audio_input_handler.np.random.normal')
    def test_add_spike_events_with_spike(self, mock_normal, mock_randint, mock_random, handler):
        """Test adding spike events when probability triggers."""
        from config import SIMULATED_SPIKE_PROBABILITY
        mock_random.return_value = SIMULATED_SPIKE_PROBABILITY - 0.01  # Trigger spike
        mock_randint.return_value = 0  # Spike starts at beginning
        mock_normal.return_value = np.ones(10) * 0.5  # Spike values
        signal = np.zeros(CHUNK_SIZE)
        original_sum = np.sum(signal)
        
        result = handler._add_spike_events(signal)
        
        assert np.sum(result) > original_sum  # Signal was modified (spike added)
        assert np.sum(result[:10]) > 0  # First 10 samples have spike
        mock_random.assert_called_once()
    
    @patch('core.audio_input_handler.np.random.random')
    @patch('core.audio_input_handler.np.random.randint')
    def test_add_silence_events_with_silence(self, mock_randint, mock_random, handler):
        """Test adding silence events when probability triggers."""
        from config import SIMULATED_SILENCE_PROBABILITY, SIMULATED_SILENCE_ATTENUATION
        mock_random.return_value = SIMULATED_SILENCE_PROBABILITY - 0.01  # Trigger silence
        mock_randint.side_effect = [0, CHUNK_SIZE // 2]  # Silence start and length
        signal = np.ones(CHUNK_SIZE)
        original_sum = np.sum(signal)
        
        result = handler._add_silence_events(signal)
        
        assert np.sum(result) < original_sum  # Signal was modified (silence added)
        assert np.allclose(result[:CHUNK_SIZE//2], np.ones(CHUNK_SIZE//2) * SIMULATED_SILENCE_ATTENUATION)  # First half attenuated
        mock_random.assert_called_once()
    
    def test_drain_audio_buffer_exception_in_loop(self, handler):
        """Test drain_audio_buffer handles exception in the while loop."""
        mock_stream = MagicMock()
        mock_stream.read_available = CHUNK_SIZE * 5
        mock_stream.read.side_effect = Exception("Read error in loop")
        handler.audio_stream = mock_stream
        handler.use_microphone = True
        
        handler.drain_audio_buffer()
        
        # Should complete without error
    
    def test_drain_audio_buffer_available_becomes_zero(self, handler):
        """Test drain_audio_buffer breaks when available becomes zero."""
        mock_stream = MagicMock()
        call_count = 0
        def read_available_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CHUNK_SIZE * 5
            elif call_count == 2:
                return 0  # Available becomes zero
            return CHUNK_SIZE - 1
        
        mock_stream.read_available = PropertyMock(side_effect=read_available_side_effect)
        mock_stream.read.return_value = (np.zeros((CHUNK_SIZE, 1)), False)
        handler.audio_stream = mock_stream
        handler.use_microphone = True
        
        handler.drain_audio_buffer()
        
        # Should complete without error

