"""
Audio Input Handler: Manages all audio input operations.
Separates audio input concerns from the main event loop (Single Responsibility Principle).
"""

import numpy as np
import sounddevice as sd
import time

from config import (
    SAMPLE_RATE, CHUNK_SIZE,
    DITHERING_NOISE_SIGMA, BUFFER_DRAIN_MULTIPLIER, BUFFER_READ_MULTIPLIER,
    OVERFLOW_WARNING_INTERVAL_SECONDS, BUFFER_DRAIN_WARNING_INTERVAL_SECONDS,
    BUFFER_FULL_THRESHOLD_MULTIPLIER,
    SIMULATED_FREQ_BASE_1, SIMULATED_FREQ_BASE_2, SIMULATED_FREQ_BASE_3,
    SIMULATED_FREQ_WALK_STD_1, SIMULATED_FREQ_WALK_STD_2, SIMULATED_FREQ_WALK_STD_3,
    SIMULATED_FREQ_MIN_1, SIMULATED_FREQ_MAX_1, SIMULATED_FREQ_MIN_2, SIMULATED_FREQ_MAX_2,
    SIMULATED_FREQ_MIN_3, SIMULATED_FREQ_MAX_3, SIMULATED_AMP_BASE, SIMULATED_AMP_RANGE,
    SIMULATED_AMP_FREQ, SIMULATED_SIGNAL_AMP_1, SIMULATED_SIGNAL_AMP_2, SIMULATED_SIGNAL_AMP_3,
    SIMULATED_NOISE_BASE, SIMULATED_NOISE_RANGE, SIMULATED_NOISE_FREQ,
    SIMULATED_SPIKE_PROBABILITY, SIMULATED_SPIKE_MAGNITUDE_MIN, SIMULATED_SPIKE_MAGNITUDE_MAX,
    SIMULATED_SPIKE_LENGTH_MIN, SIMULATED_SPIKE_LENGTH_MAX,
    SIMULATED_SILENCE_PROBABILITY, SIMULATED_SILENCE_START_RATIO,
    SIMULATED_SILENCE_LENGTH_MIN_RATIO, SIMULATED_SILENCE_LENGTH_MAX_RATIO,
    SIMULATED_SILENCE_ATTENUATION
)


class AudioInputHandler:
    """
    Handles all audio input operations.
    
    Responsibilities:
    - Microphone setup and management
    - Audio buffer management
    - Simulated audio generation
    - Feedback prevention
    """
    
    def __init__(self, afl, start_time):
        """
        Initialize audio input handler.
        
        Args:
            afl: AestheticFeedbackLoop instance (to check if OPU is speaking)
            start_time: Start time for simulated input generation
        """
        self.afl = afl
        self.start_time = start_time
        self.use_microphone = False
        self.audio_stream = None
        self._last_overflow_warn = None
        self.freq_state = None
    
    def setup_audio_input(self):
        """Setup audio input (microphone or simulation)."""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=CHUNK_SIZE,
                latency='low'
            )
            self.audio_stream.start()
            self.use_microphone = True
            print("[OPU] Microphone input enabled (low latency mode).")
        except Exception as e1:
            try:
                self.audio_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    blocksize=CHUNK_SIZE,
                    dtype=np.float32
                )
                self.audio_stream.start()
                self.use_microphone = True
                print("[OPU] Microphone input enabled.")
            except Exception as e2:
                print(f"[OPU] Microphone not available: {e1}")
                print("[OPU] Using simulated audio input.")
                self.use_microphone = False
    
    def get_audio_input(self):
        """Get audio input from microphone or generate simulated input."""
        if not self._should_use_microphone():
            return self.generate_simulated_input()
        
        if self._is_opu_speaking():
            return self._handle_feedback_prevention()
        
        return self._read_audio_from_microphone()
    
    def _should_use_microphone(self):
        """Check if microphone should be used."""
        return self.use_microphone and self.audio_stream is not None
    
    def _is_opu_speaking(self):
        """Check if OPU is currently speaking (outputting audio)."""
        return self.afl.is_active()
    
    def _handle_feedback_prevention(self):
        """Handle microphone muting when OPU is speaking to prevent feedback."""
        try:
            self._drain_audio_buffer_aggressively()
            return self._generate_dithering_noise()
        except Exception:
            return self._generate_dithering_noise()
    
    def _read_audio_from_microphone(self):
        """Read audio data from microphone with buffer management."""
        try:
            available = self.audio_stream.read_available
            
            if self._is_buffer_full(available):
                self._handle_buffer_overflow(available)
                available = self.audio_stream.read_available
            
            if available > 0:
                return self._read_and_process_audio_data(available)
            else:
                return self._generate_dithering_noise()
        except Exception:
            return self.generate_simulated_input()
    
    def _is_buffer_full(self, available):
        """Check if audio buffer is getting full."""
        return available > CHUNK_SIZE * BUFFER_FULL_THRESHOLD_MULTIPLIER
    
    def _handle_buffer_overflow(self, available):
        """Handle buffer overflow by draining aggressively."""
        self._warn_buffer_draining(available)
        self._drain_audio_buffer_aggressively()
    
    def _warn_buffer_draining(self, available):
        """Print buffer draining warning if enough time has passed."""
        if self._last_overflow_warn is None or \
           time.time() - self._last_overflow_warn > BUFFER_DRAIN_WARNING_INTERVAL_SECONDS:
            print(f"[OPU] Audio buffer draining (available: {available})")
            self._last_overflow_warn = time.time()
    
    def _drain_audio_buffer_aggressively(self):
        """Drain audio buffer aggressively to prevent overflow."""
        while self.audio_stream.read_available > CHUNK_SIZE:
            try:
                drain_size = min(self.audio_stream.read_available, CHUNK_SIZE * BUFFER_DRAIN_MULTIPLIER)
                self.audio_stream.read(drain_size, blocking=False)
            except Exception:
                break
    
    def _read_and_process_audio_data(self, available):
        """Read audio data and process it into the correct format."""
        read_size = min(available, CHUNK_SIZE * BUFFER_READ_MULTIPLIER)
        data = self._read_audio_with_fallback(read_size)
        
        if data is None:
            return self._generate_dithering_noise()
        
        self._warn_if_overflowed(data[1], available)
        return self._extract_latest_samples(data[0])
    
    def _read_audio_with_fallback(self, read_size):
        """Read audio data with fallback to smaller chunk size."""
        try:
            return self.audio_stream.read(read_size, blocking=False)
        except Exception:
            try:
                return self.audio_stream.read(CHUNK_SIZE, blocking=False)
            except Exception:
                return None
    
    def _warn_if_overflowed(self, overflowed, available):
        """Warn about buffer overflow if it occurred."""
        if overflowed:
            if self._last_overflow_warn is None or \
               time.time() - self._last_overflow_warn > OVERFLOW_WARNING_INTERVAL_SECONDS:
                print(f"[OPU] Audio buffer overflow detected! (available: {available})")
                self._last_overflow_warn = time.time()
    
    def _extract_latest_samples(self, data):
        """Extract the latest CHUNK_SIZE samples from audio data."""
        data_flat = data.flatten()
        if len(data_flat) >= CHUNK_SIZE:
            return data_flat[-CHUNK_SIZE:]
        else:
            return self._pad_audio_data(data_flat)
    
    def _pad_audio_data(self, data_flat):
        """Pad audio data to CHUNK_SIZE if it's shorter."""
        padded = np.zeros(CHUNK_SIZE, dtype=np.float32)
        padded[:len(data_flat)] = data_flat
        return padded
    
    def _generate_dithering_noise(self):
        """Generate dithering noise to prevent divide-by-zero errors."""
        return np.random.normal(0, DITHERING_NOISE_SIGMA, CHUNK_SIZE).astype(np.float32)
    
    def generate_simulated_input(self):
        """Generate simulated audio input with varying patterns for testing."""
        time_vector = self._create_time_vector()
        current_time = time.time() - self.start_time
        
        frequencies = self._update_frequencies(current_time)
        amplitude = self._calculate_amplitude(current_time)
        noise = self._generate_noise(current_time)
        
        signal = self._combine_signals(time_vector, frequencies, amplitude, noise)
        signal = self._add_spike_events(signal)
        signal = self._add_silence_events(signal)
        
        return signal.astype(np.float32)
    
    def _create_time_vector(self):
        """Create time vector for signal generation."""
        return np.linspace(0, CHUNK_SIZE / SAMPLE_RATE, CHUNK_SIZE)
    
    def _initialize_frequency_state(self):
        """Initialize frequency state if not exists."""
        if self.freq_state is None:
            self.freq_state = {
                'f1': SIMULATED_FREQ_BASE_1,
                'f2': SIMULATED_FREQ_BASE_2,
                'f3': SIMULATED_FREQ_BASE_3,
                'amp': 1.0
            }
    
    def _update_frequencies(self, current_time):
        """Update frequencies using random walk and return clamped values."""
        self._initialize_frequency_state()
        
        self.freq_state['f1'] += np.random.normal(0, SIMULATED_FREQ_WALK_STD_1)
        self.freq_state['f2'] += np.random.normal(0, SIMULATED_FREQ_WALK_STD_2)
        self.freq_state['f3'] += np.random.normal(0, SIMULATED_FREQ_WALK_STD_3)
        
        return {
            'f1': np.clip(self.freq_state['f1'], SIMULATED_FREQ_MIN_1, SIMULATED_FREQ_MAX_1),
            'f2': np.clip(self.freq_state['f2'], SIMULATED_FREQ_MIN_2, SIMULATED_FREQ_MAX_2),
            'f3': np.clip(self.freq_state['f3'], SIMULATED_FREQ_MIN_3, SIMULATED_FREQ_MAX_3)
        }
    
    def _calculate_amplitude(self, current_time):
        """Calculate amplitude modulation based on time."""
        self._initialize_frequency_state()
        self.freq_state['amp'] = SIMULATED_AMP_BASE + SIMULATED_AMP_RANGE * np.sin(current_time * SIMULATED_AMP_FREQ)
        return self.freq_state['amp']
    
    def _generate_noise(self, current_time):
        """Generate noise with time-varying level."""
        noise_level = SIMULATED_NOISE_BASE + SIMULATED_NOISE_RANGE * abs(np.sin(current_time * SIMULATED_NOISE_FREQ))
        return np.random.normal(0, noise_level, CHUNK_SIZE)
    
    def _combine_signals(self, time_vector, frequencies, amplitude, noise):
        """Combine multiple frequency components with noise."""
        signal = (
            amplitude * SIMULATED_SIGNAL_AMP_1 * np.sin(2 * np.pi * frequencies['f1'] * time_vector) +
            SIMULATED_SIGNAL_AMP_2 * np.sin(2 * np.pi * frequencies['f2'] * time_vector) +
            SIMULATED_SIGNAL_AMP_3 * np.sin(2 * np.pi * frequencies['f3'] * time_vector) +
            noise
        )
        return signal
    
    def _add_spike_events(self, signal):
        """Add random spike events to create high surprise."""
        if np.random.random() < SIMULATED_SPIKE_PROBABILITY:
            spike = self._generate_spike()
            spike_start = np.random.randint(0, max(1, CHUNK_SIZE - len(spike)))
            signal[spike_start:spike_start+len(spike)] += spike
        return signal
    
    def _generate_spike(self):
        """Generate a single spike event."""
        spike_magnitude = np.random.uniform(SIMULATED_SPIKE_MAGNITUDE_MIN, SIMULATED_SPIKE_MAGNITUDE_MAX)
        spike_length = np.random.randint(SIMULATED_SPIKE_LENGTH_MIN, SIMULATED_SPIKE_LENGTH_MAX)
        spike_length = min(spike_length, CHUNK_SIZE)
        return np.random.normal(0, spike_magnitude, spike_length)
    
    def _add_silence_events(self, signal):
        """Add random silence events to create surprise."""
        if np.random.random() < SIMULATED_SILENCE_PROBABILITY:
            silence_start, silence_length = self._calculate_silence_region()
            signal[silence_start:silence_start+silence_length] *= SIMULATED_SILENCE_ATTENUATION
        return signal
    
    def _calculate_silence_region(self):
        """Calculate silence region start and length."""
        silence_start = np.random.randint(0, int(CHUNK_SIZE * SIMULATED_SILENCE_START_RATIO))
        min_length = int(CHUNK_SIZE * SIMULATED_SILENCE_LENGTH_MIN_RATIO)
        max_length = int(CHUNK_SIZE * SIMULATED_SILENCE_LENGTH_MAX_RATIO)
        silence_length = np.random.randint(min_length, max_length)
        return silence_start, silence_length
    
    def drain_audio_buffer(self):
        """
        Aggressively drain audio buffer to prevent overflow.
        Called frequently to keep buffer from filling up.
        """
        if not self.use_microphone or self.audio_stream is None:
            return
        
        try:
            available = self.audio_stream.read_available
            if available > CHUNK_SIZE * 4:
                while available > CHUNK_SIZE:
                    try:
                        drain_size = min(available, CHUNK_SIZE * 8)
                        self.audio_stream.read(drain_size, blocking=False)
                        available = self.audio_stream.read_available
                        if available <= 0:
                            break
                    except:
                        break
        except Exception:
            pass

