"""
The Voice: Aesthetic Feedback Loop & Phoneme Analysis.
Maps surprise scores to audio frequencies and phonemes.
"""

import numpy as np
import sounddevice as sd
import time
from collections import deque
from threading import Lock
from config import BASE_FREQUENCY, SAMPLE_RATE


class AestheticFeedbackLoop:
    """
    Maps surprise score (s_score) to audio frequency.
    Uses sounddevice OutputStream with queue for reliable non-blocking audio playback.
    """
    
    def __init__(self, base_pitch=440.0):
        self.base_pitch = base_pitch
        self.sample_rate = SAMPLE_RATE
        self.current_frequency = base_pitch
        
        # Audio output queue and stream
        self.audio_queue = deque(maxlen=1)  # Only one tone at a time
        self.queue_lock = Lock()
        self.output_stream = None
        self.is_playing = False  # Track if currently playing
        
        # Throttle to prevent too many tones too quickly
        self.last_tone_time = 0.0
        self.min_tone_interval = 0.15  # Minimum 150ms between tones
        
        self._init_output_stream()
    
    def _init_output_stream(self):
        """Initialize a non-blocking audio output stream with callback."""
        try:
            def audio_callback(outdata, frames, time_info, status):
                """Callback to fill output buffer from queue."""
                if status:
                    print(f"[AFL] Audio output status: {status}")
                
                # Always start with silence
                outdata.fill(0.0)
                
                with self.queue_lock:
                    if len(self.audio_queue) > 0:
                        # Get next audio chunk from queue
                        audio_data = self.audio_queue.popleft()
                        self.is_playing = True
                        
                        # Copy as much as we can
                        copy_len = min(len(audio_data), frames)
                        if copy_len > 0:
                            outdata[:copy_len, 0] = audio_data[:copy_len]
                        
                        # If we used all the audio data, mark as not playing
                        if copy_len >= len(audio_data):
                            self.is_playing = False
                    else:
                        # No audio in queue, mark as not playing
                        self.is_playing = False
            
            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=0,  # Let sounddevice choose optimal blocksize
                latency='low',  # Low latency for responsive feedback
                callback=audio_callback
            )
            self.output_stream.start()
            print("[AFL] Audio output stream initialized.")
        except Exception as e:
            print(f"[AFL] Warning: Could not initialize audio output stream: {e}")
            print("[AFL] Audio feedback will be disabled.")
            self.output_stream = None
    
    def update_pitch(self, base_pitch):
        """Update the base pitch (called when character evolves)."""
        self.base_pitch = base_pitch
    
    def generate_tone(self, s_score, duration=0.1):
        """
        Generates a tone based on surprise score.
        
        Formula: frequency = base_pitch * (1 + s_score / 10)
        
        Args:
            s_score: surprise score
            duration: duration in seconds
            
        Returns:
            numpy array of audio samples
        """
        # Map s_score to frequency
        # Higher surprise = higher pitch
        frequency = self.base_pitch * (1.0 + s_score / 10.0)
        
        # Clamp frequency to reasonable range
        frequency = np.clip(frequency, 50.0, 2000.0)
        
        self.current_frequency = frequency
        
        # Generate sine wave with reduced amplitude to prevent clipping
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Apply smooth envelope: attack, sustain, release
        # Attack: 10% of duration
        # Release: 30% of duration
        attack_samples = int(num_samples * 0.1)
        release_samples = int(num_samples * 0.3)
        sustain_samples = num_samples - attack_samples - release_samples
        
        envelope = np.ones(num_samples)
        
        # Attack (fade in)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Sustain (full volume)
        # Already 1.0, no change needed
        
        # Release (fade out)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        # Apply envelope and reduce amplitude significantly to prevent clipping
        # Use 0.05 amplitude (5% of max) to leave plenty of headroom
        tone = tone * envelope * 0.05
        
        # Add longer silence padding at the end to prevent overlap
        silence_padding = int(self.sample_rate * 0.05)  # 50ms silence between tones
        tone_with_padding = np.concatenate([tone, np.zeros(silence_padding, dtype=np.float32)])
        
        return tone_with_padding.astype(np.float32)
    
    def play_tone(self, s_score, duration=0.1):
        """
        Plays a tone based on surprise score.
        Queues audio for non-blocking playback via OutputStream callback.
        Includes throttling to prevent too many tones too quickly.
        Only one tone plays at a time to prevent clipping.
        
        Args:
            s_score: surprise score
            duration: duration in seconds (default 0.1 for clearer tones)
        """
        # Only play if s_score is significant (reduce audio spam)
        if s_score < 0.8:
            return  # Skip very low surprise events
        
        # If output stream is not available, skip
        if self.output_stream is None:
            return
        
        # Throttle: don't play if we just played a tone recently
        current_time = time.time()
        if current_time - self.last_tone_time < self.min_tone_interval:
            return  # Skip this tone, too soon after last one
        
        try:
            with self.queue_lock:
                # Only play if not currently playing and queue is empty
                if not self.is_playing and len(self.audio_queue) == 0:
                    tone = self.generate_tone(s_score, duration)
                    self.audio_queue.append(tone)
                    self.last_tone_time = current_time
                # If already playing, skip this tone to prevent overlap/clipping
        except Exception as e:
            # Silently fail to avoid spam
            pass
    
    def cleanup(self):
        """Clean up audio output stream."""
        if self.output_stream is not None:
            try:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            except:
                pass


class PhonemeAnalyzer:
    """
    Maps s_score ranges to phonemes.
    Filters out noise and only recognizes "Spoken" sounds with Structural Intent.
    """
    
    def __init__(self, speech_threshold=1.5):
        self.speech_threshold = speech_threshold
        self.phoneme_history = []
    
    def analyze(self, s_score, pitch):
        """
        Returns a phoneme only if the sound has Structural Intent.
        
        Phoneme Map:
        - 0.0-1.5: Ignored (Noise)
        - 1.5-3.0: Vowels (Low Tension)
        - 3.0-6.0: Fricatives (Flowing tension)
        - 6.0+: Plosives (High Tension)
        
        Args:
            s_score: surprise score
            pitch: current audio pitch
            
        Returns:
            phoneme string or None if below speech threshold
        """
        # INTENT FILTER: Ignore "Background Hum"
        if s_score < self.speech_threshold:
            return None
        
        # TENSION MAPPING
        if s_score < 3.0:
            # Vowels (Low Tension)
            phoneme = "a" if pitch > 200 else "o"
        elif s_score < 6.0:
            # Fricatives (Flowing tension)
            phoneme = "s"
        else:
            # Plosives (Hard break)
            phoneme = "k"
        
        # Store in history
        self.phoneme_history.append({
            'phoneme': phoneme,
            's_score': s_score,
            'pitch': pitch
        })
        
        return phoneme
    
    def get_recent_phonemes(self, count=10):
        """
        Returns recent phonemes for analysis.
        
        Args:
            count: number of recent phonemes to return
            
        Returns:
            list of phoneme strings
        """
        recent = self.phoneme_history[-count:]
        return [p['phoneme'] for p in recent if p['phoneme'] is not None]
    
    def get_phoneme_statistics(self):
        """
        Returns statistics about learned phonemes.
        
        Returns:
            dict with counts and distribution
        """
        phonemes = [p['phoneme'] for p in self.phoneme_history if p['phoneme'] is not None]
        
        if len(phonemes) == 0:
            return {
                'total': 0,
                'vowels': 0,
                'fricatives': 0,
                'plosives': 0
            }
        
        vowels = sum(1 for p in phonemes if p in ['a', 'o', 'e', 'i', 'u'])
        fricatives = sum(1 for p in phonemes if p in ['s', 'f', 'h'])
        plosives = sum(1 for p in phonemes if p in ['k', 'p', 't', 'b', 'd', 'g'])
        
        return {
            'total': len(phonemes),
            'vowels': vowels,
            'fricatives': fricatives,
            'plosives': plosives,
            'distribution': {
                'vowels': vowels / len(phonemes) if len(phonemes) > 0 else 0,
                'fricatives': fricatives / len(phonemes) if len(phonemes) > 0 else 0,
                'plosives': plosives / len(phonemes) if len(phonemes) > 0 else 0
            }
        }

