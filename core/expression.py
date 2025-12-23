"""
The Voice: Aesthetic Feedback Loop & Phoneme Analysis.
Maps surprise scores to audio frequencies and phonemes.
"""

import numpy as np
import sounddevice as sd
from config import BASE_FREQUENCY, SAMPLE_RATE


class AestheticFeedbackLoop:
    """
    Maps surprise score (s_score) to audio frequency.
    Uses sounddevice for real-time audio playback.
    """
    
    def __init__(self, base_pitch=440.0):
        self.base_pitch = base_pitch
        self.sample_rate = SAMPLE_RATE
        self.current_frequency = base_pitch
        self.is_playing = False
    
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
        
        # Generate sine wave
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope to avoid clicks
        envelope = np.exp(-t * 5)  # Exponential decay
        tone = tone * envelope
        
        return tone.astype(np.float32)
    
    def play_tone(self, s_score, duration=0.05):
        """
        Plays a tone based on surprise score.
        Uses very short duration to minimize blocking.
        
        Args:
            s_score: surprise score
            duration: duration in seconds (default 0.05 for quick feedback)
        """
        # Only play if s_score is significant (reduce audio spam)
        if s_score < 0.8:
            return  # Skip very low surprise events
        
        # Check if previous playback is done
        try:
            # Try to get active streams - if none, we're not playing
            active_streams = sd.get_stream().active if hasattr(sd, 'get_stream') else False
            if self.is_playing and active_streams:
                return  # Still playing previous tone
        except:
            pass  # If check fails, proceed anyway
        
        try:
            tone = self.generate_tone(s_score, duration)
            self.is_playing = True
            # Play with very short duration - blocking is minimal
            sd.play(tone, samplerate=self.sample_rate)
            # For 0.05s, blocking is acceptable and ensures clean playback
            sd.wait()
            self.is_playing = False
        except Exception as e:
            # Silently fail to avoid spam
            self.is_playing = False


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

