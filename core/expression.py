"""
The Voice: Aesthetic Feedback Loop & Phoneme Analysis.
Maps surprise scores to audio frequencies and phonemes.
"""

import numpy as np
import sounddevice as sd
from config import BASE_FREQUENCY, SAMPLE_RATE


class AestheticFeedbackLoop:
    """
    Continuous Phase Oscillator.
    Eliminates clipping by maintaining phase continuity across buffer callbacks.
    Creates a smooth, theremin-like voice that glides between pitches.
    """
    
    def __init__(self, base_pitch=440.0):
        self.base_pitch = base_pitch
        self.sample_rate = SAMPLE_RATE
        self.current_frequency = base_pitch
        self.target_frequency = base_pitch
        self.amplitude = 0.0
        self.phase = 0.0
        
        # Audio Stream State
        self.stream = None
        self.running = False
        
        self.start()
    
    def callback(self, outdata, frames, time_info, status):
        """Audio callback - generates continuous waveform with phase continuity."""
        if status:
            print(f"[AFL] Audio output status: {status}")
        
        # Time array for this buffer chunk
        t = np.arange(frames) / self.sample_rate
        
        # 1. SMOOTH PITCH GLIDE (Portamento)
        # Move current freq 10% closer to target freq per buffer
        # This creates smooth pitch transitions instead of instant jumps
        self.current_frequency += (self.target_frequency - self.current_frequency) * 0.1
        
        # 2. CALCULATE PHASE INCREMENT
        # phase_increment = 2 * pi * freq / sample_rate
        phase_increment = 2 * np.pi * self.current_frequency / self.sample_rate
        
        # 3. GENERATE WAVEFORM WITH PHASE CONTINUITY
        # We add the increment to the accumulated phase
        # This ensures the wave connects perfectly to the previous chunk
        phases = self.phase + np.arange(frames) * phase_increment
        self.phase = phases[-1] % (2 * np.pi)  # Wrap phase to keep numbers small
        
        # Sine Wave
        waveform = np.sin(phases)
        
        # 4. APPLY AMPLITUDE (Volume)
        # Smoothly decay amplitude if no new input (natural fade)
        self.amplitude *= 0.98
        
        # Fill buffer
        # 0.1 is a safe master volume to prevent clipping
        outdata[:] = (waveform * self.amplitude * 0.1).reshape(-1, 1).astype(np.float32)
    
    def start(self):
        """Start the audio output stream."""
        try:
            self.stream = sd.OutputStream(
                channels=1,
                callback=self.callback,
                samplerate=self.sample_rate,
                dtype=np.float32,
                latency='low'
            )
            self.stream.start()
            self.running = True
            print("[AFL] Continuous phase oscillator initialized.")
        except Exception as e:
            print(f"[AFL] Warning: Could not initialize audio output stream: {e}")
            print("[AFL] Audio feedback will be disabled.")
            self.stream = None
            self.running = False
    
    def update_pitch(self, base_pitch):
        """Update the base pitch (called when character evolves)."""
        self.base_pitch = base_pitch
        # Update target frequency to reflect new base pitch
        # Keep the relative s_score mapping intact
    
    def play_tone(self, s_score, duration=None):
        """
        Updates the continuous oscillator based on surprise.
        The sound persists until s_score drops or changes.
        
        Args:
            s_score: surprise score
            duration: ignored (continuous oscillator)
        """
        # Only respond to significant surprise (reduce audio spam)
        if s_score < 0.8:
            # Fade out quickly if below threshold
            self.amplitude *= 0.9
            return
        
        # Map S_Score to Pitch (Surprise = High Pitch)
        # Formula: base_pitch * (1 + s_score / 10)
        # This maintains the original mapping but allows smooth transitions
        new_freq = self.base_pitch * (1.0 + s_score / 10.0)
        self.target_frequency = np.clip(new_freq, 50.0, 2000.0)
        
        # Map S_Score to Volume (Attention = Louder)
        # Higher surprise = louder response
        target_amp = min(1.0, s_score / 3.0)
        # Smooth amplitude transition
        self.amplitude += (target_amp - self.amplitude) * 0.2
    
    def cleanup(self):
        """Clean up audio output stream."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                self.running = False
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

