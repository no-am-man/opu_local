"""
The Voice: Aesthetic Feedback Loop & Phoneme Analysis.
Maps surprise scores to audio frequencies and phonemes.
"""

import numpy as np
import sounddevice as sd
from config import BASE_FREQUENCY, SAMPLE_RATE


class AestheticFeedbackLoop:
    """
    Continuous Phase Oscillator with Syllabic Articulation.
    """
    
    def __init__(self, base_pitch=440.0):
        self.base_pitch = base_pitch
        self.sample_rate = SAMPLE_RATE
        
        # Oscillator State
        self.current_frequency = base_pitch
        self.target_frequency = base_pitch
        self.phase = 0.0
        
        # Amplitude State
        self.current_amp = 0.0
        self.target_amp = 0.0
        
        # Articulation (The "Talking" Rhythm)
        self.syllable_phase = 0.0
        self.is_speaking = False
        
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
        self.current_frequency += (self.target_frequency - self.current_frequency) * 0.1
        self.current_amp += (self.target_amp - self.current_amp) * 0.1
        
        # 2. GENERATE CARRIER WAVE
        phase_increment = 2 * np.pi * self.current_frequency / self.sample_rate
        phases = self.phase + np.arange(frames) * phase_increment
        self.phase = phases[-1] % (2 * np.pi) 
        
        carrier = np.sin(phases)
        
        # 3. APPLY SYLLABIC RHYTHM (The "Words")
        # If we are "speaking" (High S_Score), we modulate volume at 8Hz
        if self.is_speaking:
            # 8Hz LFO for syllable rate
            syllable_inc = 2 * np.pi * 8.0 / self.sample_rate
            syllable_phases = self.syllable_phase + np.arange(frames) * syllable_inc
            self.syllable_phase = syllable_phases[-1] % (2 * np.pi)
            
            # Create a "wah-wah" envelope (0.6 to 1.0 amplitude)
            articulation = 0.6 + (0.4 * np.sin(syllable_phases))
        else:
            articulation = 1.0 # Flat drone if just humming
        
        # 4. OUTPUT
        # Signal = Wave * Volume * Rhythm * MasterGain (0.5)
        # We removed the decay (*= 0.98) because it killed the sound too fast
        signal = carrier * self.current_amp * articulation * 0.5
        
        outdata[:] = signal.reshape(-1, 1).astype(np.float32)
    
    def start(self):
        try:
            self.stream = sd.OutputStream(
                channels=1,
                callback=self.callback,
                samplerate=self.sample_rate,
                dtype=np.float32,
                latency='low',
                blocksize=1024 # Slightly larger buffer for stability
            )
            self.stream.start()
            self.running = True
            print("[AFL] Continuous phase oscillator initialized.")
        except Exception as e:
            print(f"[AFL] Error: {e}")
            self.running = False
            
    def update_pitch(self, base_pitch):
        self.base_pitch = base_pitch

    def play_tone(self, s_score, duration=None):
        """
        Updates the continuous oscillator based on surprise.
        """
        # NOISE GATE: If bored, go silent
        if s_score < 0.2:
            self.target_amp = 0.0
            self.is_speaking = False
            return
        
        # 1. SET VOLUME (Attention)
        self.target_amp = min(1.0, s_score / 3.0)
        
        # 2. SET PITCH (Surprise)
        # Base * (1 + score/10)
        new_freq = self.base_pitch * (1.0 + s_score / 10.0)
        self.target_frequency = np.clip(new_freq, 50.0, 2000.0)
        
        # 3. TRIGGER ARTICULATION
        # If surprise is high, turn on the "Syllable LFO"
        if s_score > 1.5:
            self.is_speaking = True
        else:
            self.is_speaking = False

    def cleanup(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class PhonemeAnalyzer:
    """
    Maps s_score ranges to phonemes.
    Filters out noise and only recognizes "Spoken" sounds with Structural Intent.
    """
    
    def __init__(self, speech_threshold=1.5, max_history=10000):
        self.speech_threshold = speech_threshold
        self.phoneme_history = []
        self.max_history = max_history 
    
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
        
        # Store in history (with cap to prevent unbounded growth)
        self.phoneme_history.append({
            'phoneme': phoneme,
            's_score': s_score,
            'pitch': pitch
        })
        
        # Cap history to prevent memory leak over very long runtimes (years)
        if len(self.phoneme_history) > self.max_history:
            self.phoneme_history = self.phoneme_history[-self.max_history:]
        
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
