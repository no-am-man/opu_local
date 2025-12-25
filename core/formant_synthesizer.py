"""
Formant Synthesizer: Real-time phoneme synthesis using formant frequencies.
Generates speech sounds from phoneme definitions using formant synthesis.
"""

import numpy as np
from typing import Optional
from core.phoneme_inventory import PhonemeDefinition, PHONEME_INVENTORY
from config import SAMPLE_RATE


class FormantSynthesizer:
    """
    Synthesizes phonemes using formant synthesis.
    Uses formant frequencies (F1, F2, F3) for vowels and noise bands for consonants.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.current_phoneme: Optional[PhonemeDefinition] = None
        self.phoneme_phase = 0.0
    
    def synthesize_phoneme(self, phoneme_def: PhonemeDefinition, duration_ms: Optional[float] = None) -> np.ndarray:
        """
        Synthesize a phoneme using formant synthesis.
        
        Args:
            phoneme_def: PhonemeDefinition to synthesize
            duration_ms: Duration in milliseconds (uses phoneme default if None)
            
        Returns:
            numpy array of audio samples
        """
        duration = duration_ms if duration_ms is not None else phoneme_def.duration_ms
        num_samples = int(self.sample_rate * duration / 1000.0)
        
        if phoneme_def.category == "vowel":
            return self._synthesize_vowel(phoneme_def, num_samples)
        elif phoneme_def.category == "diphthong":
            return self._synthesize_diphthong(phoneme_def, num_samples)
        elif phoneme_def.articulation == "nasal":
            return self._synthesize_nasal(phoneme_def, num_samples)
        elif phoneme_def.articulation == "liquid":
            return self._synthesize_liquid(phoneme_def, num_samples)
        elif phoneme_def.articulation == "glide":
            return self._synthesize_glide(phoneme_def, num_samples)
        elif phoneme_def.articulation == "plosive":
            return self._synthesize_plosive(phoneme_def, num_samples)
        elif phoneme_def.articulation == "fricative":
            return self._synthesize_fricative(phoneme_def, num_samples)
        elif phoneme_def.articulation == "affricate":
            return self._synthesize_affricate(phoneme_def, num_samples)
        else:
            # Default: simple sine wave
            return self._synthesize_simple(phoneme_def, num_samples)
    
    def _synthesize_vowel(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize vowel using formant frequencies."""
        t = np.arange(num_samples) / self.sample_rate
        
        # Generate formant frequencies
        f1 = phoneme_def.formant_f1
        f2 = phoneme_def.formant_f2
        f3 = phoneme_def.formant_f3
        
        # Formant bandwidths (typical values)
        b1 = 60.0  # F1 bandwidth
        b2 = 100.0  # F2 bandwidth
        b3 = 120.0  # F3 bandwidth
        
        # Generate formant resonances
        formant1 = generate_formant_resonance(t, f1, b1, amplitude=1.0)
        formant2 = generate_formant_resonance(t, f2, b2, amplitude=0.5)
        formant3 = generate_formant_resonance(t, f3, b3, amplitude=0.25)
        
        # Combine formants
        signal = formant1 + formant2 + formant3
        
        # Apply envelope (attack and decay)
        envelope = create_audio_envelope(num_samples, self.sample_rate, attack_ms=10, decay_ms=20)
        
        return signal * envelope
    
    def _synthesize_diphthong(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize diphthong (vowel glide)."""
        # Simple implementation: glide between two formant positions
        t = np.arange(num_samples) / self.sample_rate
        
        # Start and end formant frequencies (simplified)
        f1_start = phoneme_def.formant_f1
        f1_end = phoneme_def.formant_f1 * 0.8
        f2_start = phoneme_def.formant_f2
        f2_end = phoneme_def.formant_f2 * 1.2
        
        # Interpolate formants
        f1 = f1_start + (f1_end - f1_start) * (t / t[-1])
        f2 = f2_start + (f2_end - f2_start) * (t / t[-1])
        
        # Generate gliding formants
        formant1 = np.sin(2 * np.pi * f1 * t)
        formant2 = np.sin(2 * np.pi * f2 * t) * 0.5
        
        signal = formant1 + formant2
        envelope = create_audio_envelope(num_samples, self.sample_rate, attack_ms=20, decay_ms=30)
        
        return signal * envelope
    
    def _synthesize_nasal(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize nasal consonant."""
        # Nasals have formant structure similar to vowels but with nasal resonance
        t = np.arange(num_samples) / self.sample_rate
        
        f1 = phoneme_def.formant_f1
        f2 = phoneme_def.formant_f2
        f3 = phoneme_def.formant_f3
        
        formant1 = generate_formant_resonance(t, f1, 60.0, amplitude=1.0)
        formant2 = generate_formant_resonance(t, f2, 100.0, amplitude=0.5)
        formant3 = generate_formant_resonance(t, f3, 120.0, amplitude=0.25)
        
        signal = formant1 + formant2 + formant3
        envelope = create_audio_envelope(num_samples, self.sample_rate, attack_ms=5, decay_ms=15)
        
        return signal * envelope
    
    def _synthesize_liquid(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize liquid consonant (l, r)."""
        # Similar to nasals but with different formant structure
        return self._synthesize_nasal(phoneme_def, num_samples)
    
    def _synthesize_glide(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize glide consonant (w, j)."""
        # Glides are like very short vowels
        return self._synthesize_vowel(phoneme_def, num_samples)
    
    def _synthesize_plosive(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize plosive consonant (p, b, t, d, k, g)."""
        # Plosives: brief silence + burst + formant transition
        t = np.arange(num_samples) / self.sample_rate
        
        # Generate noise burst
        if phoneme_def.noise_band:
            low_freq, high_freq = phoneme_def.noise_band
            noise = np.random.uniform(-1, 1, num_samples)
            # Filter noise to frequency band (simplified)
            signal = noise * 0.3
        else:
            signal = np.zeros(num_samples)
        
        # Add formant transition if voiced
        if phoneme_def.voiced:
            # Brief formant structure
            formant = np.sin(2 * np.pi * 500 * t) * 0.2
            signal += formant
        
        # Sharp attack, quick decay
        envelope = create_audio_envelope(num_samples, self.sample_rate, attack_ms=2, decay_ms=5)
        
        return signal * envelope
    
    def _synthesize_fricative(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize fricative consonant (f, v, s, z, etc.)."""
        t = np.arange(num_samples) / self.sample_rate
        
        # Generate noise in frequency band
        if phoneme_def.noise_band:
            low_freq, high_freq = phoneme_def.noise_band
            # Generate white noise
            noise = np.random.uniform(-1, 1, num_samples)
            # Simple bandpass filter approximation
            signal = noise * 0.4
        else:
            signal = np.zeros(num_samples)
        
        # Add voicing if voiced
        if phoneme_def.voiced:
            voicing = np.sin(2 * np.pi * 150 * t) * 0.1
            signal += voicing
        
        # Smooth envelope
        envelope = create_audio_envelope(num_samples, self.sample_rate, attack_ms=10, decay_ms=15)
        
        return signal * envelope
    
    def _synthesize_affricate(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Synthesize affricate (ch, j)."""
        # Affricates: plosive + fricative
        plosive_samples = num_samples // 3
        fricative_samples = num_samples - plosive_samples
        
        plosive = self._synthesize_plosive(phoneme_def, plosive_samples)
        fricative = self._synthesize_fricative(phoneme_def, fricative_samples)
        
        return np.concatenate([plosive, fricative])
    
    def _synthesize_simple(self, phoneme_def: PhonemeDefinition, num_samples: int) -> np.ndarray:
        """Simple synthesis fallback."""
        t = np.arange(num_samples) / self.sample_rate
        freq = 200.0  # Default frequency
        signal = np.sin(2 * np.pi * freq * t)
        envelope = create_audio_envelope(num_samples, self.sample_rate, attack_ms=10, decay_ms=20)
        return signal * envelope


# Global instance
FORMANT_SYNTHESIZER = FormantSynthesizer()

