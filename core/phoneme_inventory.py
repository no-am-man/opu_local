"""
English Phoneme Inventory: Full IPA phoneme set for English.
Provides comprehensive phoneme definitions with formant frequencies and articulation parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class PhonemeDefinition:
    """Definition of a single phoneme with synthesis parameters."""
    symbol: str  # IPA symbol
    name: str  # Phoneme name
    category: str  # vowel, consonant, diphthong
    articulation: str  # plosive, fricative, nasal, liquid, etc.
    voiced: bool  # True for voiced, False for voiceless
    formant_f1: float = 0.0  # First formant frequency (Hz) - for vowels
    formant_f2: float = 0.0  # Second formant frequency (Hz) - for vowels
    formant_f3: float = 0.0  # Third formant frequency (Hz) - for vowels
    noise_band: Optional[tuple] = None  # (low_freq, high_freq) for consonants
    duration_ms: float = 100.0  # Typical duration in milliseconds


class EnglishPhonemeInventory:
    """
    Complete English phoneme inventory with ~44 phonemes.
    Organized by category with formant frequencies for synthesis.
    """
    
    def __init__(self):
        self.phonemes: Dict[str, PhonemeDefinition] = {}
        self._initialize_vowels()
        self._initialize_consonants()
        self._initialize_diphthongs()
    
    def _initialize_vowels(self):
        """Initialize vowel phonemes with formant frequencies."""
        # Formant frequencies (F1, F2, F3) in Hz for typical male voice
        vowels = [
            # High front vowels
            PhonemeDefinition("/i/", "high front unrounded", "vowel", "high", True, 270, 2290, 3010, duration_ms=120),
            PhonemeDefinition("/ɪ/", "near-high front unrounded", "vowel", "high", True, 390, 1990, 2550, duration_ms=100),
            
            # Mid front vowels
            PhonemeDefinition("/e/", "mid front unrounded", "vowel", "mid", True, 530, 1840, 2480, duration_ms=110),
            PhonemeDefinition("/ɛ/", "open-mid front unrounded", "vowel", "mid", True, 610, 1900, 2600, duration_ms=100),
            
            # Low front vowel
            PhonemeDefinition("/æ/", "near-open front unrounded", "vowel", "low", True, 660, 1720, 2410, duration_ms=120),
            
            # Low back vowel
            PhonemeDefinition("/ɑ/", "open back unrounded", "vowel", "low", True, 730, 1090, 2440, duration_ms=130),
            
            # Mid back vowels
            PhonemeDefinition("/ɔ/", "open-mid back rounded", "vowel", "mid", True, 570, 840, 2410, duration_ms=120),
            PhonemeDefinition("/o/", "mid back rounded", "vowel", "mid", True, 450, 760, 2570, duration_ms=110),
            
            # High back vowels
            PhonemeDefinition("/ʊ/", "near-high back rounded", "vowel", "high", True, 370, 950, 2670, duration_ms=100),
            PhonemeDefinition("/u/", "high back rounded", "vowel", "high", True, 300, 870, 2240, duration_ms=110),
            
            # Central vowels
            PhonemeDefinition("/ʌ/", "open-mid central unrounded", "vowel", "mid", True, 600, 1170, 2390, duration_ms=100),
            PhonemeDefinition("/ə/", "mid central unrounded (schwa)", "vowel", "mid", True, 500, 1500, 2500, duration_ms=80),
        ]
        
        for vowel in vowels:
            self.phonemes[vowel.symbol] = vowel
    
    def _initialize_consonants(self):
        """Initialize consonant phonemes with articulation parameters."""
        consonants = [
            # Plosives (stops)
            PhonemeDefinition("/p/", "voiceless bilabial plosive", "consonant", "plosive", False, noise_band=(0, 2000), duration_ms=80),
            PhonemeDefinition("/b/", "voiced bilabial plosive", "consonant", "plosive", True, noise_band=(0, 2000), duration_ms=80),
            PhonemeDefinition("/t/", "voiceless alveolar plosive", "consonant", "plosive", False, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/d/", "voiced alveolar plosive", "consonant", "plosive", True, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/k/", "voiceless velar plosive", "consonant", "plosive", False, noise_band=(1000, 6000), duration_ms=80),
            PhonemeDefinition("/g/", "voiced velar plosive", "consonant", "plosive", True, noise_band=(1000, 6000), duration_ms=80),
            
            # Fricatives
            PhonemeDefinition("/f/", "voiceless labiodental fricative", "consonant", "fricative", False, noise_band=(2000, 8000), duration_ms=120),
            PhonemeDefinition("/v/", "voiced labiodental fricative", "consonant", "fricative", True, noise_band=(2000, 8000), duration_ms=120),
            PhonemeDefinition("/θ/", "voiceless dental fricative (th)", "consonant", "fricative", False, noise_band=(3000, 8000), duration_ms=120),
            PhonemeDefinition("/ð/", "voiced dental fricative (th)", "consonant", "fricative", True, noise_band=(3000, 8000), duration_ms=120),
            PhonemeDefinition("/s/", "voiceless alveolar fricative", "consonant", "fricative", False, noise_band=(4000, 8000), duration_ms=120),
            PhonemeDefinition("/z/", "voiced alveolar fricative", "consonant", "fricative", True, noise_band=(4000, 8000), duration_ms=120),
            PhonemeDefinition("/ʃ/", "voiceless postalveolar fricative (sh)", "consonant", "fricative", False, noise_band=(2000, 6000), duration_ms=120),
            PhonemeDefinition("/ʒ/", "voiced postalveolar fricative (zh)", "consonant", "fricative", True, noise_band=(2000, 6000), duration_ms=120),
            PhonemeDefinition("/h/", "voiceless glottal fricative", "consonant", "fricative", False, noise_band=(500, 2000), duration_ms=80),
            
            # Affricates
            PhonemeDefinition("/tʃ/", "voiceless postalveolar affricate (ch)", "consonant", "affricate", False, noise_band=(2000, 6000), duration_ms=150),
            PhonemeDefinition("/dʒ/", "voiced postalveolar affricate (j)", "consonant", "affricate", True, noise_band=(2000, 6000), duration_ms=150),
            
            # Nasals
            PhonemeDefinition("/m/", "bilabial nasal", "consonant", "nasal", True, formant_f1=300, formant_f2=1200, formant_f3=2400, duration_ms=100),
            PhonemeDefinition("/n/", "alveolar nasal", "consonant", "nasal", True, formant_f1=400, formant_f2=1800, formant_f3=2800, duration_ms=100),
            PhonemeDefinition("/ŋ/", "velar nasal (ng)", "consonant", "nasal", True, formant_f1=350, formant_f2=1500, formant_f3=2500, duration_ms=100),
            
            # Liquids
            PhonemeDefinition("/l/", "alveolar lateral approximant", "consonant", "liquid", True, formant_f1=400, formant_f2=1200, formant_f3=2800, duration_ms=100),
            PhonemeDefinition("/r/", "alveolar approximant", "consonant", "liquid", True, formant_f1=400, formant_f2=1200, formant_f3=2800, duration_ms=100),
            
            # Glides
            PhonemeDefinition("/w/", "labial-velar approximant", "consonant", "glide", True, formant_f1=300, formant_f2=700, formant_f3=2200, duration_ms=80),
            PhonemeDefinition("/j/", "palatal approximant (y)", "consonant", "glide", True, formant_f1=300, formant_f2=2100, formant_f3=3000, duration_ms=80),
        ]
        
        for consonant in consonants:
            self.phonemes[consonant.symbol] = consonant
    
    def _initialize_diphthongs(self):
        """Initialize diphthong phonemes (vowel glides)."""
        diphthongs = [
            PhonemeDefinition("/aɪ/", "diphthong (eye)", "diphthong", "glide", True, formant_f1=660, formant_f2=1720, formant_f3=2410, duration_ms=200),
            PhonemeDefinition("/eɪ/", "diphthong (ay)", "diphthong", "glide", True, formant_f1=530, formant_f2=1840, formant_f3=2480, duration_ms=200),
            PhonemeDefinition("/ɔɪ/", "diphthong (oy)", "diphthong", "glide", True, formant_f1=570, formant_f2=840, formant_f3=2410, duration_ms=200),
            PhonemeDefinition("/aʊ/", "diphthong (ow)", "diphthong", "glide", True, formant_f1=660, formant_f2=1090, formant_f3=2440, duration_ms=200),
            PhonemeDefinition("/oʊ/", "diphthong (oh)", "diphthong", "glide", True, formant_f1=450, formant_f2=760, formant_f3=2570, duration_ms=200),
        ]
        
        for diphthong in diphthongs:
            self.phonemes[diphthong.symbol] = diphthong
    
    def get_phoneme(self, symbol: str) -> Optional[PhonemeDefinition]:
        """Get phoneme definition by IPA symbol."""
        return self.phonemes.get(symbol)
    
    def get_vowels(self) -> List[PhonemeDefinition]:
        """Get all vowel phonemes."""
        return [p for p in self.phonemes.values() if p.category == "vowel"]
    
    def get_consonants(self) -> List[PhonemeDefinition]:
        """Get all consonant phonemes."""
        return [p for p in self.phonemes.values() if p.category == "consonant"]
    
    def get_diphthongs(self) -> List[PhonemeDefinition]:
        """Get all diphthong phonemes."""
        return [p for p in self.phonemes.values() if p.category == "diphthong"]
    
    def get_by_articulation(self, articulation: str) -> List[PhonemeDefinition]:
        """Get phonemes by articulation type."""
        return [p for p in self.phonemes.values() if p.articulation == articulation]
    
    def get_by_voicing(self, voiced: bool) -> List[PhonemeDefinition]:
        """Get phonemes by voicing."""
        return [p for p in self.phonemes.values() if p.voiced == voiced]


# Global instance
PHONEME_INVENTORY = EnglishPhonemeInventory()

