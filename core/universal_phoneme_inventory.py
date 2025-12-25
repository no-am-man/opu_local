"""
Universal Phoneme Inventory: Complete IPA phoneme set for all languages.
Extends English inventory with phonemes from major language families worldwide.

Based on IPA (International Phonetic Alphabet) - the universal phonetic system
that can represent sounds from all human languages.
"""

from typing import Dict, List, Optional, Set, Callable
from core.phoneme_inventory import PhonemeDefinition, EnglishPhonemeInventory


class UniversalPhonemeInventory(EnglishPhonemeInventory):
    """
    Universal phoneme inventory supporting all major language families.
    Extends EnglishPhonemeInventory with phonemes from:
    - Romance languages (Spanish, French, Italian, Portuguese)
    - Germanic languages (German, Dutch, Swedish)
    - Slavic languages (Russian, Polish, Czech)
    - Semitic languages (Arabic, Hebrew)
    - Sino-Tibetan (Mandarin, Cantonese)
    - Dravidian (Tamil, Telugu)
    - Japonic (Japanese)
    - Koreanic (Korean)
    - And more...
    
    Total: ~150+ phonemes covering all major language families.
    """
    
    # Language family registry - maps family name to initialization method
    _FAMILY_REGISTRY: Dict[str, Callable] = {}
    
    def __init__(self, enabled_families: Optional[Set[str]] = None):
        """
        Initialize universal phoneme inventory.
        
        Args:
            enabled_families: Set of language families to include.
                            If None, includes all families.
                            Options: 'romance', 'germanic', 'slavic', 'semitic',
                                   'sino-tibetan', 'dravidian', 'japonic', 'koreanic',
                                   'african', 'polynesian', 'native_american', etc.
        """
        # Initialize base English inventory
        super().__init__()
        
        # Default: enable all families
        if enabled_families is None:
            enabled_families = self._get_all_families()
        
        self.enabled_families = enabled_families
        
        # Add phonemes from enabled language families using registry
        self._add_phonemes_from_families(enabled_families)
    
    @classmethod
    def _get_all_families(cls) -> Set[str]:
        """Get set of all available language families."""
        return {
            'romance', 'germanic', 'slavic', 'semitic', 'sino-tibetan',
            'dravidian', 'japonic', 'koreanic', 'african', 'polynesian',
            'native_american', 'austronesian', 'uralic', 'turkic'
        }
    
    def _add_phonemes_from_families(self, families: Set[str]):
        """Add phonemes from enabled language families using registry pattern."""
        for family in families:
            method = self._get_family_method(family)
            if method:
                method()
    
    def _get_family_method(self, family: str) -> Optional[Callable]:
        """Get initialization method for a language family."""
        method_name = f"_add_{family}_phonemes"
        return getattr(self, method_name, None)
    
    # Helper methods for adding phonemes
    def _add_phoneme_safe(self, phoneme: PhonemeDefinition):
        """Safely add a phoneme if it doesn't already exist."""
        if phoneme.symbol not in self.phonemes:
            self.phonemes[phoneme.symbol] = phoneme
    
    def _add_phonemes_batch(self, phonemes: List[PhonemeDefinition]):
        """Add multiple phonemes safely."""
        for phoneme in phonemes:
            self._add_phoneme_safe(phoneme)
    
    def _add_phoneme_if_missing(self, symbol: str, **kwargs):
        """Add a single phoneme if missing, using PhonemeDefinition constructor."""
        if symbol not in self.phonemes:
            self.phonemes[symbol] = PhonemeDefinition(symbol=symbol, **kwargs)
    
    # Language family phoneme definitions
    def _add_romance_phonemes(self):
        """Add phonemes from Romance languages (Spanish, French, Italian, Portuguese)."""
        # Spanish: trilled R, palatal nasal
        self._add_phoneme_if_missing(
            "/r/", name="alveolar trill", category="consonant", articulation="trill",
            voiced=True, formant_f1=400, formant_f2=1200, formant_f3=2800, duration_ms=100
        )
        
        # French: nasal vowels
        nasal_vowels = [
            PhonemeDefinition("/ɑ̃/", "open back nasal", "vowel", "nasal", True, 700, 1100, 2400, duration_ms=120),
            PhonemeDefinition("/ɛ̃/", "open-mid front nasal", "vowel", "nasal", True, 600, 1900, 2600, duration_ms=120),
            PhonemeDefinition("/ɔ̃/", "open-mid back nasal", "vowel", "nasal", True, 570, 840, 2410, duration_ms=120),
            PhonemeDefinition("/œ̃/", "open-mid front rounded nasal", "vowel", "nasal", True, 600, 1500, 2500, duration_ms=120),
        ]
        self._add_phonemes_batch(nasal_vowels)
        
        # Uvular R (French, German)
        self._add_phoneme_if_missing(
            "/ʁ/", name="uvular fricative", category="consonant", articulation="fricative",
            voiced=True, noise_band=(500, 2000), duration_ms=100
        )
        
        # Palatal nasal (Spanish "ñ", Italian "gn")
        self._add_phoneme_if_missing(
            "/ɲ/", name="palatal nasal", category="consonant", articulation="nasal",
            voiced=True, formant_f1=350, formant_f2=2000, formant_f3=3000, duration_ms=100
        )
        
        # Palatal lateral (Italian "gl")
        self._add_phoneme_if_missing(
            "/ʎ/", name="palatal lateral", category="consonant", articulation="liquid",
            voiced=True, formant_f1=350, formant_f2=2000, formant_f3=3000, duration_ms=100
        )
    
    def _add_germanic_phonemes(self):
        """Add phonemes from Germanic languages (German, Dutch, Swedish)."""
        # German: Umlaut vowels
        umlaut_vowels = [
            PhonemeDefinition("/y/", "high front rounded", "vowel", "high", True, 270, 2290, 3010, duration_ms=110),
            PhonemeDefinition("/ø/", "mid front rounded", "vowel", "mid", True, 530, 1840, 2480, duration_ms=110),
            PhonemeDefinition("/œ/", "open-mid front rounded", "vowel", "mid", True, 610, 1900, 2600, duration_ms=110),
        ]
        self._add_phonemes_batch(umlaut_vowels)
        
        # Voiceless velar fricative (German "ch" in "Bach")
        self._add_phoneme_if_missing(
            "/x/", name="voiceless velar fricative", category="consonant", articulation="fricative",
            voiced=False, noise_band=(1000, 4000), duration_ms=120
        )
        
        # Voiced velar fricative (Dutch "g")
        self._add_phoneme_if_missing(
            "/ɣ/", name="voiced velar fricative", category="consonant", articulation="fricative",
            voiced=True, noise_band=(1000, 4000), duration_ms=120
        )
    
    def _add_slavic_phonemes(self):
        """Add phonemes from Slavic languages (Russian, Polish, Czech)."""
        # Palatalized consonants (Russian soft consonants)
        palatalized = [
            PhonemeDefinition("/tʲ/", "palatalized alveolar plosive", "consonant", "plosive", False, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/dʲ/", "palatalized alveolar plosive", "consonant", "plosive", True, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/nʲ/", "palatalized alveolar nasal", "consonant", "nasal", True, formant_f1=400, formant_f2=2000, formant_f3=3000, duration_ms=100),
        ]
        self._add_phonemes_batch(palatalized)
        
        # Voiceless alveolar affricate (Polish "c")
        self._add_phoneme_if_missing(
            "/ts/", name="voiceless alveolar affricate", category="consonant", articulation="affricate",
            voiced=False, noise_band=(4000, 8000), duration_ms=150
        )
        
        # Voiced alveolar affricate (Polish "dz")
        self._add_phoneme_if_missing(
            "/dz/", name="voiced alveolar affricate", category="consonant", articulation="affricate",
            voiced=True, noise_band=(4000, 8000), duration_ms=150
        )
    
    def _add_semitic_phonemes(self):
        """Add phonemes from Semitic languages (Arabic, Hebrew)."""
        # Pharyngeal consonants (Arabic)
        pharyngeal = [
            PhonemeDefinition("/ħ/", "voiceless pharyngeal fricative", "consonant", "fricative", False, noise_band=(200, 1000), duration_ms=120),
            PhonemeDefinition("/ʕ/", "voiced pharyngeal fricative", "consonant", "fricative", True, noise_band=(200, 1000), duration_ms=120),
        ]
        self._add_phonemes_batch(pharyngeal)
        
        # Emphatic consonants (Arabic)
        emphatic = [
            PhonemeDefinition("/tˤ/", "emphatic alveolar plosive", "consonant", "plosive", False, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/dˤ/", "emphatic alveolar plosive", "consonant", "plosive", True, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/sˤ/", "emphatic alveolar fricative", "consonant", "fricative", False, noise_band=(4000, 8000), duration_ms=120),
        ]
        self._add_phonemes_batch(emphatic)
        
        # Uvular plosive (Arabic "q")
        self._add_phoneme_if_missing(
            "/q/", name="voiceless uvular plosive", category="consonant", articulation="plosive",
            voiced=False, noise_band=(500, 2000), duration_ms=80
        )
    
    def _add_sino_tibetan_phonemes(self):
        """Add phonemes from Sino-Tibetan languages (Mandarin, Cantonese)."""
        # Retroflex consonants (Mandarin)
        retroflex = [
            PhonemeDefinition("/ʈ/", "voiceless retroflex plosive", "consonant", "plosive", False, noise_band=(2000, 6000), duration_ms=80),
            PhonemeDefinition("/ɖ/", "voiced retroflex plosive", "consonant", "plosive", True, noise_band=(2000, 6000), duration_ms=80),
            PhonemeDefinition("/ʂ/", "voiceless retroflex fricative", "consonant", "fricative", False, noise_band=(2000, 6000), duration_ms=120),
            PhonemeDefinition("/ʐ/", "voiced retroflex fricative", "consonant", "fricative", True, noise_band=(2000, 6000), duration_ms=120),
        ]
        self._add_phonemes_batch(retroflex)
        
        # Alveolo-palatal consonants (Mandarin)
        alveolo_palatal = [
            PhonemeDefinition("/tɕ/", "voiceless alveolo-palatal affricate", "consonant", "affricate", False, noise_band=(2000, 6000), duration_ms=150),
            PhonemeDefinition("/dʑ/", "voiced alveolo-palatal affricate", "consonant", "affricate", True, noise_band=(2000, 6000), duration_ms=150),
            PhonemeDefinition("/ɕ/", "voiceless alveolo-palatal fricative", "consonant", "fricative", False, noise_band=(2000, 6000), duration_ms=120),
        ]
        self._add_phonemes_batch(alveolo_palatal)
        
        # Additional vowels
        self._add_phoneme_if_missing(
            "/ɨ/", name="close central unrounded", category="vowel", articulation="high",
            voiced=True, formant_f1=300, formant_f2=1500, formant_f3=2500, duration_ms=100
        )
    
    def _add_dravidian_phonemes(self):
        """Add phonemes from Dravidian languages (Tamil, Telugu)."""
        # Retroflex lateral (Tamil)
        self._add_phoneme_if_missing(
            "/ɭ/", name="retroflex lateral", category="consonant", articulation="liquid",
            voiced=True, formant_f1=400, formant_f2=1200, formant_f3=2800, duration_ms=100
        )
        
        # Retroflex nasal (Tamil, Telugu)
        self._add_phoneme_if_missing(
            "/ɳ/", name="retroflex nasal", category="consonant", articulation="nasal",
            voiced=True, formant_f1=400, formant_f2=1200, formant_f3=2800, duration_ms=100
        )
    
    def _add_japonic_phonemes(self):
        """Add phonemes from Japonic languages (Japanese)."""
        # Length marker (can be applied to any phoneme)
        self._add_phoneme_if_missing(
            "/ː/", name="length marker", category="consonant", articulation="geminate",
            voiced=True, duration_ms=200
        )
    
    def _add_koreanic_phonemes(self):
        """Add phonemes from Koreanic languages (Korean)."""
        # Tense consonants (Korean)
        tense = [
            PhonemeDefinition("/p͈/", "tense bilabial plosive", "consonant", "plosive", False, noise_band=(0, 2000), duration_ms=80),
            PhonemeDefinition("/t͈/", "tense alveolar plosive", "consonant", "plosive", False, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/k͈/", "tense velar plosive", "consonant", "plosive", False, noise_band=(1000, 6000), duration_ms=80),
        ]
        self._add_phonemes_batch(tense)
    
    def _add_african_phonemes(self):
        """Add phonemes from African languages (click consonants, etc.)."""
        # Click consonants (Khoisan, Bantu languages)
        clicks = [
            PhonemeDefinition("/ʘ/", "bilabial click", "consonant", "click", False, noise_band=(500, 3000), duration_ms=80),
            PhonemeDefinition("/ǀ/", "dental click", "consonant", "click", False, noise_band=(2000, 6000), duration_ms=80),
            PhonemeDefinition("/ǃ/", "alveolar click", "consonant", "click", False, noise_band=(2000, 6000), duration_ms=80),
            PhonemeDefinition("/ǂ/", "palato-alveolar click", "consonant", "click", False, noise_band=(2000, 6000), duration_ms=80),
            PhonemeDefinition("/ǁ/", "alveolar lateral click", "consonant", "click", False, noise_band=(2000, 6000), duration_ms=80),
        ]
        self._add_phonemes_batch(clicks)
    
    def _add_polynesian_phonemes(self):
        """Add phonemes from Polynesian languages (Hawaiian, Maori, etc.)."""
        # Glottal stop
        self._add_phoneme_if_missing(
            "/ʔ/", name="glottal stop", category="consonant", articulation="plosive",
            voiced=False, noise_band=(0, 500), duration_ms=50
        )
    
    def _add_native_american_phonemes(self):
        """Add phonemes from Native American languages."""
        # Ejective consonants (Navajo, Quechua, etc.)
        ejectives = [
            PhonemeDefinition("/pʼ/", "ejective bilabial plosive", "consonant", "plosive", False, noise_band=(0, 2000), duration_ms=80),
            PhonemeDefinition("/tʼ/", "ejective alveolar plosive", "consonant", "plosive", False, noise_band=(2000, 8000), duration_ms=80),
            PhonemeDefinition("/kʼ/", "ejective velar plosive", "consonant", "plosive", False, noise_band=(1000, 6000), duration_ms=80),
        ]
        self._add_phonemes_batch(ejectives)
    
    def _add_austronesian_phonemes(self):
        """Add phonemes from Austronesian languages (Indonesian, Tagalog, etc.)."""
        # Most Austronesian phonemes already in base inventory
        pass
    
    def _add_uralic_phonemes(self):
        """Add phonemes from Uralic languages (Finnish, Hungarian, etc.)."""
        # Most Uralic phonemes already in base inventory
        # Length is typically marked with /ː/ which we already have
        pass
    
    def _add_turkic_phonemes(self):
        """Add phonemes from Turkic languages (Turkish, Kazakh, etc.)."""
        # Most Turkic phonemes already in base inventory
        pass
    
    def get_phoneme_count(self) -> int:
        """Get total number of phonemes in inventory."""
        return len(self.phonemes)
    
    def get_enabled_families(self) -> Set[str]:
        """Get set of enabled language families."""
        return self.enabled_families.copy()
    
    def get_phonemes_by_family(self, family: str) -> List[PhonemeDefinition]:
        """
        Get phonemes from a specific language family.
        
        Note: This is approximate since many phonemes are shared across families.
        """
        # This would require tracking which family each phoneme belongs to
        # For now, return all phonemes if family is enabled
        if family in self.enabled_families:
            return list(self.phonemes.values())
        return []


# Global instance - can be configured via config.py
UNIVERSAL_PHONEME_INVENTORY = UniversalPhonemeInventory()
