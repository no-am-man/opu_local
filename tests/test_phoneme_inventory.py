"""
Tests for core/phoneme_inventory.py - English Phoneme Inventory
"""

import pytest
from core.phoneme_inventory import PHONEME_INVENTORY, PhonemeDefinition


class TestPhonemeInventory:
    """Test suite for EnglishPhonemeInventory."""
    
    def test_inventory_initialized(self):
        """Test that inventory is initialized with phonemes."""
        assert len(PHONEME_INVENTORY.phonemes) > 0
        assert len(PHONEME_INVENTORY.phonemes) >= 40  # Should have ~41 phonemes
    
    def test_get_vowels(self):
        """Test getting vowel phonemes."""
        vowels = PHONEME_INVENTORY.get_vowels()
        assert len(vowels) > 0
        assert all(v.category == "vowel" for v in vowels)
        assert len(vowels) == 12  # Should have 12 vowels
    
    def test_get_consonants(self):
        """Test getting consonant phonemes."""
        consonants = PHONEME_INVENTORY.get_consonants()
        assert len(consonants) > 0
        assert all(c.category == "consonant" for c in consonants)
        assert len(consonants) == 24  # Should have 24 consonants
    
    def test_get_diphthongs(self):
        """Test getting diphthong phonemes."""
        diphthongs = PHONEME_INVENTORY.get_diphthongs()
        assert len(diphthongs) > 0
        assert all(d.category == "diphthong" for d in diphthongs)
        assert len(diphthongs) == 5  # Should have 5 diphthongs
    
    def test_get_phoneme(self):
        """Test getting phoneme by symbol."""
        # Test vowel
        vowel = PHONEME_INVENTORY.get_phoneme("/i/")
        assert vowel is not None
        assert vowel.category == "vowel"
        assert vowel.symbol == "/i/"
        
        # Test consonant
        consonant = PHONEME_INVENTORY.get_phoneme("/s/")
        assert consonant is not None
        assert consonant.category == "consonant"
        assert consonant.articulation == "fricative"
        
        # Test non-existent
        missing = PHONEME_INVENTORY.get_phoneme("/xyz/")
        assert missing is None
    
    def test_get_by_articulation(self):
        """Test getting phonemes by articulation type."""
        plosives = PHONEME_INVENTORY.get_by_articulation("plosive")
        assert len(plosives) > 0
        assert all(p.articulation == "plosive" for p in plosives)
        assert len(plosives) == 6  # p, b, t, d, k, g
        
        fricatives = PHONEME_INVENTORY.get_by_articulation("fricative")
        assert len(fricatives) > 0
        assert all(f.articulation == "fricative" for f in fricatives)
    
    def test_get_by_voicing(self):
        """Test getting phonemes by voicing."""
        voiced = PHONEME_INVENTORY.get_by_voicing(True)
        assert len(voiced) > 0
        assert all(v.voiced for v in voiced)
        
        voiceless = PHONEME_INVENTORY.get_by_voicing(False)
        assert len(voiceless) > 0
        assert all(not v.voiced for v in voiceless)
    
    def test_vowel_formants(self):
        """Test that vowels have formant frequencies."""
        vowels = PHONEME_INVENTORY.get_vowels()
        for vowel in vowels:
            assert vowel.formant_f1 > 0
            assert vowel.formant_f2 > 0
            assert vowel.formant_f3 > 0
    
    def test_consonant_noise_bands(self):
        """Test that consonants have noise bands or formants."""
        consonants = PHONEME_INVENTORY.get_consonants()
        for consonant in consonants:
            # Should have either noise_band or formants
            has_noise = consonant.noise_band is not None
            has_formants = (consonant.formant_f1 > 0 or 
                          consonant.formant_f2 > 0 or 
                          consonant.formant_f3 > 0)
            assert has_noise or has_formants, f"Consonant {consonant.symbol} has no synthesis parameters"
    
    def test_phoneme_duration(self):
        """Test that all phonemes have duration."""
        for phoneme in PHONEME_INVENTORY.phonemes.values():
            assert phoneme.duration_ms > 0

