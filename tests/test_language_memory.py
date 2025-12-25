"""
Tests for core/language_memory.py - Language Memory System
"""

import pytest
import time
from core.language_memory import LanguageMemory, WordEntry


class TestLanguageMemory:
    """Test suite for LanguageMemory class."""
    
    def test_init(self):
        """Test LanguageMemory initialization."""
        memory = LanguageMemory()
        assert len(memory.words) == 0
        assert len(memory.word_frequency) == 0
        assert len(memory.phoneme_to_words) == 0
        assert len(memory.word_sequences) == 0
    
    def test_learn_word(self):
        """Test learning a new word."""
        memory = LanguageMemory()
        entry = memory.learn_word("hello", phonemes=["/h/", "/ɛ/", "/l/", "/oʊ/"])
        
        assert entry.word == "hello"
        assert entry.phonemes == ["/h/", "/ɛ/", "/l/", "/oʊ/"]
        assert entry.frequency == 1
        assert "hello" in memory.words
        assert memory.word_frequency["hello"] == 1
    
    def test_learn_word_updates_frequency(self):
        """Test that learning same word updates frequency."""
        memory = LanguageMemory()
        entry1 = memory.learn_word("hello")
        entry2 = memory.learn_word("hello")
        
        assert entry2.frequency == 2
        assert memory.word_frequency["hello"] == 2
    
    def test_learn_word_phoneme_indexing(self):
        """Test that phonemes are indexed for word lookup."""
        memory = LanguageMemory()
        memory.learn_word("hello", phonemes=["/h/", "/ɛ/", "/l/", "/oʊ/"])
        
        # Check phoneme indexing
        assert "/h/" in memory.phoneme_to_words
        assert "hello" in memory.phoneme_to_words["/h/"]
        assert "/ɛ/" in memory.phoneme_to_words
        assert "hello" in memory.phoneme_to_words["/ɛ/"]
    
    def test_learn_word_with_emotion(self):
        """Test learning word with emotion association."""
        memory = LanguageMemory()
        entry = memory.learn_word("happy", emotion="joy", s_score=2.5)
        
        assert "joy" in entry.emotions
        assert entry.emotions["joy"] == 1
    
    def test_learn_phrase(self):
        """Test learning a phrase."""
        memory = LanguageMemory()
        memory.learn_phrase(["hello", "world"], s_score=3.0)
        
        assert len(memory.word_sequences) == 1
        assert memory.word_sequences[0] == ["hello", "world"]
    
    def test_learn_phrase_creates_associations(self):
        """Test that learning phrase creates word associations."""
        memory = LanguageMemory()
        memory.learn_word("hello")
        memory.learn_word("world")
        memory.learn_phrase(["hello", "world"], s_score=3.0)
        
        entry = memory.get_word("hello")
        assert "world" in entry.associations
        assert entry.associations["world"] > 0
    
    def test_get_word(self):
        """Test getting word entry."""
        memory = LanguageMemory()
        memory.learn_word("hello")
        
        entry = memory.get_word("hello")
        assert entry is not None
        assert entry.word == "hello"
        
        missing = memory.get_word("nonexistent")
        assert missing is None
    
    def test_get_words_by_phoneme(self):
        """Test getting words by phoneme."""
        memory = LanguageMemory()
        memory.learn_word("hello", phonemes=["/h/", "/ɛ/", "/l/", "/oʊ/"])
        memory.learn_word("hi", phonemes=["/h/", "/aɪ/"])
        
        words_with_h = memory.get_words_by_phoneme("/h/")
        assert "hello" in words_with_h
        assert "hi" in words_with_h
    
    def test_get_most_frequent_words(self):
        """Test getting most frequent words."""
        memory = LanguageMemory()
        memory.learn_word("hello")
        memory.learn_word("hello")
        memory.learn_word("world")
        
        top_words = memory.get_most_frequent_words(2)
        assert len(top_words) == 2
        assert top_words[0].word == "hello"  # Most frequent
        assert top_words[0].frequency == 2
    
    def test_get_recent_words(self):
        """Test getting recently encountered words."""
        memory = LanguageMemory()
        memory.learn_word("first")
        time.sleep(0.01)  # Small delay
        memory.learn_word("second")
        
        recent = memory.get_recent_words(1)
        assert len(recent) == 1
        assert recent[0].word == "second"
    
    def test_get_word_associations(self):
        """Test getting word associations."""
        memory = LanguageMemory()
        memory.learn_word("hello")
        memory.learn_word("world")
        memory.learn_phrase(["hello", "world"], s_score=2.0)
        
        associations = memory.get_word_associations("hello", count=5)
        assert len(associations) > 0
        assert any(word == "world" for word, _ in associations)
    
    def test_search_words(self):
        """Test searching for words."""
        memory = LanguageMemory()
        memory.learn_word("hello")
        memory.learn_word("world")
        memory.learn_word("hell")
        
        results = memory.search_words("hell")
        assert len(results) >= 2  # "hello" and "hell"
        assert any(w.word == "hello" for w in results)
        assert any(w.word == "hell" for w in results)
    
    def test_get_statistics(self):
        """Test getting language memory statistics."""
        memory = LanguageMemory()
        memory.learn_word("hello")
        memory.learn_word("world")
        memory.learn_phrase(["hello", "world"])
        
        stats = memory.get_statistics()
        assert stats['total_words'] == 2
        assert stats['total_phrases'] == 1
        assert 'top_words' in stats
    
    def test_max_words_limit(self):
        """Test that max_words limit is enforced."""
        memory = LanguageMemory(max_words=3)
        
        # Add more than max_words
        memory.learn_word("word1")
        memory.learn_word("word2")
        memory.learn_word("word3")
        memory.learn_word("word4")
        
        # Should have removed least frequent
        assert len(memory.words) <= 3
    
    def test_export_import_vocabulary(self):
        """Test exporting and importing vocabulary."""
        memory = LanguageMemory()
        memory.learn_word("hello", phonemes=["/h/", "/ɛ/", "/l/", "/oʊ/"], emotion="joy")
        memory.learn_phrase(["hello", "world"])
        
        # Export
        vocabulary = memory.export_vocabulary()
        assert len(vocabulary) == 1
        assert vocabulary[0]['word'] == "hello"
        
        # Import
        memory2 = LanguageMemory()
        memory2.import_vocabulary(vocabulary)
        assert "hello" in memory2.words
        assert memory2.words["hello"].phonemes == ["/h/", "/ɛ/", "/l/", "/oʊ/"]

