"""
Language Memory: Word learning and semantic association.
Stores word meanings, associations, and learns from experience.
Integrates with OPU memory abstraction system.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class WordEntry:
    """Entry for a learned word."""
    word: str
    phonemes: List[str]  # IPA phoneme sequence
    frequency: int = 0  # How many times encountered
    first_seen: float = field(default_factory=time.time)  # Timestamp
    last_seen: float = field(default_factory=time.time)  # Timestamp
    associations: Dict[str, float] = field(default_factory=dict)  # word -> strength
    emotions: Dict[str, float] = field(default_factory=dict)  # emotion -> frequency
    contexts: List[str] = field(default_factory=list)  # Context strings
    semantic_vector: Optional[List[float]] = None  # Semantic embedding (future)


class LanguageMemory:
    """
    Language memory system for word learning and semantic association.
    Learns words from speech recognition and associates them with experiences.
    """
    
    def __init__(self, max_words: int = 10000):
        self.max_words = max_words
        self.words: Dict[str, WordEntry] = {}
        self.word_frequency: Dict[str, int] = {}
        self.phoneme_to_words: Dict[str, Set[str]] = {}  # phoneme -> set of words
        self.word_sequences: List[List[str]] = []  # Learned word sequences (phrases)
        self.max_sequences = 1000
    
    def learn_word(self, word: str, phonemes: Optional[List[str]] = None, 
                   context: Optional[str] = None, emotion: Optional[str] = None,
                   s_score: float = 0.0) -> WordEntry:
        """
        Learn a new word or update existing word entry.
        
        Args:
            word: Word text
            phonemes: IPA phoneme sequence (optional)
            context: Context where word was encountered
            emotion: Associated emotion (optional)
            s_score: Surprise score when word was encountered
            
        Returns:
            WordEntry for the word
        """
        word_lower = word.lower().strip()
        
        if word_lower not in self.words:
            # New word
            if len(self.words) >= self.max_words:
                # Remove least frequent word
                self._remove_least_frequent()
            
            self.words[word_lower] = WordEntry(
                word=word_lower,
                phonemes=phonemes or [],
                frequency=1,
                first_seen=time.time(),
                last_seen=time.time()
            )
            
            # Index by phonemes
            if phonemes:
                for phoneme in phonemes:
                    if phoneme not in self.phoneme_to_words:
                        self.phoneme_to_words[phoneme] = set()
                    self.phoneme_to_words[phoneme].add(word_lower)
        else:
            # Update existing word
            entry = self.words[word_lower]
            entry.frequency += 1
            entry.last_seen = time.time()
        
        entry = self.words[word_lower]
        
        # Add context
        if context:
            entry.contexts.append(context)
            # Keep only recent contexts
            if len(entry.contexts) > 10:
                entry.contexts = entry.contexts[-10:]
        
        # Add emotion association
        if emotion:
            if emotion not in entry.emotions:
                entry.emotions[emotion] = 0
            entry.emotions[emotion] += 1
        
        # Update word frequency
        self.word_frequency[word_lower] = entry.frequency
        
        return entry
    
    def learn_phrase(self, words: List[str], s_score: float = 0.0):
        """
        Learn a phrase (sequence of words).
        
        Args:
            words: List of words in phrase
            s_score: Surprise score when phrase was encountered
        """
        if len(words) < 2:
            return
        
        # Store phrase
        word_sequence = [w.lower().strip() for w in words]
        self.word_sequences.append(word_sequence)
        
        # Keep only recent sequences
        if len(self.word_sequences) > self.max_sequences:
            self.word_sequences = self.word_sequences[-self.max_sequences:]
        
        # Create associations between adjacent words
        for i in range(len(word_sequence) - 1):
            word1 = word_sequence[i]
            word2 = word_sequence[i + 1]
            
            if word1 in self.words and word2 in self.words:
                # Strengthen association
                if word2 not in self.words[word1].associations:
                    self.words[word1].associations[word2] = 0.0
                self.words[word1].associations[word2] += 1.0 + s_score * 0.1
    
    def get_word(self, word: str) -> Optional[WordEntry]:
        """Get word entry if it exists."""
        return self.words.get(word.lower().strip())
    
    def get_words_by_phoneme(self, phoneme: str) -> List[str]:
        """Get all words containing a specific phoneme."""
        return list(self.phoneme_to_words.get(phoneme, set()))
    
    def get_most_frequent_words(self, count: int = 10) -> List[WordEntry]:
        """Get most frequently encountered words."""
        sorted_words = sorted(self.words.values(), key=lambda w: w.frequency, reverse=True)
        return sorted_words[:count]
    
    def get_recent_words(self, count: int = 10) -> List[WordEntry]:
        """Get most recently encountered words."""
        sorted_words = sorted(self.words.values(), key=lambda w: w.last_seen, reverse=True)
        return sorted_words[:count]
    
    def get_word_associations(self, word: str, count: int = 5) -> List[tuple]:
        """
        Get words associated with a given word.
        
        Returns:
            List of (word, strength) tuples, sorted by strength
        """
        entry = self.get_word(word)
        if not entry:
            return []
        
        associations = sorted(entry.associations.items(), key=lambda x: x[1], reverse=True)
        return associations[:count]
    
    def search_words(self, query: str) -> List[WordEntry]:
        """Search for words matching query (simple substring match)."""
        query_lower = query.lower().strip()
        matches = []
        
        for word, entry in self.words.items():
            if query_lower in word or word in query_lower:
                matches.append(entry)
        
        # Sort by frequency
        matches.sort(key=lambda w: w.frequency, reverse=True)
        return matches
    
    def get_statistics(self) -> Dict:
        """Get language memory statistics."""
        total_words = len(self.words)
        total_phrases = len(self.word_sequences)
        total_phonemes = len(self.phoneme_to_words)
        
        # Average word frequency
        avg_frequency = sum(w.frequency for w in self.words.values()) / max(total_words, 1)
        
        # Most common words
        top_words = [w.word for w in self.get_most_frequent_words(10)]
        
        return {
            'total_words': total_words,
            'total_phrases': total_phrases,
            'total_phonemes': total_phonemes,
            'average_frequency': avg_frequency,
            'top_words': top_words
        }
    
    def _remove_least_frequent(self):
        """Remove the least frequent word to make room."""
        if not self.words:
            return
        
        # Find least frequent word
        least_frequent = min(self.words.items(), key=lambda x: x[1].frequency)
        word_to_remove = least_frequent[0]
        entry = least_frequent[1]
        
        # Remove from phoneme index
        if entry.phonemes:
            for phoneme in entry.phonemes:
                if phoneme in self.phoneme_to_words:
                    self.phoneme_to_words[phoneme].discard(word_to_remove)
                    if not self.phoneme_to_words[phoneme]:
                        del self.phoneme_to_words[phoneme]
        
        # Remove word
        del self.words[word_to_remove]
        if word_to_remove in self.word_frequency:
            del self.word_frequency[word_to_remove]
    
    def export_vocabulary(self) -> List[Dict]:
        """Export vocabulary for persistence."""
        return [
            {
                'word': entry.word,
                'phonemes': entry.phonemes,
                'frequency': entry.frequency,
                'first_seen': entry.first_seen,
                'last_seen': entry.last_seen,
                'associations': entry.associations,
                'emotions': entry.emotions,
                'contexts': entry.contexts[-5:]  # Keep only recent contexts
            }
            for entry in self.words.values()
        ]
    
    def import_vocabulary(self, vocabulary: List[Dict]):
        """Import vocabulary from persistence."""
        for word_data in vocabulary:
            word = word_data.get('word', '').lower().strip()
            if word:
                entry = WordEntry(
                    word=word,
                    phonemes=word_data.get('phonemes', []),
                    frequency=word_data.get('frequency', 1),
                    first_seen=word_data.get('first_seen', time.time()),
                    last_seen=word_data.get('last_seen', time.time()),
                    associations=word_data.get('associations', {}),
                    emotions=word_data.get('emotions', {}),
                    contexts=word_data.get('contexts', [])
                )
                self.words[word] = entry
                self.word_frequency[word] = entry.frequency
                
                # Rebuild phoneme index
                if entry.phonemes:
                    for phoneme in entry.phonemes:
                        if phoneme not in self.phoneme_to_words:
                            self.phoneme_to_words[phoneme] = set()
                        self.phoneme_to_words[phoneme].add(word)

