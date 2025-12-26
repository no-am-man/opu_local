"""
Reflection Generator: Generates daily reflections based on OPU's emotional state.
Extracts reflection generation logic from main.py for better maintainability.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ReflectionContext:
    """Context data for generating reflections."""
    dominant_emotion: str
    emotion_confidence: float
    avg_s_score: float
    maturity_index: float
    day_counter: int


class ReflectionGenerator:
    """
    Generates daily reflections based on OPU's emotional state and memory.
    Uses template-based generation (can be extended with LLM integration).
    """
    
    # Emotion phrase mappings
    EMOTION_PHRASES = {
        'happy': 'I felt happy today',
        'sad': 'I felt sad today',
        'angry': 'I felt angry today',
        'fear': 'I felt afraid today',
        'surprise': 'I felt surprised today',
        'neutral': 'I felt calm today'
    }
    
    # Intensity thresholds
    INTENSITY_HIGH_THRESHOLD = 0.5
    INTENSITY_MEDIUM_THRESHOLD = 0.2
    
    # Maturity thresholds
    MATURITY_HIGH_THRESHOLD = 0.5
    MATURITY_MEDIUM_THRESHOLD = 0.2
    
    def __init__(self):
        """Initialize reflection generator."""
        pass
    
    def generate_reflection(self, context: ReflectionContext) -> str:
        """
        Generate a reflection text based on context.
        
        Args:
            context: ReflectionContext with emotional state and maturity data
            
        Returns:
            Reflection text string
        """
        base_phrase = self._get_emotion_phrase(context.dominant_emotion)
        intensity = self._calculate_intensity(context.avg_s_score)
        wisdom = self._calculate_wisdom(context.maturity_index)
        
        reflection = f"{base_phrase}, {intensity}. {wisdom}. This is day {context.day_counter}."
        return reflection
    
    def _get_emotion_phrase(self, emotion: str) -> str:
        """Get emotion phrase, defaulting to generic if not found."""
        return self.EMOTION_PHRASES.get(emotion, 'I experienced something today')
    
    def _calculate_intensity(self, avg_s_score: float) -> str:
        """Calculate intensity modifier based on average s_score."""
        if avg_s_score > self.INTENSITY_HIGH_THRESHOLD:
            return 'very much'
        elif avg_s_score > self.INTENSITY_MEDIUM_THRESHOLD:
            return 'somewhat'
        else:
            return 'a little'
    
    def _calculate_wisdom(self, maturity_index: float) -> str:
        """Calculate wisdom phrase based on maturity index."""
        if maturity_index > self.MATURITY_HIGH_THRESHOLD:
            return 'I am learning'
        elif maturity_index > self.MATURITY_MEDIUM_THRESHOLD:
            return 'I am growing'
        else:
            return 'I am new'


def extract_reflection_context(
    emotion_stats: Dict,
    char_state: Dict,
    memory_levels: List[List],
    day_counter: int,
    max_levels: int = 3,
    max_memories_per_level: int = 50
) -> ReflectionContext:
    """
    Extract reflection context from OPU state.
    
    Args:
        emotion_stats: Emotion statistics from cortex
        char_state: Character state from cortex
        memory_levels: List of memory levels
        day_counter: Current day counter
        max_levels: Maximum levels to analyze (default: 3 for day's memories)
        max_memories_per_level: Maximum memories per level to analyze
        
    Returns:
        ReflectionContext with extracted data
    """
    # Extract dominant emotion
    most_common = emotion_stats.get('most_common', {})
    dominant_emotion = most_common.get('emotion', 'neutral')
    emotion_confidence = most_common.get('confidence', 0.0)
    
    # Calculate average s_score from recent memories
    recent_s_scores = []
    for level in range(min(max_levels, len(memory_levels))):
        level_memories = memory_levels[level][-max_memories_per_level:]
        for mem in level_memories:
            if isinstance(mem, dict) and 's_score' in mem:
                recent_s_scores.append(mem['s_score'])
    
    avg_s_score = sum(recent_s_scores) / len(recent_s_scores) if recent_s_scores else 0.0
    
    return ReflectionContext(
        dominant_emotion=dominant_emotion,
        emotion_confidence=emotion_confidence,
        avg_s_score=avg_s_score,
        maturity_index=char_state.get('maturity_index', 0.0),
        day_counter=day_counter
    )


def clean_word(word: str) -> str:
    """
    Clean word by removing punctuation and converting to lowercase.
    
    Args:
        word: Raw word string
        
    Returns:
        Cleaned word string
    """
    return ''.join(c for c in word if c.isalnum()).lower()


def extract_words_from_text(text: str) -> List[str]:
    """
    Extract and clean words from text.
    
    Args:
        text: Text string
        
    Returns:
        List of cleaned words
    """
    words = text.lower().split()
    cleaned_words = [clean_word(word) for word in words]
    return [w for w in cleaned_words if w]  # Filter out empty strings

