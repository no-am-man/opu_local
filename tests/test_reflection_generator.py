"""
Tests for utils/reflection_generator.py - Reflection generation utilities.
Targets 100% code coverage.
"""

import pytest
from utils.reflection_generator import (
    ReflectionGenerator,
    ReflectionContext,
    extract_reflection_context,
    clean_word,
    extract_words_from_text
)


class TestReflectionContext:
    """Tests for ReflectionContext dataclass."""
    
    def test_reflection_context_creation(self):
        """Test creating a ReflectionContext."""
        context = ReflectionContext(
            dominant_emotion='happy',
            emotion_confidence=0.8,
            avg_s_score=0.5,
            maturity_index=0.3,
            day_counter=5
        )
        assert context.dominant_emotion == 'happy'
        assert context.emotion_confidence == 0.8
        assert context.avg_s_score == 0.5
        assert context.maturity_index == 0.3
        assert context.day_counter == 5


class TestReflectionGenerator:
    """Tests for ReflectionGenerator class."""
    
    def test_init(self):
        """Test ReflectionGenerator initialization."""
        generator = ReflectionGenerator()
        assert generator is not None
    
    def test_generate_reflection_happy(self):
        """Test generating reflection with happy emotion."""
        generator = ReflectionGenerator()
        context = ReflectionContext(
            dominant_emotion='happy',
            emotion_confidence=0.8,
            avg_s_score=0.6,
            maturity_index=0.4,
            day_counter=3
        )
        reflection = generator.generate_reflection(context)
        assert 'I felt happy today' in reflection
        assert 'very much' in reflection or 'somewhat' in reflection or 'a little' in reflection
        assert 'This is day 3' in reflection
    
    def test_generate_reflection_sad(self):
        """Test generating reflection with sad emotion."""
        generator = ReflectionGenerator()
        context = ReflectionContext(
            dominant_emotion='sad',
            emotion_confidence=0.7,
            avg_s_score=0.3,
            maturity_index=0.5,
            day_counter=1
        )
        reflection = generator.generate_reflection(context)
        assert 'I felt sad today' in reflection
        assert 'This is day 1' in reflection
    
    def test_generate_reflection_angry(self):
        """Test generating reflection with angry emotion."""
        generator = ReflectionGenerator()
        context = ReflectionContext(
            dominant_emotion='angry',
            emotion_confidence=0.9,
            avg_s_score=0.7,
            maturity_index=0.2,
            day_counter=10
        )
        reflection = generator.generate_reflection(context)
        assert 'I felt angry today' in reflection
        assert 'This is day 10' in reflection
    
    def test_generate_reflection_unknown_emotion(self):
        """Test generating reflection with unknown emotion."""
        generator = ReflectionGenerator()
        context = ReflectionContext(
            dominant_emotion='unknown_emotion',
            emotion_confidence=0.5,
            avg_s_score=0.4,
            maturity_index=0.3,
            day_counter=2
        )
        reflection = generator.generate_reflection(context)
        assert 'I experienced something today' in reflection
    
    def test_get_emotion_phrase_all_emotions(self):
        """Test getting emotion phrases for all known emotions."""
        generator = ReflectionGenerator()
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']
        for emotion in emotions:
            phrase = generator._get_emotion_phrase(emotion)
            assert isinstance(phrase, str)
            assert len(phrase) > 0
    
    def test_get_emotion_phrase_unknown(self):
        """Test getting emotion phrase for unknown emotion."""
        generator = ReflectionGenerator()
        phrase = generator._get_emotion_phrase('unknown')
        assert phrase == 'I experienced something today'
    
    def test_calculate_intensity_high(self):
        """Test intensity calculation for high s_score."""
        generator = ReflectionGenerator()
        intensity = generator._calculate_intensity(0.6)
        assert intensity == 'very much'
    
    def test_calculate_intensity_medium(self):
        """Test intensity calculation for medium s_score."""
        generator = ReflectionGenerator()
        intensity = generator._calculate_intensity(0.3)
        assert intensity == 'somewhat'
    
    def test_calculate_intensity_low(self):
        """Test intensity calculation for low s_score."""
        generator = ReflectionGenerator()
        intensity = generator._calculate_intensity(0.1)
        assert intensity == 'a little'
    
    def test_calculate_intensity_boundary_high(self):
        """Test intensity calculation at high threshold boundary."""
        generator = ReflectionGenerator()
        # At exactly 0.5, it's > 0.5 is False, so it goes to medium check
        intensity = generator._calculate_intensity(0.5)
        assert intensity == 'somewhat'  # 0.5 is not > 0.5, so it checks > 0.2
        # But 0.51 should be 'very much'
        intensity2 = generator._calculate_intensity(0.51)
        assert intensity2 == 'very much'
    
    def test_calculate_intensity_boundary_medium(self):
        """Test intensity calculation at medium threshold boundary."""
        generator = ReflectionGenerator()
        # At exactly 0.2, it's > 0.2 is False, so it goes to 'a little'
        intensity = generator._calculate_intensity(0.2)
        assert intensity == 'a little'  # 0.2 is not > 0.2
        # But 0.21 should be 'somewhat'
        intensity2 = generator._calculate_intensity(0.21)
        assert intensity2 == 'somewhat'
    
    def test_calculate_wisdom_high(self):
        """Test wisdom calculation for high maturity."""
        generator = ReflectionGenerator()
        wisdom = generator._calculate_wisdom(0.6)
        assert wisdom == 'I am learning'
    
    def test_calculate_wisdom_medium(self):
        """Test wisdom calculation for medium maturity."""
        generator = ReflectionGenerator()
        wisdom = generator._calculate_wisdom(0.3)
        assert wisdom == 'I am growing'
    
    def test_calculate_wisdom_low(self):
        """Test wisdom calculation for low maturity."""
        generator = ReflectionGenerator()
        wisdom = generator._calculate_wisdom(0.1)
        assert wisdom == 'I am new'
    
    def test_calculate_wisdom_boundary_high(self):
        """Test wisdom calculation at high threshold boundary."""
        generator = ReflectionGenerator()
        # At exactly 0.5, it's > 0.5 is False, so it goes to medium check
        wisdom = generator._calculate_wisdom(0.5)
        assert wisdom == 'I am growing'  # 0.5 is not > 0.5, so it checks > 0.2
        # But 0.51 should be 'I am learning'
        wisdom2 = generator._calculate_wisdom(0.51)
        assert wisdom2 == 'I am learning'
    
    def test_calculate_wisdom_boundary_medium(self):
        """Test wisdom calculation at medium threshold boundary."""
        generator = ReflectionGenerator()
        # At exactly 0.2, it's > 0.2 is False, so it goes to 'I am new'
        wisdom = generator._calculate_wisdom(0.2)
        assert wisdom == 'I am new'  # 0.2 is not > 0.2
        # But 0.21 should be 'I am growing'
        wisdom2 = generator._calculate_wisdom(0.21)
        assert wisdom2 == 'I am growing'


class TestExtractReflectionContext:
    """Tests for extract_reflection_context function."""
    
    def test_extract_context_basic(self):
        """Test extracting context with basic data."""
        emotion_stats = {
            'most_common': {
                'emotion': 'happy',
                'confidence': 0.8
            }
        }
        char_state = {
            'maturity_index': 0.4
        }
        memory_levels = [
            [{'s_score': 0.5}, {'s_score': 0.6}],
            [{'s_score': 0.4}],
            []
        ]
        context = extract_reflection_context(
            emotion_stats, char_state, memory_levels, day_counter=5
        )
        assert context.dominant_emotion == 'happy'
        assert context.emotion_confidence == 0.8
        assert context.maturity_index == 0.4
        assert context.day_counter == 5
        assert context.avg_s_score > 0
    
    def test_extract_context_no_emotion_stats(self):
        """Test extracting context with missing emotion stats."""
        emotion_stats = {}
        char_state = {'maturity_index': 0.3}
        memory_levels = [[{'s_score': 0.5}]]
        context = extract_reflection_context(
            emotion_stats, char_state, memory_levels, day_counter=1
        )
        assert context.dominant_emotion == 'neutral'
        assert context.emotion_confidence == 0.0
    
    def test_extract_context_no_memories(self):
        """Test extracting context with no memories."""
        emotion_stats = {
            'most_common': {
                'emotion': 'sad',
                'confidence': 0.6
            }
        }
        char_state = {'maturity_index': 0.2}
        memory_levels = [[], [], []]
        context = extract_reflection_context(
            emotion_stats, char_state, memory_levels, day_counter=2
        )
        assert context.avg_s_score == 0.0
    
    def test_extract_context_memories_without_s_score(self):
        """Test extracting context with memories missing s_score."""
        emotion_stats = {
            'most_common': {
                'emotion': 'neutral',
                'confidence': 0.5
            }
        }
        char_state = {'maturity_index': 0.1}
        memory_levels = [
            [{'data': 'test'}, {'s_score': 0.3}],
            [{'other': 'value'}]
        ]
        context = extract_reflection_context(
            emotion_stats, char_state, memory_levels, day_counter=3
        )
        assert context.avg_s_score == 0.3  # Only one memory has s_score
    
    def test_extract_context_max_levels_limit(self):
        """Test extracting context with max_levels limit."""
        emotion_stats = {
            'most_common': {
                'emotion': 'happy',
                'confidence': 0.7
            }
        }
        char_state = {'maturity_index': 0.5}
        memory_levels = [
            [{'s_score': 0.5}],
            [{'s_score': 0.6}],
            [{'s_score': 0.7}],
            [{'s_score': 0.8}]  # Should be ignored with max_levels=3
        ]
        context = extract_reflection_context(
            emotion_stats, char_state, memory_levels, day_counter=4, max_levels=3
        )
        # Should only include first 3 levels
        assert context.avg_s_score < 0.8  # Should not include level 3's 0.8
    
    def test_extract_context_max_memories_per_level(self):
        """Test extracting context with max_memories_per_level limit."""
        emotion_stats = {
            'most_common': {
                'emotion': 'surprise',
                'confidence': 0.9
            }
        }
        char_state = {'maturity_index': 0.6}
        # Create 100 memories, but max_memories_per_level=50
        memory_levels = [
            [{'s_score': 0.1 + i * 0.01} for i in range(100)]
        ]
        context = extract_reflection_context(
            emotion_stats, char_state, memory_levels, day_counter=5,
            max_memories_per_level=50
        )
        # Should only use last 50 memories
        assert context.avg_s_score > 0.5  # Average of last 50 (0.5-0.99 range)


class TestCleanWord:
    """Tests for clean_word function."""
    
    def test_clean_word_basic(self):
        """Test cleaning a basic word."""
        result = clean_word('Hello')
        assert result == 'hello'
    
    def test_clean_word_with_punctuation(self):
        """Test cleaning word with punctuation."""
        result = clean_word('Hello!')
        assert result == 'hello'
    
    def test_clean_word_with_numbers(self):
        """Test cleaning word with numbers."""
        result = clean_word('Hello123')
        assert result == 'hello123'
    
    def test_clean_word_only_punctuation(self):
        """Test cleaning word with only punctuation."""
        result = clean_word('!!!')
        assert result == ''
    
    def test_clean_word_mixed(self):
        """Test cleaning word with mixed characters."""
        result = clean_word('Hello, World!')
        assert result == 'helloworld'
    
    def test_clean_word_empty(self):
        """Test cleaning empty word."""
        result = clean_word('')
        assert result == ''
    
    def test_clean_word_whitespace(self):
        """Test cleaning word with whitespace."""
        result = clean_word('  hello  ')
        assert result == 'hello'


class TestExtractWordsFromText:
    """Tests for extract_words_from_text function."""
    
    def test_extract_words_basic(self):
        """Test extracting words from basic text."""
        result = extract_words_from_text('Hello world')
        assert result == ['hello', 'world']
    
    def test_extract_words_with_punctuation(self):
        """Test extracting words with punctuation."""
        result = extract_words_from_text('Hello, world!')
        assert result == ['hello', 'world']
    
    def test_extract_words_multiple_spaces(self):
        """Test extracting words with multiple spaces."""
        result = extract_words_from_text('Hello   world   test')
        assert result == ['hello', 'world', 'test']
    
    def test_extract_words_empty_string(self):
        """Test extracting words from empty string."""
        result = extract_words_from_text('')
        assert result == []
    
    def test_extract_words_only_punctuation(self):
        """Test extracting words from text with only punctuation."""
        result = extract_words_from_text('!!! ... ---')
        assert result == []
    
    def test_extract_words_mixed_case(self):
        """Test extracting words with mixed case."""
        result = extract_words_from_text('Hello WORLD Test')
        assert result == ['hello', 'world', 'test']
    
    def test_extract_words_with_numbers(self):
        """Test extracting words with numbers."""
        result = extract_words_from_text('Hello 123 world')
        assert result == ['hello', '123', 'world']
    
    def test_extract_words_complex(self):
        """Test extracting words from complex text."""
        text = 'Hello, world! This is a test. 123 numbers.'
        result = extract_words_from_text(text)
        assert 'hello' in result
        assert 'world' in result
        assert 'this' in result
        assert 'is' in result
        assert 'a' in result
        assert 'test' in result
        assert '123' in result
        assert 'numbers' in result

