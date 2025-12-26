"""
Tests for utils/cycle_utils.py - Cycle processing utilities.
Targets 100% code coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional

from utils.cycle_utils import (
    fuse_scores,
    apply_ethical_veto,
    extract_visual_bit,
    extract_emotion_from_detections,
    format_emotion_string,
    LogCounter,
    create_processing_result,
    get_consolidation_ratio,
    can_consolidate_at_level,
    approximate_timestamp,
    get_time_window_for_level,
    group_memories_by_time_window
)


class TestFuseScores:
    """Tests for fuse_scores function."""
    
    def test_fuse_scores_audio_higher(self):
        """Test fusion when audio score is higher."""
        result = fuse_scores(0.8, 0.5)
        assert result == 0.8
    
    def test_fuse_scores_visual_higher(self):
        """Test fusion when visual score is higher."""
        result = fuse_scores(0.3, 0.9)
        assert result == 0.9
    
    def test_fuse_scores_equal(self):
        """Test fusion when scores are equal."""
        result = fuse_scores(0.5, 0.5)
        assert result == 0.5
    
    def test_fuse_scores_zero(self):
        """Test fusion with zero scores."""
        result = fuse_scores(0.0, 0.0)
        assert result == 0.0


class TestApplyEthicalVeto:
    """Tests for apply_ethical_veto function."""
    
    def test_apply_ethical_veto_with_action(self):
        """Test ethical veto with valid action."""
        genesis = Mock()
        genesis.ethical_veto = Mock(return_value=np.array([0.5, 0.3]))
        
        result = apply_ethical_veto(genesis, 0.8, 0.2)
        assert result == 0.5
        genesis.ethical_veto.assert_called_once()
    
    def test_apply_ethical_veto_empty_action(self):
        """Test ethical veto with empty action array."""
        genesis = Mock()
        genesis.ethical_veto = Mock(return_value=np.array([]))
        
        result = apply_ethical_veto(genesis, 0.8, 0.2)
        assert result == 0.8  # Should return original fused_score
    
    def test_apply_ethical_veto_single_element(self):
        """Test ethical veto with single element action."""
        genesis = Mock()
        genesis.ethical_veto = Mock(return_value=np.array([0.4]))
        
        result = apply_ethical_veto(genesis, 0.8, 0.2)
        assert result == 0.4


class TestExtractVisualBit:
    """Tests for extract_visual_bit function."""
    
    def test_extract_visual_bit_max_value(self):
        """Test extracting maximum value from visual vector."""
        vector = np.array([0.1, 0.5, 0.3])
        result = extract_visual_bit(vector)
        assert result == 0.5
    
    def test_extract_visual_bit_empty(self):
        """Test extracting from empty vector."""
        vector = np.array([])
        result = extract_visual_bit(vector)
        assert result == 0.0
    
    def test_extract_visual_bit_single_element(self):
        """Test extracting from single element vector."""
        vector = np.array([0.7])
        result = extract_visual_bit(vector)
        assert result == 0.7


class TestExtractEmotionFromDetections:
    """Tests for extract_emotion_from_detections function."""
    
    def test_extract_emotion_with_face(self):
        """Test extracting emotion from face detection."""
        detections = [
            {'label': 'face', 'emotion': {'emotion': 'happy', 'confidence': 0.8}},
            {'label': 'person', 'emotion': {'emotion': 'sad', 'confidence': 0.6}}
        ]
        result = extract_emotion_from_detections(detections)
        assert result == {'emotion': 'happy', 'confidence': 0.8}
    
    def test_extract_emotion_no_face(self):
        """Test extracting emotion when no face detected."""
        detections = [
            {'label': 'person', 'emotion': {'emotion': 'sad', 'confidence': 0.6}}
        ]
        result = extract_emotion_from_detections(detections)
        assert result is None
    
    def test_extract_emotion_empty_list(self):
        """Test extracting emotion from empty list."""
        result = extract_emotion_from_detections([])
        assert result is None
    
    def test_extract_emotion_none(self):
        """Test extracting emotion from None."""
        result = extract_emotion_from_detections(None)
        assert result is None
    
    def test_extract_emotion_no_emotion_key(self):
        """Test extracting emotion when emotion key is missing."""
        detections = [
            {'label': 'face'}
        ]
        result = extract_emotion_from_detections(detections)
        assert result is None


class TestFormatEmotionString:
    """Tests for format_emotion_string function."""
    
    def test_format_emotion_with_emotion(self):
        """Test formatting emotion dict."""
        emotion = {'emotion': 'happy', 'confidence': 0.85}
        result = format_emotion_string(emotion)
        assert result == " | Emotion: happy (0.85)"
    
    def test_format_emotion_none(self):
        """Test formatting None emotion."""
        result = format_emotion_string(None)
        assert result == ""
    
    def test_format_emotion_empty_dict(self):
        """Test formatting empty emotion dict."""
        emotion = {}
        result = format_emotion_string(emotion)
        # Should handle missing keys gracefully
        assert isinstance(result, str)


class TestLogCounter:
    """Tests for LogCounter class."""
    
    def test_init_default_interval(self):
        """Test LogCounter initialization with default interval."""
        counter = LogCounter()
        assert counter.interval == 100
        assert counter.count == 0
    
    def test_init_custom_interval(self):
        """Test LogCounter initialization with custom interval."""
        counter = LogCounter(interval=50)
        assert counter.interval == 50
        assert counter.count == 0
    
    def test_should_log_first_time(self):
        """Test should_log on first call."""
        counter = LogCounter(interval=10)
        assert counter.should_log() is False
        assert counter.count == 1
    
    def test_should_log_at_interval(self):
        """Test should_log when interval is reached."""
        counter = LogCounter(interval=5)
        for i in range(4):
            assert counter.should_log() is False
        assert counter.should_log() is True  # 5th call
        assert counter.count == 5
    
    def test_should_log_multiple_intervals(self):
        """Test should_log across multiple intervals."""
        counter = LogCounter(interval=3)
        assert counter.should_log() is False  # 1
        assert counter.should_log() is False  # 2
        assert counter.should_log() is True   # 3
        assert counter.should_log() is False  # 4
        assert counter.should_log() is False  # 5
        assert counter.should_log() is True   # 6
    
    def test_reset(self):
        """Test reset method."""
        counter = LogCounter(interval=10)
        counter.should_log()  # Increment to 1
        counter.should_log()  # Increment to 2
        counter.reset()
        assert counter.count == 0


class TestCreateProcessingResult:
    """Tests for create_processing_result function."""
    
    def test_create_processing_result_basic(self):
        """Test creating basic processing result."""
        result = create_processing_result(0.5, 0.7, 0.6)
        
        assert result['s_audio'] == 0.5
        assert result['s_visual'] == 0.7
        assert result['safe_score'] == 0.6
        assert result['fused_score'] == 0.7  # max(0.5, 0.7)
    
    def test_create_processing_result_with_channels(self):
        """Test creating processing result with channel scores."""
        channel_scores = {'R': 0.3, 'G': 0.5, 'B': 0.2}
        result = create_processing_result(0.4, 0.6, 0.5, channel_scores=channel_scores)
        
        assert result['s_audio'] == 0.4
        assert result['s_visual'] == 0.6
        assert result['safe_score'] == 0.5
        assert result['channel_scores'] == channel_scores
    
    def test_create_processing_result_no_channels(self):
        """Test creating processing result without channel scores."""
        result = create_processing_result(0.3, 0.4, 0.35)
        
        assert 'channel_scores' not in result


class TestGetConsolidationRatio:
    """Tests for get_consolidation_ratio function."""
    
    def test_get_consolidation_ratio_level_0(self):
        """Test getting consolidation ratio for level 0."""
        ratio = get_consolidation_ratio(0)
        assert ratio == 20
    
    def test_get_consolidation_ratio_level_1(self):
        """Test getting consolidation ratio for level 1."""
        ratio = get_consolidation_ratio(1)
        assert ratio == 60
    
    def test_get_consolidation_ratio_level_2(self):
        """Test getting consolidation ratio for level 2."""
        ratio = get_consolidation_ratio(2)
        assert ratio == 60
    
    def test_get_consolidation_ratio_level_7(self):
        """Test getting consolidation ratio for level 7."""
        ratio = get_consolidation_ratio(7)
        assert ratio == 10
    
    def test_get_consolidation_ratio_invalid_level(self):
        """Test getting consolidation ratio for invalid level."""
        ratio = get_consolidation_ratio(99)
        # Should return default
        from config import BRAIN_DEFAULT_CONSOLIDATION_RATIO
        assert ratio == BRAIN_DEFAULT_CONSOLIDATION_RATIO


class TestCanConsolidateAtLevel:
    """Tests for can_consolidate_at_level function."""
    
    def test_can_consolidate_at_level_sufficient(self):
        """Test consolidation check with sufficient items."""
        assert can_consolidate_at_level(0, 20) is True
        assert can_consolidate_at_level(0, 21) is True
        assert can_consolidate_at_level(1, 60) is True
        assert can_consolidate_at_level(1, 61) is True
    
    def test_can_consolidate_at_level_insufficient(self):
        """Test consolidation check with insufficient items."""
        assert can_consolidate_at_level(0, 19) is False
        assert can_consolidate_at_level(1, 59) is False
        assert can_consolidate_at_level(2, 59) is False
    
    def test_can_consolidate_at_level_exact(self):
        """Test consolidation check with exact count."""
        assert can_consolidate_at_level(0, 20) is True
        assert can_consolidate_at_level(1, 60) is True


class TestApproximateTimestamp:
    """Tests for approximate_timestamp function."""
    
    def test_approximate_timestamp_1s_window(self):
        """Test timestamp approximation with 1 second window."""
        timestamp = 1000.5
        result = approximate_timestamp(timestamp, 1.0)
        assert result == 1000.0
    
    def test_approximate_timestamp_60s_window(self):
        """Test timestamp approximation with 60 second window."""
        timestamp = 3661.5
        result = approximate_timestamp(timestamp, 60.0)
        assert result == 3660.0
    
    def test_approximate_timestamp_exact_boundary(self):
        """Test timestamp approximation at exact boundary."""
        timestamp = 120.0
        result = approximate_timestamp(timestamp, 60.0)
        assert result == 120.0
    
    def test_approximate_timestamp_zero(self):
        """Test timestamp approximation with zero timestamp."""
        result = approximate_timestamp(0.0, 60.0)
        assert result == 0.0


class TestGetTimeWindowForLevel:
    """Tests for get_time_window_for_level function."""
    
    def test_get_time_window_level_0(self):
        """Test getting time window for level 0."""
        window = get_time_window_for_level(0)
        assert window == 1.0
    
    def test_get_time_window_level_1(self):
        """Test getting time window for level 1."""
        window = get_time_window_for_level(1)
        assert window == 60.0
    
    def test_get_time_window_level_2(self):
        """Test getting time window for level 2."""
        window = get_time_window_for_level(2)
        assert window == 3600.0
    
    def test_get_time_window_level_7(self):
        """Test getting time window for level 7."""
        window = get_time_window_for_level(7)
        assert window == 126230400.0
    
    def test_get_time_window_invalid_level(self):
        """Test getting time window for invalid level."""
        window = get_time_window_for_level(99)
        assert window == 1.0  # Default


class TestGroupMemoriesByTimeWindow:
    """Tests for group_memories_by_time_window function."""
    
    def test_group_memories_empty_list(self):
        """Test grouping empty memory list."""
        result = group_memories_by_time_window([], 10, 60.0)
        assert result == []
    
    def test_group_memories_same_time_window(self):
        """Test grouping memories in same time window."""
        memories = [
            {'timestamp': 1000.0, 'data': 'a'},
            {'timestamp': 1000.5, 'data': 'b'},
            {'timestamp': 1000.8, 'data': 'c'},
        ]
        result = group_memories_by_time_window(memories, 3, 1.0)
        assert len(result) == 3
        assert all(m in result for m in memories)
    
    def test_group_memories_different_time_windows(self):
        """Test grouping memories across different time windows."""
        memories = [
            {'timestamp': 1000.0, 'data': 'a'},
            {'timestamp': 1000.5, 'data': 'b'},
            {'timestamp': 1061.0, 'data': 'c'},  # Different 60s window
        ]
        result = group_memories_by_time_window(memories, 3, 60.0)
        # Should group first two (same window), then third
        assert len(result) >= 2
    
    def test_group_memories_target_size(self):
        """Test grouping with target size limit."""
        memories = [
            {'timestamp': 1000.0 + i * 0.1, 'data': f'item_{i}'}
            for i in range(20)
        ]
        result = group_memories_by_time_window(memories, 10, 1.0)
        assert len(result) == 10
    
    def test_group_memories_sorted(self):
        """Test that memories are sorted by timestamp."""
        memories = [
            {'timestamp': 1002.0, 'data': 'c'},
            {'timestamp': 1000.0, 'data': 'a'},
            {'timestamp': 1001.0, 'data': 'b'},
        ]
        result = group_memories_by_time_window(memories, 3, 1.0)
        # Should be sorted oldest first
        assert result[0]['data'] == 'a'
        assert result[1]['data'] == 'b'
        assert result[2]['data'] == 'c'
    
    def test_group_memories_missing_timestamp(self):
        """Test grouping memories with missing timestamp."""
        memories = [
            {'timestamp': 1000.0, 'data': 'a'},
            {'data': 'b'},  # Missing timestamp
            {'timestamp': 1001.0, 'data': 'c'},
        ]
        result = group_memories_by_time_window(memories, 3, 1.0)
        # Should handle missing timestamps (defaults to 0)
        assert len(result) >= 2
    
    def test_group_memories_exact_target_size(self):
        """Test grouping with exact target size."""
        memories = [
            {'timestamp': 1000.0 + i * 0.1, 'data': f'item_{i}'}
            for i in range(15)
        ]
        result = group_memories_by_time_window(memories, 15, 1.0)
        assert len(result) == 15
    
    def test_group_memories_larger_than_target(self):
        """Test grouping when more memories than target."""
        memories = [
            {'timestamp': 1000.0 + i * 0.1, 'data': f'item_{i}'}
            for i in range(25)
        ]
        result = group_memories_by_time_window(memories, 10, 1.0)
        assert len(result) == 10
    
    def test_group_memories_different_window_break(self):
        """Test grouping stops when different time window is encountered."""
        memories = [
            {'timestamp': 1000.0, 'data': 'a'},
            {'timestamp': 1000.5, 'data': 'b'},
            {'timestamp': 1061.0, 'data': 'c'},  # Different 60s window
        ]
        result = group_memories_by_time_window(memories, 2, 60.0)
        # Should get first two (same window), then check third but may stop
        assert len(result) >= 2
    
    def test_group_memories_sorted_empty_after_sort(self):
        """Test edge case where sorted_memories becomes empty."""
        # This tests the early return after sorting
        memories = []
        result = group_memories_by_time_window(memories, 10, 60.0)
        assert result == []
    
    def test_group_memories_sorted_empty_check(self):
        """Test the sorted_memories empty check (line 253)."""
        # This should trigger the empty check after sorting
        # But since we already check if memories is empty, 
        # we need to test the case where sorting returns empty
        # Actually, if memories is empty, we return early, so this path
        # might not be reachable. Let's test with None timestamps that sort weirdly
        memories = [{'data': 'a'}]  # Missing timestamp, defaults to 0
        result = group_memories_by_time_window(memories, 1, 60.0)
        # Should still work with default timestamp
        assert len(result) == 1
    
    def test_group_memories_chunk_less_than_target(self):
        """Test when chunk is less than target size."""
        memories = [
            {'timestamp': 1000.0, 'data': 'a'},
            {'timestamp': 1000.5, 'data': 'b'},
        ]
        result = group_memories_by_time_window(memories, 5, 1.0)
        # Should return all available (2) even though target is 5
        assert len(result) == 2
    
    def test_group_memories_chunk_exceeds_target(self):
        """Test when chunk exceeds target and needs truncation."""
        memories = [
            {'timestamp': 1000.0 + i * 0.1, 'data': f'item_{i}'}
            for i in range(15)
        ]
        result = group_memories_by_time_window(memories, 10, 1.0)
        # Should truncate to target size
        assert len(result) == 10
    
    def test_group_memories_else_break_path(self):
        """Test the else break path when chunk >= target_size."""
        # Create memories where we'll hit the break condition
        memories = [
            {'timestamp': 1000.0 + i, 'data': f'item_{i}'}
            for i in range(15)
        ]
        # Use large time window so first few are in same window, then different
        result = group_memories_by_time_window(memories, 5, 60.0)
        # Should get at least 5 items
        assert len(result) >= 5
    
    def test_group_memories_second_break_condition(self):
        """Test the second break condition in the else clause."""
        # Create scenario where mem_time_window != base_time_window
        # and chunk already has enough items
        base_time = 1000.0
        memories = [
            {'timestamp': base_time + i * 0.5, 'data': f'item_{i}'}
            for i in range(12)
        ]
        # Small time window so we get different windows quickly
        result = group_memories_by_time_window(memories, 5, 1.0)
        # Should get exactly target_size or more from first window
        assert len(result) >= 5

