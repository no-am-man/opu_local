"""
Tests for utils/youtube_processor.py - YouTube OPU Processor.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from utils.youtube_processor import YouTubeOPUProcessor
from config import CHUNK_SIZE, VISUAL_SURPRISE_THRESHOLD


class TestYouTubeOPUProcessor:
    """Tests for YouTubeOPUProcessor class."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for processor."""
        youtube_streamer = Mock()
        youtube_streamer.get_audio_chunk.return_value = np.zeros(CHUNK_SIZE, dtype=np.float32)
        youtube_streamer.get_video_frame.return_value = np.zeros((360, 640, 3), dtype=np.uint8)
        youtube_streamer.video_info = {'title': 'Test Video'}
        
        cortex = Mock()
        cortex.introspect.return_value = 0.5
        cortex.introspect_visual.return_value = (0.6, {'R': 0.3, 'G': 0.4, 'B': 0.5})
        cortex.get_character_state.return_value = {'base_pitch': 220.0, 'maturity_index': 0.5, 'maturity_level': 0}
        cortex.get_current_state.return_value = {'coherence': 0.7}
        cortex.store_memory = Mock()
        
        genesis = Mock()
        genesis.ethical_veto.return_value = np.array([0.5])
        
        afl = Mock()
        afl.update_pitch = Mock()
        afl.play_tone = Mock()
        afl.current_frequency = 220.0
        
        phoneme_analyzer = Mock()
        phoneme_analyzer.analyze.return_value = None
        
        visualizer = Mock()
        visualizer.update_state = Mock()
        visualizer.draw_cognitive_map = Mock()
        visualizer.render_to_image.return_value = np.zeros((400, 400, 3), dtype=np.uint8)
        
        visual_perception = Mock()
        visual_perception.analyze_frame.return_value = np.array([0.1, 0.2, 0.3])
        
        object_detector = Mock()
        object_detector.detect_objects.return_value = []
        object_detector.draw_detections.return_value = np.zeros((360, 640, 3), dtype=np.uint8)
        
        return {
            'youtube_streamer': youtube_streamer,
            'cortex': cortex,
            'genesis': genesis,
            'afl': afl,
            'phoneme_analyzer': phoneme_analyzer,
            'visualizer': visualizer,
            'visual_perception': visual_perception,
            'object_detector': object_detector
        }
    
    @pytest.fixture
    def processor(self, mock_components):
        """Create processor instance with mocked components."""
        return YouTubeOPUProcessor(**mock_components)
    
    def test_init(self, mock_components):
        """Test processor initialization."""
        processor = YouTubeOPUProcessor(**mock_components)
        
        assert processor.yt == mock_components['youtube_streamer']
        assert processor.cortex == mock_components['cortex']
        assert processor.safe_score == 0.0
        assert processor.frame_count == 0
        assert processor.start_time > 0
    
    def test_process_audio_channel_valid_chunk(self, processor, mock_components):
        """Test processing audio channel with valid chunk."""
        audio_chunk = np.random.randn(CHUNK_SIZE).astype(np.float32)
        mock_components['youtube_streamer'].get_audio_chunk.return_value = audio_chunk
        
        s_audio, audio_perception = processor.process_audio_channel()
        
        assert s_audio == 0.5  # From mock
        assert 'genomic_bit' in audio_perception
    
    def test_process_audio_channel_invalid_chunk(self, processor, mock_components):
        """Test processing audio channel with invalid chunk size."""
        mock_components['youtube_streamer'].get_audio_chunk.return_value = np.zeros(100, dtype=np.float32)
        
        s_audio, audio_perception = processor.process_audio_channel()
        
        assert s_audio == 0.0
        assert audio_perception['genomic_bit'] == 0.0
    
    def test_process_video_channel_with_frame(self, processor, mock_components):
        """Test processing video channel with valid frame."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        mock_components['youtube_streamer'].get_video_frame.return_value = frame
        
        s_audio = 0.5
        audio_perception = {'genomic_bit': 0.6}
        
        result = processor.process_video_channel(s_audio, audio_perception)
        
        assert 's_visual' in result
        assert 'processed_frame' in result
        assert 'detections' in result
        assert 'emotion' in result
    
    def test_process_video_channel_no_frame(self, processor, mock_components):
        """Test processing video channel when no frame available."""
        mock_components['youtube_streamer'].get_video_frame.return_value = None
        
        result = processor.process_video_channel(0.5, {'genomic_bit': 0.6})
        
        assert result['s_visual'] == 0.0
        assert result['processed_frame'] is None
        assert result['detections'] == []
    
    def test_process_fusion_and_veto(self, processor, mock_components):
        """Test fusion and ethical veto calculation."""
        safe_score = processor.process_fusion_and_veto(0.5, 0.6, {'genomic_bit': 0.7})
        
        assert safe_score == 0.5  # From mock
        mock_components['genesis'].ethical_veto.assert_called_once()
    
    def test_store_memories_audio(self, processor, mock_components):
        """Test storing audio memory."""
        audio_perception = {'genomic_bit': 0.5}
        safe_score = 0.6
        video_result = {'s_visual': 0.1, 'visual_vector_annotated': np.array([0.1, 0.2, 0.3]), 'emotion': None}
        cycle_timestamp = time.time()
        
        processor.store_memories(audio_perception, safe_score, video_result, cycle_timestamp)
        
        # Should store audio memory
        assert mock_components['cortex'].store_memory.call_count >= 1
    
    def test_store_memories_visual_above_threshold(self, processor, mock_components):
        """Test storing visual memory when above threshold."""
        audio_perception = {'genomic_bit': 0.5}
        safe_score = 0.6
        video_result = {
            's_visual': VISUAL_SURPRISE_THRESHOLD + 0.1,
            'visual_vector_annotated': np.array([0.1, 0.2, 0.3]),
            'emotion': None
        }
        cycle_timestamp = time.time()
        
        processor.store_memories(audio_perception, safe_score, video_result, cycle_timestamp)
        
        # Should store both audio and visual memories
        assert mock_components['cortex'].store_memory.call_count >= 2
    
    def test_store_memories_visual_below_threshold(self, processor, mock_components):
        """Test not storing visual memory when below threshold."""
        audio_perception = {'genomic_bit': 0.5}
        safe_score = 0.6
        video_result = {
            's_visual': VISUAL_SURPRISE_THRESHOLD - 0.1,
            'visual_vector_annotated': np.array([0.1, 0.2, 0.3]),
            'emotion': None
        }
        cycle_timestamp = time.time()
        
        processor.store_memories(audio_perception, safe_score, video_result, cycle_timestamp)
        
        # Should only store audio memory
        assert mock_components['cortex'].store_memory.call_count == 1
    
    def test_update_expression(self, processor, mock_components):
        """Test updating expression."""
        processor.update_expression(0.5)
        
        mock_components['afl'].update_pitch.assert_called_once()
        mock_components['afl'].play_tone.assert_called_once()
        mock_components['phoneme_analyzer'].analyze.assert_called_once()
    
    def test_update_visualization(self, processor, mock_components):
        """Test updating visualization."""
        result = processor.update_visualization(0.5)
        
        mock_components['visualizer'].update_state.assert_called_once()
        mock_components['visualizer'].draw_cognitive_map.assert_called_once()
        mock_components['visualizer'].render_to_image.assert_called_once()
        assert result is not None
    
    def test_process_cycle_complete(self, processor, mock_components):
        """Test complete processing cycle."""
        result = processor.process_cycle()
        
        assert 's_audio' in result
        assert 's_visual' in result
        assert 'safe_score' in result
        assert 'frame_count' in result
        assert result['frame_count'] == 1  # Incremented
    
    def test_process_cycle_increments_frame_count(self, processor, mock_components):
        """Test that process_cycle increments frame count."""
        initial_count = processor.frame_count
        
        processor.process_cycle()
        
        assert processor.frame_count == initial_count + 1

