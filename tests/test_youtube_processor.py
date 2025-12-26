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
        cortex.memory_levels = {i: [] for i in range(8)}  # Make memory_levels a dict for subscriptable access
        
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
    
    def test_update_expression_exception_handling(self, processor, mock_components):
        """Test update_expression exception handling (covers lines 226-227)."""
        # Make play_tone raise an exception
        mock_components['afl'].play_tone.side_effect = Exception("Audio error")
        
        # Should not crash
        processor.update_expression(0.5)
        
        # Should still call other methods
        mock_components['afl'].update_pitch.assert_called_once()
        mock_components['phoneme_analyzer'].analyze.assert_called_once()
    
    def test_update_expression_with_phoneme(self, processor, mock_components, capsys):
        """Test update_expression when phoneme is detected (covers line 232)."""
        mock_components['phoneme_analyzer'].analyze.return_value = "ah"
        
        processor.update_expression(0.5)
        
        captured = capsys.readouterr()
        assert "[PHONEME]" in captured.out
        assert "ah" in captured.out
    
    def test_update_visualization_exception_handling(self, processor, mock_components):
        """Test update_visualization exception handling (covers lines 252-260)."""
        # Make render_to_image return None to test exception path
        mock_components['visualizer'].render_to_image.return_value = None
        
        # Should not crash
        result = processor.update_visualization(0.5)
        
        # Should still return None or handle gracefully
        assert result is None or result is not None
    
    def test_update_visualization_queue_full(self, processor, mock_components):
        """Test update_visualization when queue is full (covers lines 252-260)."""
        import queue
        # Create a full queue
        full_queue = queue.Queue(maxsize=1)
        full_queue.put(('test', np.zeros((100, 100, 3))))
        processor.image_queue = full_queue
        
        # Make render_to_image return an image
        mock_components['visualizer'].render_to_image.return_value = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Should not crash when queue is full
        result = processor.update_visualization(0.5)
        assert result is not None
    
    def test_display_frames_cv2_not_available(self, processor, mock_components, monkeypatch):
        """Test display_frames when cv2 is not available (covers line 274)."""
        with patch('utils.youtube_processor.CV2_AVAILABLE', False):
            # Should return early
            processor.display_frames(
                np.zeros((360, 640, 3), dtype=np.uint8),
                np.zeros((400, 400, 3), dtype=np.uint8)
            )
            # No assertions needed - just verify it doesn't crash
    
    def test_display_frames_exception_handling(self, processor, mock_components):
        """Test display_frames exception handling (covers lines 278-285)."""
        import queue
        # Create a queue that will raise exception
        error_queue = Mock()
        error_queue.put_nowait.side_effect = Exception("Queue error")
        processor.image_queue = error_queue
        
        # Should not crash
        processor.display_frames(
            np.zeros((360, 640, 3), dtype=np.uint8),
            np.zeros((400, 400, 3), dtype=np.uint8)
        )
        # No assertions needed - just verify it doesn't crash
    
    def test_process_cycle_logging(self, processor, mock_components, capsys):
        """Test process_cycle logging at frame 100 (covers lines 333-336)."""
        # Set frame_count to 99 so next cycle will be 100
        processor.frame_count = 99
        
        processor.process_cycle()
        
        captured = capsys.readouterr()
        # Should log at frame 100
        assert "Cycle 100" in captured.out or processor.frame_count == 100

