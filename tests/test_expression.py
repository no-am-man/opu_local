"""
Tests for core/expression.py - The Voice (AFL & Phoneme Analysis)
100% code coverage target
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from core.expression import AestheticFeedbackLoop, PhonemeAnalyzer
from config import SAMPLE_RATE, BASE_FREQUENCY


class TestAestheticFeedbackLoop:
    """Test suite for AestheticFeedbackLoop class."""
    
    @patch('core.expression.sd.OutputStream')
    def test_init(self, mock_output_stream):
        """Test AestheticFeedbackLoop initialization."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop(base_pitch=440.0)
        assert afl.base_pitch == 440.0
        assert afl.sample_rate == SAMPLE_RATE
        assert afl.current_frequency == 440.0
        assert afl.target_frequency == 440.0
        assert afl.phase == 0.0
        assert afl.breath_rate == 0.2
        assert afl.running is True
        mock_stream.start.assert_called_once()
    
    @patch('core.expression.sd.OutputStream')
    def test_play_tone_noise_gate(self, mock_output_stream):
        """Test play_tone with s_score below noise gate threshold."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        afl.play_tone(0.1, duration=None)  # Below 0.2 threshold
        assert afl.target_amp == 0.0
        assert afl.is_speaking is False
    
    @patch('core.expression.sd.OutputStream')
    def test_play_tone_attention_mapping(self, mock_output_stream):
        """Test play_tone volume mapping to attention."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        afl.play_tone(3.0, duration=None)
        # Volume should map to s_score / 3.0, capped at 1.0
        assert afl.target_amp == 1.0  # min(1.0, 3.0 / 3.0)
    
    @patch('core.expression.sd.OutputStream')
    def test_play_tone_pitch_mapping(self, mock_output_stream):
        """Test play_tone pitch mapping to surprise."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop(base_pitch=440.0)
        afl.play_tone(5.0, duration=None)
        # Pitch should be base_pitch * (1.0 + s_score / 10.0)
        expected_freq = 440.0 * (1.0 + 5.0 / 10.0)
        assert afl.target_frequency == expected_freq
    
    @patch('core.expression.sd.OutputStream')
    def test_play_tone_speaking_threshold(self, mock_output_stream):
        """Test play_tone speaking threshold."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        # Below threshold
        afl.play_tone(1.0, duration=None)
        assert afl.is_speaking is False
        
        # Above threshold
        afl.play_tone(2.0, duration=None)
        assert afl.is_speaking is True
    
    @patch('core.expression.sd.OutputStream')
    def test_play_tone_pitch_clipping(self, mock_output_stream):
        """Test that pitch is clipped to valid range."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop(base_pitch=440.0)
        # Very high s_score should clip pitch
        afl.play_tone(100.0, duration=None)
        assert 50.0 <= afl.target_frequency <= 2000.0
    
    @patch('core.expression.sd.OutputStream')
    def test_callback_basic(self, mock_output_stream):
        """Test audio callback generates valid output."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        outdata = np.zeros((1024, 1), dtype=np.float32)
        time_info = {}
        status = None
        
        # Should not raise exception
        afl.callback(outdata, 1024, time_info, status)
        assert outdata.shape == (1024, 1)
        assert outdata.dtype == np.float32
    
    @patch('core.expression.sd.OutputStream')
    def test_callback_with_status(self, mock_output_stream, capsys):
        """Test callback with status message (covers line 48)."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        outdata = np.zeros((1024, 1), dtype=np.float32)
        time_info = {}
        status = {'status': 'test'}
        
        afl.callback(outdata, 1024, time_info, status)
        captured = capsys.readouterr()
        assert "[AFL] Audio output status" in captured.out
    
    @patch('core.expression.sd.OutputStream')
    def test_callback_syllable_articulation(self, mock_output_stream):
        """Test callback with syllable articulation when speaking (covers lines 87-92)."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        afl.is_speaking = True  # Enable speaking mode
        afl.target_amp = 0.5
        
        outdata = np.zeros((1024, 1), dtype=np.float32)
        afl.callback(outdata, 1024, {}, None)
        
        # Should have updated syllable_phase
        assert afl.syllable_phase >= 0
        assert afl.syllable_phase < 2 * np.pi
    
    @patch('core.expression.sd.OutputStream')
    def test_callback_breath_rate_update(self, mock_output_stream):
        """Test that breath rate updates based on target_amp."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        initial_breath_rate = afl.breath_rate
        afl.target_amp = 1.0  # High attention
        
        outdata = np.zeros((1024, 1), dtype=np.float32)
        # Run callback multiple times to allow smoothing
        for _ in range(100):
            afl.callback(outdata, 1024, {}, None)
        
        # Breath rate should have increased (target is 0.2 + amp * 2.0)
        assert afl.current_breath_rate >= initial_breath_rate
    
    @patch('core.expression.sd.OutputStream')
    def test_callback_phase_continuity(self, mock_output_stream):
        """Test that phase is continuous across callbacks."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        outdata1 = np.zeros((1024, 1), dtype=np.float32)
        outdata2 = np.zeros((1024, 1), dtype=np.float32)
        
        phase_before = afl.phase
        afl.callback(outdata1, 1024, {}, None)
        phase_after_first = afl.phase
        afl.callback(outdata2, 1024, {}, None)
        phase_after_second = afl.phase
        
        # Phase should advance continuously
        assert phase_after_first != phase_before
        assert phase_after_second != phase_after_first
    
    @patch('core.expression.sd.OutputStream')
    def test_cleanup(self, mock_output_stream):
        """Test cleanup stops and closes stream."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop()
        afl.cleanup()
        
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
    
    @patch('core.expression.sd.OutputStream')
    def test_start_exception_handling(self, mock_output_stream, capsys):
        """Test start method exception handling (covers lines 116-118)."""
        mock_output_stream.side_effect = Exception("Stream error")
        
        afl = AestheticFeedbackLoop.__new__(AestheticFeedbackLoop)
        afl.base_pitch = 440.0
        afl.sample_rate = SAMPLE_RATE
        afl.current_frequency = 440.0
        afl.target_frequency = 440.0
        afl.phase = 0.0
        afl.current_amp = 0.0
        afl.target_amp = 0.0
        afl.breath_phase = 0.0
        afl.breath_rate = 0.2
        afl.current_breath_rate = 0.2
        afl.syllable_phase = 0.0
        afl.is_speaking = False
        afl.stream = None
        afl.running = False
        
        afl.start()
        captured = capsys.readouterr()
        assert "[AFL] Error" in captured.out
        assert afl.running is False
    
    @patch('core.expression.sd.OutputStream')
    def test_update_pitch(self, mock_output_stream):
        """Test update_pitch method (covers line 121)."""
        mock_stream = MagicMock()
        mock_output_stream.return_value = mock_stream
        
        afl = AestheticFeedbackLoop(base_pitch=440.0)
        assert afl.base_pitch == 440.0
        
        afl.update_pitch(220.0)
        assert afl.base_pitch == 220.0


class TestPhonemeAnalyzer:
    """Test suite for PhonemeAnalyzer class."""
    
    def test_init_default(self):
        """Test PhonemeAnalyzer initialization with defaults."""
        analyzer = PhonemeAnalyzer()
        assert analyzer.speech_threshold == 1.5
        assert analyzer.max_history == 10000
        assert len(analyzer.phoneme_history) == 0
    
    def test_init_custom(self):
        """Test PhonemeAnalyzer initialization with custom values."""
        analyzer = PhonemeAnalyzer(speech_threshold=2.0, max_history=5000)
        assert analyzer.speech_threshold == 2.0
        assert analyzer.max_history == 5000
    
    def test_analyze_below_threshold(self):
        """Test analyze with s_score below speech threshold."""
        analyzer = PhonemeAnalyzer(speech_threshold=1.5)
        result = analyzer.analyze(1.0, 300.0)
        assert result is None
        assert len(analyzer.phoneme_history) == 0
    
    def test_analyze_vowel_high_pitch(self):
        """Test analyze returns vowel 'a' for high pitch."""
        analyzer = PhonemeAnalyzer()
        result = analyzer.analyze(2.0, 300.0)  # s_score < 3.0, pitch > 200
        assert result == "a"
        assert len(analyzer.phoneme_history) == 1
        assert analyzer.phoneme_history[0]['phoneme'] == "a"
    
    def test_analyze_vowel_low_pitch(self):
        """Test analyze returns vowel 'o' for low pitch."""
        analyzer = PhonemeAnalyzer()
        result = analyzer.analyze(2.0, 150.0)  # s_score < 3.0, pitch <= 200
        assert result == "o"
        assert analyzer.phoneme_history[0]['phoneme'] == "o"
    
    def test_analyze_fricative(self):
        """Test analyze returns fricative 's'."""
        analyzer = PhonemeAnalyzer()
        result = analyzer.analyze(4.0, 300.0)  # 3.0 <= s_score < 6.0
        assert result == "s"
        assert analyzer.phoneme_history[0]['phoneme'] == "s"
    
    def test_analyze_plosive(self):
        """Test analyze returns plosive 'k'."""
        analyzer = PhonemeAnalyzer()
        result = analyzer.analyze(7.0, 300.0)  # s_score >= 6.0
        assert result == "k"
        assert analyzer.phoneme_history[0]['phoneme'] == "k"
    
    def test_analyze_boundary_vowel_fricative(self):
        """Test analyze at boundary between vowel and fricative."""
        analyzer = PhonemeAnalyzer()
        # Just below 3.0
        result1 = analyzer.analyze(2.99, 300.0)
        assert result1 == "a"
        # Just at 3.0
        result2 = analyzer.analyze(3.0, 300.0)
        assert result2 == "s"
    
    def test_analyze_boundary_fricative_plosive(self):
        """Test analyze at boundary between fricative and plosive."""
        analyzer = PhonemeAnalyzer()
        # Just below 6.0
        result1 = analyzer.analyze(5.99, 300.0)
        assert result1 == "s"
        # Just at 6.0
        result2 = analyzer.analyze(6.0, 300.0)
        assert result2 == "k"
    
    def test_analyze_stores_history(self):
        """Test that analyze stores phoneme in history."""
        analyzer = PhonemeAnalyzer()
        analyzer.analyze(2.0, 300.0)
        assert len(analyzer.phoneme_history) == 1
        assert analyzer.phoneme_history[0]['s_score'] == 2.0
        assert analyzer.phoneme_history[0]['pitch'] == 300.0
    
    def test_analyze_history_capping(self):
        """Test that phoneme history is capped."""
        analyzer = PhonemeAnalyzer(max_history=10)
        # Add more than max_history
        for i in range(15):
            analyzer.analyze(2.0, 300.0)
        
        assert len(analyzer.phoneme_history) <= analyzer.max_history
    
    def test_get_recent_phonemes(self):
        """Test get_recent_phonemes."""
        analyzer = PhonemeAnalyzer()
        analyzer.analyze(2.0, 300.0)  # 'a'
        analyzer.analyze(4.0, 300.0)  # 's'
        analyzer.analyze(7.0, 300.0)  # 'k'
        
        recent = analyzer.get_recent_phonemes(2)
        assert len(recent) == 2
        assert recent == ['s', 'k']
    
    def test_get_recent_phonemes_count_exceeds_history(self):
        """Test get_recent_phonemes when count exceeds history."""
        analyzer = PhonemeAnalyzer()
        analyzer.analyze(2.0, 300.0)
        analyzer.analyze(4.0, 300.0)
        
        recent = analyzer.get_recent_phonemes(10)
        assert len(recent) == 2
    
    def test_get_phoneme_statistics_empty(self):
        """Test get_phoneme_statistics with empty history."""
        analyzer = PhonemeAnalyzer()
        stats = analyzer.get_phoneme_statistics()
        assert stats['total'] == 0
        assert stats['vowels'] == 0
        assert stats['fricatives'] == 0
        assert stats['plosives'] == 0
    
    def test_get_phoneme_statistics_with_data(self):
        """Test get_phoneme_statistics with phoneme data."""
        analyzer = PhonemeAnalyzer()
        analyzer.analyze(2.0, 300.0)  # 'a' (vowel)
        analyzer.analyze(2.0, 150.0)  # 'o' (vowel)
        analyzer.analyze(4.0, 300.0)  # 's' (fricative)
        analyzer.analyze(7.0, 300.0)  # 'k' (plosive)
        
        stats = analyzer.get_phoneme_statistics()
        assert stats['total'] == 4
        assert stats['vowels'] == 2
        assert stats['fricatives'] == 1
        assert stats['plosives'] == 1
        assert 'distribution' in stats
        assert stats['distribution']['vowels'] == 0.5

