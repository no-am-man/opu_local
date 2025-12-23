"""
Tests for core/patterns/processing_pipeline.py - Template Method Pattern
100% code coverage target
"""

import pytest
import numpy as np
from core.patterns.processing_pipeline import ProcessingPipeline
from core.opu import OrthogonalProcessingUnit
from core.genesis import GenesisKernel


class ConcreteProcessingPipeline(ProcessingPipeline):
    """Concrete implementation for testing."""
    
    def __init__(self, opu, genesis):
        self.opu = opu
        self.genesis = genesis
        self.capture_audio_called = False
        self.capture_visual_called = False
    
    def capture_audio(self):
        """Capture audio (simulated)."""
        self.capture_audio_called = True
        return np.random.randn(1024).astype(np.float32)
    
    def capture_visual(self):
        """Capture visual (simulated)."""
        self.capture_visual_called = True
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def extract_audio_genomic(self, audio_data):
        """Extract audio genomic bit."""
        from core.mic import perceive
        result = perceive(audio_data)
        return result['genomic_bit']
    
    def extract_visual_genomic(self, visual_data):
        """Extract visual genomic vector."""
        if visual_data is None:
            return np.array([0.0, 0.0, 0.0])
        try:
            from core.camera import VisualPerception
            vp = VisualPerception()
            if not vp.active:
                return np.array([0.0, 0.0, 0.0])
            result = vp.analyze_frame(visual_data)
            if result is None or len(result) == 0:
                return np.array([0.0, 0.0, 0.0])
            return result
        except Exception:
            return np.array([0.0, 0.0, 0.0])
    
    def introspect_audio(self, genomic_bit):
        """Introspect audio."""
        return self.opu.introspect(genomic_bit)
    
    def introspect_visual(self, visual_vector):
        """Introspect visual."""
        result = self.opu.introspect_visual(visual_vector)
        # introspect_visual returns (s_visual, channel_scores) tuple
        if isinstance(result, tuple):
            return result
        # If it's not a tuple, wrap it
        return (result, {})
    
    def apply_safety(self, s_score, genomic_bit):
        """Apply safety kernel."""
        action_vector = np.array([s_score, genomic_bit])
        safe_vector = self.genesis.ethical_veto(action_vector)
        return safe_vector[0]
    
    def store_memory(self, audio_genomic, s_score, visual_genomic, s_visual):
        """Store memory."""
        self.opu.store_memory(audio_genomic, s_score, sense_label="AUDIO_V1")
        if s_visual > 0.5:
            visual_genomic_bit = np.max(visual_genomic) if len(visual_genomic) > 0 else 0.0
            self.opu.store_memory(visual_genomic_bit, s_visual, sense_label="VIDEO_V1")
    
    def generate_expression(self, s_score):
        """Generate expression (placeholder)."""
        pass


class TestProcessingPipelineAbstract:
    """Test suite for abstract ProcessingPipeline class."""
    
    def test_pipeline_is_abstract(self):
        """Test that ProcessingPipeline cannot be instantiated."""
        from abc import ABC
        assert issubclass(ProcessingPipeline, ABC)
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            ProcessingPipeline()


class TestProcessingPipeline:
    """Test suite for ProcessingPipeline."""
    
    def test_process_template_method(self):
        """Test the template method process()."""
        opu = OrthogonalProcessingUnit()
        genesis = GenesisKernel()
        pipeline = ConcreteProcessingPipeline(opu, genesis)
        
        result = pipeline.process()
        
        assert pipeline.capture_audio_called is True
        assert pipeline.capture_visual_called is True
        assert 's_audio' in result
        assert 's_visual' in result
        assert 's_global' in result
        assert 'safe_score' in result
    
    def test_fuse_scores_default(self):
        """Test default fuse_scores implementation."""
        opu = OrthogonalProcessingUnit()
        genesis = GenesisKernel()
        pipeline = ConcreteProcessingPipeline(opu, genesis)
        
        # Default: max of both scores
        result = pipeline.fuse_scores(1.5, 2.3)
        assert result == 2.3
    
    def test_fuse_scores_custom(self):
        """Test custom fuse_scores implementation."""
        class CustomPipeline(ConcreteProcessingPipeline):
            def fuse_scores(self, s_audio, s_visual):
                # Custom: average instead of max
                return (s_audio + s_visual) / 2.0
        
        opu = OrthogonalProcessingUnit()
        genesis = GenesisKernel()
        pipeline = CustomPipeline(opu, genesis)
        
        result = pipeline.fuse_scores(1.5, 2.3)
        assert result == 1.9  # (1.5 + 2.3) / 2
    
    def test_create_result(self):
        """Test create_result method."""
        opu = OrthogonalProcessingUnit()
        genesis = GenesisKernel()
        pipeline = ConcreteProcessingPipeline(opu, genesis)
        
        result = pipeline.create_result(
            s_audio=1.5,
            s_visual=2.3,
            s_global=2.3,
            safe_score=2.0,
            channel_scores={'R': 1.0, 'G': 2.0, 'B': 0.5}
        )
        
        assert result['s_audio'] == 1.5
        assert result['s_visual'] == 2.3
        assert result['s_global'] == 2.3
        assert result['safe_score'] == 2.0
        assert result['channel_scores']['R'] == 1.0
    
    def test_create_result_no_channel_scores(self):
        """Test create_result without channel scores."""
        opu = OrthogonalProcessingUnit()
        genesis = GenesisKernel()
        pipeline = ConcreteProcessingPipeline(opu, genesis)
        
        result = pipeline.create_result(
            s_audio=1.5,
            s_visual=2.3,
            s_global=2.3,
            safe_score=2.0
        )
        
        assert result['channel_scores'] == {}
    
    def test_process_full_cycle(self):
        """Test full processing cycle."""
        opu = OrthogonalProcessingUnit()
        genesis = GenesisKernel()
        pipeline = ConcreteProcessingPipeline(opu, genesis)
        
        # Process multiple cycles
        for _ in range(3):
            result = pipeline.process()
            assert 's_audio' in result
            assert 's_visual' in result
            assert 'safe_score' in result

