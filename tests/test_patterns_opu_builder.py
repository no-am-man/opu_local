"""
Tests for core/patterns/opu_builder.py - Builder Pattern
100% code coverage target
"""

import pytest
from core.patterns.opu_builder import OPUBuilder
from core.opu import OrthogonalProcessingUnit


class TestOPUBuilder:
    """Test suite for OPUBuilder."""
    
    def test_init(self):
        """Test OPUBuilder initialization."""
        builder = OPUBuilder()
        assert builder._audio_config['max_history_size'] == 50  # Updated default for higher sensitivity
        assert builder._visual_config['max_history'] == 50  # Updated default for higher sensitivity
        assert len(builder._senses) == 0
        assert builder._use_strategies is False
        assert builder._use_maturity_context is False
    
    def test_with_brain_config(self):
        """Test with_brain_config method."""
        builder = OPUBuilder()
        builder.with_brain_config(test_param=123)
        assert builder._brain_config['test_param'] == 123
    
    def test_with_audio_cortex(self):
        """Test with_audio_cortex method."""
        builder = OPUBuilder()
        builder.with_audio_cortex(max_history=5000, use_strategy=True)
        assert builder._audio_config['max_history_size'] == 5000
        assert builder._use_strategies is True
    
    def test_with_visual_cortex(self):
        """Test with_visual_cortex method."""
        builder = OPUBuilder()
        builder.with_visual_cortex(max_history=50, use_strategy=True)
        assert builder._visual_config['max_history'] == 50
        assert builder._use_strategies is True
    
    def test_with_genesis_kernel(self):
        """Test with_genesis_kernel method."""
        builder = OPUBuilder()
        builder.with_genesis_kernel(g_empty_set=2.0)
        assert builder._genesis_config['g_empty_set'] == 2.0
    
    def test_add_sense(self):
        """Test add_sense method."""
        builder = OPUBuilder()
        builder.add_sense("AUDIO_V1")
        assert len(builder._senses) == 1
        assert builder._senses[0].get_label() == "AUDIO_V1"
    
    def test_add_multiple_senses(self):
        """Test adding multiple senses."""
        builder = OPUBuilder()
        builder.add_sense("AUDIO_V1")
        builder.add_sense("VIDEO_V1")
        assert len(builder._senses) == 2
    
    def test_with_maturity_context(self):
        """Test with_maturity_context method."""
        builder = OPUBuilder()
        builder.with_maturity_context(enabled=True)
        assert builder._use_maturity_context is True
    
    def test_build_basic(self):
        """Test building basic OPU."""
        builder = OPUBuilder()
        opu = builder.build()
        assert isinstance(opu, OrthogonalProcessingUnit)
        assert opu.audio_cortex.max_history_size == 50  # Updated default for higher sensitivity
        assert opu.vision_cortex.max_visual_history == 50  # Updated default for higher sensitivity
    
    def test_build_with_custom_config(self):
        """Test building OPU with custom configuration."""
        builder = (OPUBuilder()
                   .with_audio_cortex(max_history=5000)
                   .with_visual_cortex(max_history=50)
                   .with_genesis_kernel(g_empty_set=2.0))
        opu = builder.build()
        assert opu.audio_cortex.max_history_size == 5000
        assert opu.vision_cortex.max_visual_history == 50
    
    def test_build_with_senses(self):
        """Test building OPU with senses."""
        builder = (OPUBuilder()
                   .add_sense("AUDIO_V1")
                   .add_sense("VIDEO_V1"))
        opu = builder.build()
        assert isinstance(opu, OrthogonalProcessingUnit)
    
    def test_build_with_maturity_context(self):
        """Test building OPU with maturity context."""
        builder = (OPUBuilder()
                   .with_maturity_context(enabled=True))
        opu = builder.build()
        assert isinstance(opu, OrthogonalProcessingUnit)
        assert hasattr(opu, 'maturity_context')
    
    def test_build_with_strategies(self):
        """Test building OPU with strategies."""
        builder = (OPUBuilder()
                   .with_audio_cortex(use_strategy=True)
                   .with_visual_cortex(use_strategy=True))
        opu = builder.build()
        assert isinstance(opu, OrthogonalProcessingUnit)
    
    def test_build_fluent_interface(self):
        """Test fluent interface (method chaining)."""
        builder = (OPUBuilder()
                   .with_audio_cortex(max_history=5000)
                   .with_visual_cortex(max_history=50)
                   .with_genesis_kernel(g_empty_set=1.0)
                   .add_sense("AUDIO_V1")
                   .add_sense("VIDEO_V1")
                   .with_maturity_context(enabled=True))
        opu = builder.build()
        assert isinstance(opu, OrthogonalProcessingUnit)
    
    def test_reset(self):
        """Test reset method."""
        builder = (OPUBuilder()
                   .with_audio_cortex(max_history=5000)
                   .add_sense("AUDIO_V1"))
        builder.reset()
        assert builder._audio_config['max_history_size'] == 50  # Updated default for higher sensitivity
        assert len(builder._senses) == 0
    
    def test_build_replaces_components(self):
        """Test that build replaces OPU components."""
        builder = OPUBuilder()
        opu = builder.build()
        # Components should be replaced
        assert opu.brain is not None
        assert opu.audio_cortex is not None
        assert opu.vision_cortex is not None

