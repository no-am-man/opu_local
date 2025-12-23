"""
Tests for core/patterns/maturity_state.py - State Pattern
100% code coverage target
"""

import pytest
from core.patterns.maturity_state import (
    MaturityState,
    MaturityContext,
    ChildState,
    InfantState,
    AdolescentState,
    AdultState,
    ElderState,
    SageState
)


class TestMaturityState:
    """Test suite for abstract MaturityState class."""
    
    def test_state_is_abstract(self):
        """Test that MaturityState cannot be instantiated."""
        from abc import ABC
        assert issubclass(MaturityState, ABC)
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            MaturityState()


class TestMaturityStates:
    """Test suite for maturity state classes."""
    
    def test_child_state(self):
        """Test ChildState properties."""
        state = ChildState()
        assert state.get_level() == 0
        assert state.get_pitch_multiplier() == 1.0
        assert state.get_stability_threshold() == 3.0
        assert state.get_time_scale() == "1 minute"
    
    def test_infant_state(self):
        """Test InfantState properties."""
        state = InfantState()
        assert state.get_level() == 1
        assert state.get_pitch_multiplier() == 0.9
        assert state.get_stability_threshold() == 3.5
        assert state.get_time_scale() == "1 hour"
    
    def test_adolescent_state(self):
        """Test AdolescentState properties."""
        state = AdolescentState()
        assert state.get_level() == 2
        assert state.get_pitch_multiplier() == 0.7
        assert state.get_stability_threshold() == 4.5
        assert state.get_time_scale() == "1 day"
    
    def test_adult_state(self):
        """Test AdultState properties."""
        state = AdultState()
        assert state.get_level() == 3
        assert state.get_pitch_multiplier() == 0.5
        assert state.get_stability_threshold() == 5.5
        assert state.get_time_scale() == "1 week"
    
    def test_elder_state(self):
        """Test ElderState properties."""
        state = ElderState()
        assert state.get_level() == 4
        assert state.get_pitch_multiplier() == 0.35
        assert state.get_stability_threshold() == 6.5
        assert state.get_time_scale() == "1 month"
    
    def test_sage_state(self):
        """Test SageState properties."""
        state = SageState()
        assert state.get_level() == 5
        assert state.get_pitch_multiplier() == 0.25
        assert state.get_stability_threshold() == 8.0
        assert state.get_time_scale() == "1 year"


class TestMaturityContext:
    """Test suite for MaturityContext."""
    
    def test_init(self):
        """Test MaturityContext initialization."""
        context = MaturityContext()
        assert context.get_level() == 0
        assert context.get_pitch() == 440.0  # 440 * 1.0
        assert context.get_stability_threshold() == 3.0
    
    def test_transition_to_level(self):
        """Test transitioning to different levels."""
        context = MaturityContext()
        
        context.transition_to_level(5)
        assert context.get_level() == 5
        assert context.get_time_scale() == "1 year"
        assert context.get_pitch() == 110.0  # 440 * 0.25
        assert context.get_stability_threshold() == 8.0
        
        context.transition_to_level(2)
        assert context.get_level() == 2
        assert context.get_time_scale() == "1 day"
    
    def test_transition_to_level_bounds(self):
        """Test transitioning with out-of-bounds levels."""
        context = MaturityContext()
        
        # Should handle gracefully
        context.transition_to_level(-1)
        assert context.get_level() == 0  # Should stay at 0
        
        context.transition_to_level(10)
        assert context.get_level() == 0  # Should stay at 0
    
    def test_get_current_state(self):
        """Test get_current_state method."""
        context = MaturityContext()
        state = context.get_current_state()
        assert isinstance(state, ChildState)
        assert state.get_level() == 0
    
    def test_get_pitch(self):
        """Test get_pitch method."""
        context = MaturityContext()
        assert context.get_pitch() == 440.0
        
        context.transition_to_level(5)
        assert context.get_pitch() == 110.0  # 440 * 0.25
    
    def test_set_base_pitch(self):
        """Test set_base_pitch method."""
        context = MaturityContext()
        context.set_base_pitch(220.0)
        assert context.get_pitch() == 220.0  # 220 * 1.0
        
        context.transition_to_level(5)
        assert context.get_pitch() == 55.0  # 220 * 0.25
    
    def test_get_stability_threshold(self):
        """Test get_stability_threshold method."""
        context = MaturityContext()
        assert context.get_stability_threshold() == 3.0
        
        context.transition_to_level(5)
        assert context.get_stability_threshold() == 8.0
    
    def test_get_time_scale(self):
        """Test get_time_scale method."""
        context = MaturityContext()
        assert context.get_time_scale() == "1 minute"
        
        context.transition_to_level(3)
        assert context.get_time_scale() == "1 week"
    
    def test_all_levels(self):
        """Test all maturity levels."""
        context = MaturityContext()
        
        for level in range(6):
            context.transition_to_level(level)
            assert context.get_level() == level
            assert context.get_pitch() > 0
            assert context.get_stability_threshold() >= 3.0
            assert len(context.get_time_scale()) > 0

