"""
Tests for core/cortex.py - The Brain (Introspection & Memory Abstraction)
100% code coverage target
"""

import pytest
import numpy as np
from core.opu import OrthogonalProcessingUnit


class TestOrthogonalProcessingUnit:
    """Test suite for OrthogonalProcessingUnit class."""
    
    def test_init(self):
        """Test OrthogonalProcessingUnit initialization."""
        opu = OrthogonalProcessingUnit()
        assert len(opu.memory_levels) == 7  # Updated for 7 levels (0-6)
        assert all(level in opu.memory_levels for level in range(7))
        assert opu.character_profile['maturity_index'] == 0.0
        assert opu.character_profile['maturity_level'] == 0
        assert opu.character_profile['base_pitch'] == 440.0
        assert opu.character_profile['stability_threshold'] == 3.0
        assert opu.s_score == 0.0
        assert opu.coherence == 0.0
        assert opu.g_now is None
    
    def test_introspect_single_value(self):
        """Test introspect with single genomic bit (no history)."""
        opu = OrthogonalProcessingUnit()
        result = opu.introspect(0.5)
        assert result == 0.0  # No surprise with only one data point
        assert opu.g_now == 0.5
        assert len(opu.genomic_bits_history) == 1
    
    def test_introspect_two_values(self):
        """Test introspect with two values (can calculate std dev)."""
        opu = OrthogonalProcessingUnit()
        opu.introspect(0.5)
        result = opu.introspect(1.5)
        assert result >= 0.0
        assert opu.s_score >= 0.0
        assert opu.coherence > 0.0
        assert len(opu.genomic_bits_history) == 2
    
    def test_introspect_zero_sigma(self):
        """Test introspect when sigma_history is zero (constant values)."""
        opu = OrthogonalProcessingUnit()
        opu.introspect(1.0)
        opu.introspect(1.0)  # Same value, sigma = 0
        assert opu.s_score >= 0.0
        assert opu.coherence > 0.0
    
    def test_introspect_history_capping(self):
        """Test that introspection history is capped."""
        opu = OrthogonalProcessingUnit()
        # Add more than max_history_size values
        for i in range(opu.max_history_size + 100):
            opu.introspect(float(i))
        
        # History should be capped
        assert len(opu.genomic_bits_history) <= opu.max_history_size
        assert len(opu.mu_history) <= opu.max_history_size
        assert len(opu.sigma_history) <= opu.max_history_size
    
    def test_store_memory_level_0(self):
        """Test store_memory for level 0 (s_score < 0.5)."""
        opu = OrthogonalProcessingUnit()
        opu.store_memory(0.3, 0.3)
        assert len(opu.memory_levels[0]) == 1
        assert opu.memory_levels[0][0]['genomic_bit'] == 0.3
        assert opu.memory_levels[0][0]['s_score'] == 0.3
    
    def test_store_memory_level_1(self):
        """Test store_memory for level 1 (0.5 <= s_score < 1.0)."""
        opu = OrthogonalProcessingUnit()
        opu.store_memory(0.5, 0.7)
        assert len(opu.memory_levels[1]) == 1
    
    def test_store_memory_level_2(self):
        """Test store_memory for level 2 (1.0 <= s_score < 2.0)."""
        opu = OrthogonalProcessingUnit()
        opu.store_memory(0.5, 1.5)
        assert len(opu.memory_levels[2]) == 1
    
    def test_store_memory_level_3(self):
        """Test store_memory for level 3 (2.0 <= s_score < 3.5)."""
        opu = OrthogonalProcessingUnit()
        opu.store_memory(0.5, 3.0)
        assert len(opu.memory_levels[3]) == 1
    
    def test_store_memory_level_4(self):
        """Test store_memory for level 4 (3.5 <= s_score < 5.0)."""
        opu = OrthogonalProcessingUnit()
        opu.store_memory(0.5, 4.0)
        assert len(opu.memory_levels[4]) == 1
    
    def test_store_memory_level_5(self):
        """Test store_memory for level 5 (s_score >= 5.0)."""
        opu = OrthogonalProcessingUnit()
        opu.store_memory(0.5, 6.0)
        assert len(opu.memory_levels[5]) == 1
    
    def test_store_memory_consolidation_trigger_level_0(self):
        """Test that consolidation is triggered at threshold for level 0."""
        opu = OrthogonalProcessingUnit()
        # Level 0 threshold is 100
        for i in range(100):
            opu.store_memory(0.3, 0.3)
        # Should have consolidated
        assert len(opu.memory_levels[1]) > 0
    
    def test_store_memory_consolidation_trigger_level_2(self):
        """Test that consolidation is triggered at threshold for level 2."""
        opu = OrthogonalProcessingUnit()
        # Level 2 threshold is 20
        for i in range(20):
            opu.store_memory(0.5, 1.5)
        # Should have consolidated
        assert len(opu.memory_levels[3]) > 0
    
    def test_consolidate_memory_empty_level(self):
        """Test consolidate_memory with empty level."""
        opu = OrthogonalProcessingUnit()
        opu.consolidate_memory(0)
        # Should not crash
    
    def test_consolidate_memory_invalid_level(self):
        """Test consolidate_memory with invalid level."""
        opu = OrthogonalProcessingUnit()
        opu.consolidate_memory(99)
        # Should not crash
    
    def test_consolidate_memory_level_0(self):
        """Test consolidate_memory for level 0."""
        opu = OrthogonalProcessingUnit()
        # Add memories to level 0
        for i in range(5):
            opu.memory_levels[0].append({
                'genomic_bit': float(i),
                's_score': 0.3
            })
        opu.consolidate_memory(0)
        # Should create abstraction in level 1
        assert len(opu.memory_levels[1]) > 0
    
    def test_consolidate_memory_with_abstractions(self):
        """Test consolidate_memory with abstracted memories."""
        opu = OrthogonalProcessingUnit()
        # Add abstracted memories (with mean_genomic_bit)
        opu.memory_levels[1].append({
            'mean_genomic_bit': 2.5,
            'pattern_strength': 0.5,
            'count': 5
        })
        opu.consolidate_memory(1)
        # Should create abstraction in level 2
        assert len(opu.memory_levels[2]) > 0
    
    def test_consolidate_memory_invalid_memory_format(self):
        """Test consolidate_memory with invalid memory format."""
        opu = OrthogonalProcessingUnit()
        # Add memory without genomic_bit or mean_genomic_bit
        opu.memory_levels[0].append({'invalid': 'data'})
        opu.consolidate_memory(0)
        # Should skip invalid memories
    
    def test_consolidate_memory_level_6_no_next_level(self):
        """Test consolidate_memory for level 6 (no next level - Scire is the highest)."""
        opu = OrthogonalProcessingUnit()
        opu.memory_levels[6].append({
            'genomic_bit': 1.0,
            's_score': 7.0
        })
        opu.consolidate_memory(6)
        # Should not create level 7 (doesn't exist - level 6 is the maximum)
        assert 7 not in opu.memory_levels
        # Level 6 should still have its memory (consolidation doesn't clear source level)
        assert len(opu.memory_levels[6]) > 0
    
    def test_consolidate_memory_triggers_evolution_level_2(self):
        """Test that consolidation at level 2+ triggers evolution."""
        opu = OrthogonalProcessingUnit()
        initial_maturity = opu.character_profile['maturity_index']
        # Add memories to level 2
        for i in range(5):
            opu.memory_levels[2].append({
                'genomic_bit': float(i),
                's_score': 1.5
            })
        opu.consolidate_memory(2)
        # Maturity should have increased
        assert opu.character_profile['maturity_index'] >= initial_maturity
    
    def test_evolve_character_level_0(self):
        """Test evolve_character at level 0."""
        opu = OrthogonalProcessingUnit()
        opu.memory_levels[0].append({'genomic_bit': 0.5})
        opu.evolve_character(0)
        assert opu.character_profile['maturity_level'] == 0
    
    def test_evolve_character_level_3(self):
        """Test evolve_character at level 3."""
        opu = OrthogonalProcessingUnit()
        opu.memory_levels[3].append({'genomic_bit': 0.5})
        opu.evolve_character(3)
        assert opu.character_profile['maturity_level'] == 3
        assert opu.character_profile['maturity_index'] > 0.0
    
    def test_evolve_character_pitch_drop(self):
        """Test that pitch drops with maturity."""
        opu = OrthogonalProcessingUnit()
        initial_pitch = opu.character_profile['base_pitch']
        opu.memory_levels[5].append({'genomic_bit': 0.5})
        opu.evolve_character(5)
        # Pitch should drop
        assert opu.character_profile['base_pitch'] < initial_pitch
        assert opu.character_profile['base_pitch'] >= 110.0  # Minimum pitch
    
    def test_evolve_character_stability_increase(self):
        """Test that stability threshold increases with maturity."""
        opu = OrthogonalProcessingUnit()
        initial_threshold = opu.character_profile['stability_threshold']
        opu.memory_levels[5].append({'genomic_bit': 0.5})
        opu.evolve_character(5)
        # Threshold should increase
        assert opu.character_profile['stability_threshold'] >= initial_threshold
        assert opu.character_profile['stability_threshold'] <= 8.0  # Maximum threshold
    
    def test_evolve_character_maturity_capped(self):
        """Test that maturity_index is capped at 1.0."""
        opu = OrthogonalProcessingUnit()
        # Fill all levels to maximum
        for level in range(7):
            for i in range(20):
                opu.memory_levels[level].append({'genomic_bit': float(i)})
        opu.evolve_character(5)
        assert opu.character_profile['maturity_index'] <= 1.0
    
    def test_get_character_state(self):
        """Test get_character_state."""
        opu = OrthogonalProcessingUnit()
        state = opu.get_character_state()
        assert 'maturity_index' in state
        assert 'maturity_level' in state
        assert 'base_pitch' in state
        assert 'stability_threshold' in state
        # Should be a copy
        state['maturity_index'] = 999.0
        assert opu.character_profile['maturity_index'] != 999.0
    
    def test_get_current_state(self):
        """Test get_current_state."""
        opu = OrthogonalProcessingUnit()
        opu.introspect(0.5)
        opu.introspect(1.0)
        state = opu.get_current_state()
        assert 's_score' in state
        assert 'coherence' in state
        assert 'g_now' in state
        assert 'maturity' in state
        assert state['s_score'] >= 0.0
        assert 0.0 <= state['coherence'] <= 1.0
    
    def test_introspect_coherence_calculation(self):
        """Test that coherence is calculated correctly."""
        opu = OrthogonalProcessingUnit()
        opu.introspect(0.5)
        opu.introspect(1.0)
        # Coherence should be inverse of surprise
        expected_coherence = 1.0 / (1.0 + opu.s_score)
        assert abs(opu.coherence - expected_coherence) < 1e-10
    
    def test_store_memory_timestamp(self):
        """Test that store_memory includes timestamp."""
        opu = OrthogonalProcessingUnit()
        opu.genomic_bits_history = [0.1, 0.2, 0.3]
        opu.store_memory(0.4, 0.3)
        assert opu.memory_levels[0][0]['timestamp'] == 3

