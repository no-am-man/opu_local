"""
Tests for core/genesis.py - The Safety Kernel
100% code coverage target
"""

import pytest
import numpy as np
from core.genesis import GenesisKernel
from config import G_EMPTY_SET, MAX_DISSONANCE


class TestGenesisKernel:
    """Test suite for GenesisKernel class."""
    
    def test_init(self):
        """Test GenesisKernel initialization."""
        kernel = GenesisKernel()
        assert kernel.g_empty_set == G_EMPTY_SET
        assert kernel.max_dissonance == MAX_DISSONANCE
        assert kernel.max_action_magnitude == 1.0
    
    def test_ethical_veto_none_vector(self):
        """Test ethical_veto with None vector."""
        kernel = GenesisKernel()
        result = kernel.ethical_veto(None)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_ethical_veto_empty_vector(self):
        """Test ethical_veto with empty vector."""
        kernel = GenesisKernel()
        result = kernel.ethical_veto(np.array([]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_ethical_veto_within_threshold(self):
        """Test ethical_veto when dissonance is within threshold."""
        kernel = GenesisKernel()
        # Vector with magnitude < G_EMPTY_SET (dissonance < 1.0)
        vector = np.array([0.5, 0.5])
        result = kernel.ethical_veto(vector)
        np.testing.assert_array_almost_equal(result, vector)
    
    def test_ethical_veto_exceeds_threshold(self):
        """Test ethical_veto when dissonance exceeds threshold."""
        kernel = GenesisKernel()
        # Vector with magnitude > G_EMPTY_SET (dissonance > 1.0)
        vector = np.array([2.0, 2.0])  # Magnitude = sqrt(8) ≈ 2.83
        result = kernel.ethical_veto(vector)
        # Result should be clamped
        result_magnitude = np.linalg.norm(result)
        expected_magnitude = G_EMPTY_SET * MAX_DISSONANCE
        assert result_magnitude <= expected_magnitude
        # Direction should be preserved
        if np.linalg.norm(vector) > 0:
            unit_original = vector / np.linalg.norm(vector)
            unit_result = result / np.linalg.norm(result)
            np.testing.assert_array_almost_equal(unit_original, unit_result)
    
    def test_ethical_veto_exact_threshold(self):
        """Test ethical_veto at exact threshold boundary."""
        kernel = GenesisKernel()
        # Vector with magnitude exactly at threshold
        vector = np.array([G_EMPTY_SET, 0.0])
        result = kernel.ethical_veto(vector)
        # Should not be clamped (dissonance == 1.0, not > 1.0)
        np.testing.assert_array_almost_equal(result, vector)
    
    def test_ethical_veto_zero_magnitude(self):
        """Test ethical_veto with zero magnitude vector."""
        kernel = GenesisKernel()
        vector = np.array([0.0, 0.0])
        result = kernel.ethical_veto(vector)
        np.testing.assert_array_almost_equal(result, vector)
    
    def test_ethical_veto_list_input(self):
        """Test ethical_veto with list input (should convert to array)."""
        kernel = GenesisKernel()
        vector = [1.0, 2.0, 3.0]
        result = kernel.ethical_veto(vector)
        assert isinstance(result, np.ndarray)
    
    def test_ethical_veto_veto_counting(self):
        """Test that veto counting works correctly."""
        kernel = GenesisKernel()
        # Apply many vetos
        for i in range(150):
            vector = np.array([10.0, 10.0])  # High dissonance
            kernel.ethical_veto(vector)
        
        # Should have counted vetos
        assert hasattr(kernel, '_veto_count')
        assert kernel._veto_count == 150
    
    def test_ethical_veto_logging_at_50(self, capsys):
        """Test that logging occurs at 50th veto."""
        kernel = GenesisKernel()
        # Apply exactly 50 vetos
        for i in range(50):
            vector = np.array([10.0, 10.0])
            kernel.ethical_veto(vector)
        
        captured = capsys.readouterr()
        assert "[GENESIS] Veto applied 50 times" in captured.out
    
    def test_ethical_veto_logging_at_100(self, capsys):
        """Test that logging occurs at 100th veto."""
        kernel = GenesisKernel()
        # Apply exactly 100 vetos
        for i in range(100):
            vector = np.array([10.0, 10.0])
            kernel.ethical_veto(vector)
        
        captured = capsys.readouterr()
        # Should log at 50, 100
        assert "[GENESIS] Veto applied 100 times" in captured.out
    
    def test_check_order_none(self):
        """Test check_order with None."""
        kernel = GenesisKernel()
        assert kernel.check_order(None) is True
    
    def test_check_order_empty(self):
        """Test check_order with empty vector."""
        kernel = GenesisKernel()
        assert kernel.check_order(np.array([])) is True
    
    def test_check_order_within_threshold(self):
        """Test check_order when within threshold."""
        kernel = GenesisKernel()
        vector = np.array([0.5, 0.5])
        assert kernel.check_order(vector) == True  # Use == for numpy boolean
    
    def test_check_order_exceeds_threshold(self):
        """Test check_order when exceeds threshold."""
        kernel = GenesisKernel()
        vector = np.array([10.0, 10.0])
        assert kernel.check_order(vector) == False  # Use == for numpy boolean
    
    def test_check_order_at_threshold(self):
        """Test check_order at exact threshold."""
        kernel = GenesisKernel()
        vector = np.array([G_EMPTY_SET, 0.0])
        assert kernel.check_order(vector) == True  # Use == for numpy boolean
    
    def test_check_order_3d_vector(self):
        """Test check_order with 3D vector."""
        kernel = GenesisKernel()
        vector = np.array([1.0, 1.0, 1.0])
        magnitude = np.linalg.norm(vector)
        expected = (magnitude / G_EMPTY_SET) <= MAX_DISSONANCE
        assert kernel.check_order(vector) == expected

