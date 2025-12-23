"""
Tests for core/patterns/commands.py - Command Pattern
100% code coverage target
"""

import pytest
from core.patterns.commands import (
    Command,
    StoreMemoryCommand,
    EvolveCharacterCommand,
    ConsolidateMemoryCommand,
    CommandInvoker
)
from core.brain import Brain


class TestCommand:
    """Test suite for abstract Command class."""
    
    def test_command_is_abstract(self):
        """Test that Command cannot be instantiated."""
        from abc import ABC
        assert issubclass(Command, ABC)
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            Command()


class TestStoreMemoryCommand:
    """Test suite for StoreMemoryCommand."""
    
    def test_init(self):
        """Test StoreMemoryCommand initialization."""
        brain = Brain()
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        assert cmd.brain == brain
        assert cmd.genomic_bit == 0.5
        assert cmd.s_score == 1.2
        assert cmd.sense_label == "AUDIO_V1"
        assert cmd.stored is False
    
    def test_execute(self):
        """Test executing StoreMemoryCommand."""
        brain = Brain()
        initial_count = sum(len(level) for level in brain.memory_levels.values())
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        cmd.execute()
        assert cmd.stored is True
        final_count = sum(len(level) for level in brain.memory_levels.values())
        assert final_count == initial_count + 1
    
    def test_undo(self):
        """Test undoing StoreMemoryCommand."""
        brain = Brain()
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        cmd.execute()
        initial_count = sum(len(level) for level in brain.memory_levels.values())
        cmd.undo()
        final_count = sum(len(level) for level in brain.memory_levels.values())
        assert final_count == initial_count - 1
        assert cmd.stored is False
    
    def test_undo_not_executed(self):
        """Test undoing command that wasn't executed."""
        brain = Brain()
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        # Should not raise error
        cmd.undo()
    
    def test_get_description(self):
        """Test get_description method."""
        brain = Brain()
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        desc = cmd.get_description()
        assert "Store memory" in desc
        assert "AUDIO_V1" in desc


class TestEvolveCharacterCommand:
    """Test suite for EvolveCharacterCommand."""
    
    def test_init(self):
        """Test EvolveCharacterCommand initialization."""
        brain = Brain()
        cmd = EvolveCharacterCommand(brain, 3)
        assert cmd.brain == brain
        assert cmd.level == 3
        assert cmd.previous_state is None
    
    def test_execute(self):
        """Test executing EvolveCharacterCommand."""
        brain = Brain()
        # Add some memories to trigger evolution
        for _ in range(20):
            brain.store_memory(0.5, 2.5, "AUDIO_V1")
        brain.consolidate_memory(2)
        
        initial_maturity = brain.character_profile['maturity_index']
        cmd = EvolveCharacterCommand(brain, 2)
        cmd.execute()
        assert cmd.previous_state is not None
        assert 'maturity_index' in cmd.previous_state
    
    def test_undo(self):
        """Test undoing EvolveCharacterCommand."""
        brain = Brain()
        # Add memories and evolve
        for _ in range(20):
            brain.store_memory(0.5, 2.5, "AUDIO_V1")
        brain.consolidate_memory(2)
        
        initial_maturity = brain.character_profile['maturity_index']
        cmd = EvolveCharacterCommand(brain, 2)
        cmd.execute()
        cmd.undo()
        assert brain.character_profile['maturity_index'] == initial_maturity
    
    def test_get_description(self):
        """Test get_description method."""
        brain = Brain()
        cmd = EvolveCharacterCommand(brain, 3)
        desc = cmd.get_description()
        assert "Evolve character" in desc
        assert "level 3" in desc


class TestConsolidateMemoryCommand:
    """Test suite for ConsolidateMemoryCommand."""
    
    def test_init(self):
        """Test ConsolidateMemoryCommand initialization."""
        brain = Brain()
        cmd = ConsolidateMemoryCommand(brain, 2)
        assert cmd.brain == brain
        assert cmd.level == 2
        assert cmd.previous_levels is None
    
    def test_execute(self):
        """Test executing ConsolidateMemoryCommand."""
        brain = Brain()
        # Add memories to level 2
        for _ in range(20):
            brain.store_memory(0.5, 2.5, "AUDIO_V1")
        
        initial_level_3_count = len(brain.memory_levels[3])
        cmd = ConsolidateMemoryCommand(brain, 2)
        cmd.execute()
        assert cmd.previous_levels is not None
        assert 2 in cmd.previous_levels
    
    def test_undo(self):
        """Test undoing ConsolidateMemoryCommand."""
        brain = Brain()
        # Add memories directly to level 2 (bypassing store_memory which might trigger consolidation)
        for i in range(20):
            brain.memory_levels[2].append({
                'genomic_bit': 0.5,
                's_score': 2.5,
                'sense': 'AUDIO_V1',
                'timestamp': i
            })
        
        initial_level_2_count = len(brain.memory_levels[2])
        initial_level_3_count = len(brain.memory_levels[3])
        cmd = ConsolidateMemoryCommand(brain, 2)
        cmd.execute()
        # After consolidation, level 3 should have 1 more abstraction
        # Note: consolidate_memory doesn't clear level 2, it just creates an abstraction
        assert len(brain.memory_levels[3]) == initial_level_3_count + 1
        cmd.undo()
        # Memory should be restored to initial state
        assert len(brain.memory_levels[2]) == initial_level_2_count
        assert len(brain.memory_levels[3]) == initial_level_3_count
    
    def test_get_description(self):
        """Test get_description method."""
        brain = Brain()
        cmd = ConsolidateMemoryCommand(brain, 2)
        desc = cmd.get_description()
        assert "Consolidate memory" in desc
        assert "level 2" in desc


class TestCommandInvoker:
    """Test suite for CommandInvoker."""
    
    def test_init(self):
        """Test CommandInvoker initialization."""
        invoker = CommandInvoker()
        assert len(invoker.history) == 0
        assert len(invoker.undo_stack) == 0
    
    def test_init_with_max_history(self):
        """Test CommandInvoker with max_history."""
        invoker = CommandInvoker(max_history=10)
        assert invoker.max_history == 10
    
    def test_execute_command(self):
        """Test executing a command."""
        invoker = CommandInvoker()
        brain = Brain()
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        invoker.execute_command(cmd)
        assert len(invoker.history) == 1
        assert cmd.stored is True
    
    def test_execute_command_clears_undo_stack(self):
        """Test that executing command clears undo stack."""
        invoker = CommandInvoker()
        brain = Brain()
        cmd1 = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        cmd2 = StoreMemoryCommand(brain, 0.6, 1.3, "AUDIO_V1")
        invoker.execute_command(cmd1)
        invoker.undo()
        assert len(invoker.undo_stack) == 1
        invoker.execute_command(cmd2)
        assert len(invoker.undo_stack) == 0
    
    def test_execute_command_history_capping(self):
        """Test that history is capped at max_history."""
        invoker = CommandInvoker(max_history=3)
        brain = Brain()
        for i in range(5):
            cmd = StoreMemoryCommand(brain, float(i), 1.0, "AUDIO_V1")
            invoker.execute_command(cmd)
        assert len(invoker.history) == 3
    
    def test_undo(self):
        """Test undoing a command."""
        invoker = CommandInvoker()
        brain = Brain()
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        invoker.execute_command(cmd)
        initial_count = sum(len(level) for level in brain.memory_levels.values())
        result = invoker.undo()
        assert result is True
        assert len(invoker.history) == 0
        assert len(invoker.undo_stack) == 1
        final_count = sum(len(level) for level in brain.memory_levels.values())
        assert final_count == initial_count - 1
    
    def test_undo_empty_history(self):
        """Test undoing with empty history."""
        invoker = CommandInvoker()
        result = invoker.undo()
        assert result is False
    
    def test_redo(self):
        """Test redoing a command."""
        invoker = CommandInvoker()
        brain = Brain()
        cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        invoker.execute_command(cmd)
        initial_count = sum(len(level) for level in brain.memory_levels.values())
        invoker.undo()
        result = invoker.redo()
        assert result is True
        assert len(invoker.history) == 1
        assert len(invoker.undo_stack) == 0
        final_count = sum(len(level) for level in brain.memory_levels.values())
        assert final_count == initial_count
    
    def test_redo_empty_stack(self):
        """Test redoing with empty undo stack."""
        invoker = CommandInvoker()
        result = invoker.redo()
        assert result is False
    
    def test_get_history(self):
        """Test get_history method."""
        invoker = CommandInvoker()
        brain = Brain()
        cmd1 = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        cmd2 = StoreMemoryCommand(brain, 0.6, 1.3, "AUDIO_V1")
        invoker.execute_command(cmd1)
        invoker.execute_command(cmd2)
        history = invoker.get_history()
        assert len(history) == 2
        assert history[0] == cmd1
        assert history[1] == cmd2
        # Should be a copy
        assert history is not invoker.history
    
    def test_clear_history(self):
        """Test clear_history method."""
        invoker = CommandInvoker()
        brain = Brain()
        cmd1 = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
        cmd2 = StoreMemoryCommand(brain, 0.6, 1.3, "AUDIO_V1")
        invoker.execute_command(cmd1)
        invoker.undo()
        invoker.execute_command(cmd2)
        invoker.clear_history()
        assert len(invoker.history) == 0
        assert len(invoker.undo_stack) == 0

