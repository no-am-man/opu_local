"""
Command Pattern: Action Encapsulation.

Encapsulates OPU actions as commands, enabling undo/redo, queuing, and logging.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Command(ABC):
    """Abstract command interface."""
    
    @abstractmethod
    def execute(self):
        """Execute the command."""
        pass
    
    @abstractmethod
    def undo(self):
        """Undo the command (if supported)."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the command."""
        pass


class StoreMemoryCommand(Command):
    """Command to store memory."""
    
    def __init__(self, brain, genomic_bit: float, s_score: float, sense_label: str):
        """
        Initialize store memory command.
        
        Args:
            brain: Brain instance
            genomic_bit: Genomic bit to store
            s_score: Surprise score
            sense_label: Sense label
        """
        self.brain = brain
        self.genomic_bit = genomic_bit
        self.s_score = s_score
        self.sense_label = sense_label
        self.stored = False
        self.memory_index = None
    
    def execute(self):
        """Execute memory storage."""
        # Calculate timestamp
        timestamp = sum(len(level_mem) for level_mem in self.brain.memory_levels.values())
        self.brain.store_memory(
            self.genomic_bit,
            self.s_score,
            sense_label=self.sense_label,
            timestamp=timestamp
        )
        self.stored = True
    
    def undo(self):
        """Undo memory storage (remove last entry)."""
        if not self.stored:
            return
        
        # Find and remove the most recent memory with matching characteristics
        for level in range(5, -1, -1):
            memories = self.brain.memory_levels[level]
            for i in range(len(memories) - 1, -1, -1):
                mem = memories[i]
                if (mem.get('genomic_bit') == self.genomic_bit and
                    mem.get('sense') == self.sense_label):
                    memories.pop(i)
                    self.stored = False
                    return
    
    def get_description(self):
        """Get command description."""
        return f"Store memory: {self.sense_label}, s_score={self.s_score:.2f}"


class EvolveCharacterCommand(Command):
    """Command to evolve character."""
    
    def __init__(self, brain, level: int):
        """
        Initialize evolve character command.
        
        Args:
            brain: Brain instance
            level: Maturity level that triggered evolution
        """
        self.brain = brain
        self.level = level
        self.previous_state = None
    
    def execute(self):
        """Execute character evolution."""
        # Save previous state for undo
        self.previous_state = self.brain.character_profile.copy()
        self.brain.evolve_character(self.level)
    
    def undo(self):
        """Undo character evolution."""
        if self.previous_state:
            self.brain.character_profile = self.previous_state.copy()
    
    def get_description(self):
        """Get command description."""
        return f"Evolve character to level {self.level}"


class ConsolidateMemoryCommand(Command):
    """Command to consolidate memory."""
    
    def __init__(self, brain, level: int):
        """
        Initialize consolidate memory command.
        
        Args:
            brain: Brain instance
            level: Memory level to consolidate
        """
        self.brain = brain
        self.level = level
        self.previous_levels = None
    
    def execute(self):
        """Execute memory consolidation."""
        # Save previous state
        self.previous_levels = {
            lvl: [m.copy() for m in memories]
            for lvl, memories in self.brain.memory_levels.items()
        }
        self.brain.consolidate_memory(self.level)
    
    def undo(self):
        """Undo memory consolidation."""
        if self.previous_levels:
            self.brain.memory_levels = {
                lvl: [m.copy() for m in memories]
                for lvl, memories in self.previous_levels.items()
            }
    
    def get_description(self):
        """Get command description."""
        return f"Consolidate memory level {self.level}"


class CommandInvoker:
    """Invokes commands and maintains history."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize command invoker.
        
        Args:
            max_history: Maximum number of commands to keep in history
        """
        self.history: list[Command] = []
        self.undo_stack: list[Command] = []
        self.max_history = max_history
    
    def execute_command(self, command: Command):
        """
        Execute a command.
        
        Args:
            command: Command to execute
        """
        command.execute()
        self.history.append(command)
        self.undo_stack.clear()  # Clear undo stack when new command executed
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def undo(self) -> bool:
        """
        Undo last command.
        
        Returns:
            bool: True if undo was successful
        """
        if not self.history:
            return False
        
        command = self.history.pop()
        command.undo()
        self.undo_stack.append(command)
        return True
    
    def redo(self) -> bool:
        """
        Redo last undone command.
        
        Returns:
            bool: True if redo was successful
        """
        if not self.undo_stack:
            return False
        
        command = self.undo_stack.pop()
        command.execute()
        self.history.append(command)
        return True
    
    def get_history(self) -> list[Command]:
        """Get command history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear command history."""
        self.history.clear()
        self.undo_stack.clear()

