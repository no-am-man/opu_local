"""
Builder Pattern: OPU Configuration.

Allows flexible, step-by-step construction of OPU instances with complex configuration.
"""

from typing import Optional, Dict, Any, List
from core.brain import Brain
from core.audio_cortex import AudioCortex
from core.vision_cortex import VisualCortex
from core.genesis import GenesisKernel
from core.patterns.introspection_strategy import (
    AudioIntrospectionStrategy,
    VisualIntrospectionStrategy
)
from core.patterns.maturity_state import MaturityContext
from core.patterns.sense_factory import SenseFactory, Sense
from config import OPU_BUILDER_DEFAULT_AUDIO_HISTORY, OPU_BUILDER_DEFAULT_VISUAL_HISTORY
# Note: OrthogonalProcessingUnit import is lazy to avoid circular dependency


class OPUBuilder:
    """Builder for configuring OPU instances."""
    
    def __init__(self):
        """Initialize builder with default configuration."""
        self._brain_config: Dict[str, Any] = {}
        self._audio_config: Dict[str, Any] = {'max_history_size': OPU_BUILDER_DEFAULT_AUDIO_HISTORY}  # Reduced for higher sensitivity
        self._visual_config: Dict[str, Any] = {'max_history': OPU_BUILDER_DEFAULT_VISUAL_HISTORY}  # Reduced for higher sensitivity
        self._genesis_config: Dict[str, Any] = {}
        self._senses: List[Sense] = []
        self._use_strategies = False
        self._use_maturity_context = False
    
    def with_brain_config(self, **kwargs) -> 'OPUBuilder':
        """
        Configure brain settings.
        
        Args:
            **kwargs: Brain configuration parameters
            
        Returns:
            self for method chaining
        """
        self._brain_config.update(kwargs)
        return self
    
    def with_audio_cortex(self, max_history: int = OPU_BUILDER_DEFAULT_AUDIO_HISTORY,
                         use_strategy: bool = False) -> 'OPUBuilder':
        """
        Configure audio cortex.
        
        Args:
            max_history: Maximum history size
            use_strategy: Use strategy pattern for introspection
            
        Returns:
            self for method chaining
        """
        self._audio_config['max_history_size'] = max_history
        self._use_strategies = use_strategy or self._use_strategies
        return self
    
    def with_visual_cortex(self, max_history: int = OPU_BUILDER_DEFAULT_VISUAL_HISTORY,
                          use_strategy: bool = False) -> 'OPUBuilder':
        """
        Configure visual cortex.
        
        Args:
            max_history: Maximum history size
            use_strategy: Use strategy pattern for introspection
            
        Returns:
            self for method chaining
        """
        self._visual_config['max_history'] = max_history
        self._use_strategies = use_strategy or self._use_strategies
        return self
    
    def with_genesis_kernel(self, g_empty_set: float = 1.0) -> 'OPUBuilder':
        """
        Configure Genesis kernel.
        
        Args:
            g_empty_set: Genesis constant
            
        Returns:
            self for method chaining
        """
        self._genesis_config['g_empty_set'] = g_empty_set
        return self
    
    def add_sense(self, sense_label: str, **kwargs) -> 'OPUBuilder':
        """
        Add a sense to the OPU.
        
        Args:
            sense_label: Sense label (e.g., 'AUDIO_V1')
            **kwargs: Additional arguments for sense creation
            
        Returns:
            self for method chaining
        """
        sense = SenseFactory.create_sense(sense_label, **kwargs)
        self._senses.append(sense)
        return self
    
    def with_maturity_context(self, enabled: bool = True) -> 'OPUBuilder':
        """
        Enable maturity context (state pattern).
        
        Args:
            enabled: Whether to use maturity context
            
        Returns:
            self for method chaining
        """
        self._use_maturity_context = enabled
        return self
    
    def build(self):
        """
        Build configured OPU instance.
        
        Returns:
            Configured OrthogonalProcessingUnit instance
        """
        # Lazy import to avoid circular dependency
        from core.opu import OrthogonalProcessingUnit
        
        # Build brain
        brain = Brain(**self._brain_config)
        
        # Build audio cortex (with or without strategy)
        if self._use_strategies:
            audio_strategy = AudioIntrospectionStrategy(
                max_history_size=self._audio_config['max_history_size']
            )
            # For now, we'll still use AudioCortex but could refactor to use strategy directly
            audio_cortex = AudioCortex(**self._audio_config)
        else:
            audio_cortex = AudioCortex(**self._audio_config)
        
        # Build visual cortex
        visual_cortex = VisualCortex(**self._visual_config)
        
        # Build OPU
        opu = OrthogonalProcessingUnit()
        
        # Replace components if needed
        opu.brain = brain
        opu.audio_cortex = audio_cortex
        opu.vision_cortex = visual_cortex
        
        # Apply maturity context if enabled
        if self._use_maturity_context:
            maturity_context = MaturityContext()
            # This would require refactoring Brain to use MaturityContext
            # For now, we'll store it as an attribute
            opu.maturity_context = maturity_context
        
        return opu
    
    def reset(self) -> 'OPUBuilder':
        """
        Reset builder to default state.
        
        Returns:
            self for method chaining
        """
        self.__init__()
        return self

