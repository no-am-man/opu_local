"""
Design Patterns Module.

Contains implementations of Gang of Four (GOF) design patterns
for the OPU architecture.
"""

from core.patterns.introspection_strategy import (
    IntrospectionStrategy,
    AudioIntrospectionStrategy,
    VisualIntrospectionStrategy
)
from core.patterns.observer import OPUObserver, ObservableOPU
from core.patterns.sense_factory import Sense, SenseFactory, AudioSense, VisualSense
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
from core.patterns.processing_pipeline import ProcessingPipeline
from core.patterns.commands import (
    Command,
    StoreMemoryCommand,
    EvolveCharacterCommand,
    ConsolidateMemoryCommand,
    CommandInvoker
)
from core.patterns.sense_decorator import (
    SenseDecorator,
    NoiseGateDecorator,
    NormalizationDecorator,
    HighPassFilterDecorator,
    AmplificationDecorator
)
# OPUBuilder imported lazily to avoid circular dependency
# Use: from core.patterns.opu_builder import OPUBuilder

__all__ = [
    # Strategy Pattern
    'IntrospectionStrategy',
    'AudioIntrospectionStrategy',
    'VisualIntrospectionStrategy',
    # Observer Pattern
    'OPUObserver',
    'ObservableOPU',
    # Factory Pattern
    'Sense',
    'SenseFactory',
    'AudioSense',
    'VisualSense',
    # State Pattern
    'MaturityState',
    'MaturityContext',
    'ChildState',
    'InfantState',
    'AdolescentState',
    'AdultState',
    'ElderState',
    'SageState',
    # Template Method Pattern
    'ProcessingPipeline',
    # Command Pattern
    'Command',
    'StoreMemoryCommand',
    'EvolveCharacterCommand',
    'ConsolidateMemoryCommand',
    'CommandInvoker',
    # Decorator Pattern
    'SenseDecorator',
    'NoiseGateDecorator',
    'NormalizationDecorator',
    'HighPassFilterDecorator',
    'AmplificationDecorator',
    # Builder Pattern
    # Note: OPUBuilder not included here to avoid circular import
    # Use: from core.patterns.opu_builder import OPUBuilder
]

