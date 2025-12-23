# GOF Design Patterns Implementation

This document describes the Gang of Four (GOF) design patterns implemented in the OPU architecture.

## Overview

All 8 major design patterns have been implemented to improve code organization, extensibility, and maintainability:

1. **Strategy Pattern** - Introspection strategies
2. **Observer Pattern** - State change notifications
3. **Factory Pattern** - Sense creation
4. **State Pattern** - Maturity levels
5. **Template Method Pattern** - Processing pipeline
6. **Command Pattern** - Action encapsulation
7. **Decorator Pattern** - Sense preprocessing
8. **Builder Pattern** - OPU configuration

## Pattern Details

### 1. Strategy Pattern (`core/patterns/introspection_strategy.py`)

**Purpose:** Allows interchangeable introspection algorithms.

**Implementation:**
- `IntrospectionStrategy` - Abstract base class
- `AudioIntrospectionStrategy` - Audio introspection implementation
- `VisualIntrospectionStrategy` - Visual introspection implementation

**Usage:**
```python
from core.patterns import AudioIntrospectionStrategy

strategy = AudioIntrospectionStrategy(max_history_size=1000)
s_score = strategy.introspect(genomic_bit)
```

**Benefits:**
- Easy to add new introspection types (tactile, temperature, etc.)
- Consistent interface across all introspection types
- Better testability

### 2. Observer Pattern (`core/patterns/observer.py`)

**Purpose:** Decouples visualization/expression from core processing logic.

**Implementation:**
- `OPUObserver` - Observer interface
- `ObservableOPU` - Mixin class for observer functionality
- `OrthogonalProcessingUnit` now extends `ObservableOPU`

**Usage:**
```python
from core.patterns import OPUObserver

class MyObserver(OPUObserver):
    def on_state_changed(self, state):
        print(f"State: {state}")

opu = OrthogonalProcessingUnit()
opu.attach_observer(MyObserver())
```

**Benefits:**
- Reactive updates instead of polling
- Easy to add new observers (logging, metrics, etc.)
- Loose coupling between components

### 3. Factory Pattern (`core/patterns/sense_factory.py`)

**Purpose:** Centralizes sense creation and management.

**Implementation:**
- `Sense` - Abstract sense interface
- `AudioSense` - Audio sense implementation
- `VisualSense` - Visual sense implementation
- `SenseFactory` - Factory for creating senses

**Usage:**
```python
from core.patterns import SenseFactory

audio_sense = SenseFactory.create_sense("AUDIO_V1")
visual_sense = SenseFactory.create_sense("VIDEO_V1")

# Register new sense type
SenseFactory.register_sense("TEMPERATURE_V1", TemperatureSense)
```

**Benefits:**
- Centralized sense creation
- Easy to add new senses (TOUCH_V1, TEMPERATURE_V1, etc.)
- Type-safe sense handling

### 4. State Pattern (`core/patterns/maturity_state.py`)

**Purpose:** Models maturity levels as states with encapsulated behavior.

**Implementation:**
- `MaturityState` - Abstract state interface
- `ChildState`, `InfantState`, `AdolescentState`, `AdultState`, `ElderState`, `SageState` - Concrete states
- `MaturityContext` - Context that maintains current state

**Usage:**
```python
from core.patterns import MaturityContext

context = MaturityContext()
context.transition_to_level(5)  # Transition to sage
pitch = context.get_pitch()
threshold = context.get_stability_threshold()
```

**Benefits:**
- Clear state transitions
- Encapsulates state-specific behavior
- Easy to add new maturity levels

### 5. Template Method Pattern (`core/patterns/processing_pipeline.py`)

**Purpose:** Defines skeleton of processing algorithm with customizable steps.

**Implementation:**
- `ProcessingPipeline` - Abstract base class with template method
- Hook methods for customization

**Usage:**
```python
from core.patterns import ProcessingPipeline

class MyPipeline(ProcessingPipeline):
    def capture_audio(self):
        # Implement audio capture
        pass
    
    # ... implement other abstract methods

pipeline = MyPipeline()
result = pipeline.process()  # Template method
```

**Benefits:**
- Clear processing flow
- Easy to customize specific steps
- Reusable pipeline structure

### 6. Command Pattern (`core/patterns/commands.py`)

**Purpose:** Encapsulates actions as commands for undo/redo and logging.

**Implementation:**
- `Command` - Abstract command interface
- `StoreMemoryCommand` - Store memory command
- `EvolveCharacterCommand` - Evolve character command
- `ConsolidateMemoryCommand` - Consolidate memory command
- `CommandInvoker` - Command invoker with history

**Usage:**
```python
from core.patterns import StoreMemoryCommand, CommandInvoker

invoker = CommandInvoker()
cmd = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
invoker.execute_command(cmd)

# Undo/redo
invoker.undo()
invoker.redo()
```

**Benefits:**
- Undo/redo capability
- Command queuing
- Logging/auditing of actions
- Macro commands (composite commands)

### 7. Decorator Pattern (`core/patterns/sense_decorator.py`)

**Purpose:** Adds preprocessing/filtering layers to senses.

**Implementation:**
- `SenseDecorator` - Base decorator class
- `NoiseGateDecorator` - Filters low-amplitude noise
- `NormalizationDecorator` - Normalizes input amplitude
- `HighPassFilterDecorator` - Removes DC offset
- `AmplificationDecorator` - Amplifies signal

**Usage:**
```python
from core.patterns import AudioSense, NoiseGateDecorator, NormalizationDecorator

base_sense = AudioSense()
decorated = NormalizationDecorator(
    NoiseGateDecorator(base_sense, threshold=0.1)
)
result = decorated.perceive(raw_input)
```

**Benefits:**
- Composable processing pipelines
- Easy to add filters/transformations
- Maintains single responsibility

### 8. Builder Pattern (`core/patterns/opu_builder.py`)

**Purpose:** Flexible, step-by-step construction of OPU instances.

**Implementation:**
- `OPUBuilder` - Builder class with fluent interface

**Usage:**
```python
from core.patterns import OPUBuilder

opu = (OPUBuilder()
       .with_audio_cortex(max_history=5000)
       .with_visual_cortex(max_history=50)
       .with_genesis_kernel(g_empty_set=1.0)
       .add_sense("AUDIO_V1")
       .add_sense("VIDEO_V1")
       .with_maturity_context(enabled=True)
       .build())
```

**Benefits:**
- Flexible configuration
- Clear initialization
- Immutable configuration

## Integration

All patterns are integrated into the existing codebase:

- **OPU** (`core/opu.py`) - Now extends `ObservableOPU` for observer support
- **Visualizer** (`utils/visualization.py`) - Implements `OPUObserver` interface
- **Backward Compatibility** - All existing code continues to work

## Examples

See `examples/pattern_usage.py` for comprehensive usage examples of all patterns.

## Benefits Summary

1. **Extensibility** - Easy to add new introspection types, senses, maturity levels
2. **Decoupling** - Components communicate through well-defined interfaces
3. **Testability** - Patterns enable better unit testing
4. **Maintainability** - Clear structure and responsibilities
5. **Flexibility** - Easy to customize and configure behavior

## Future Enhancements

- Strategy pattern could be used to replace AudioCortex/VisualCortex directly
- Command pattern could be extended with composite commands
- Decorator pattern could add more preprocessing options
- Builder pattern could support more configuration options

