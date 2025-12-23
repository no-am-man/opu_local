# GOF Design Pattern Refactoring Opportunities

This document identifies opportunities to apply Gang of Four (GOF) design patterns to improve the OPU architecture.

## Current Architecture Analysis

### Strengths
- Clear separation of concerns (Perception, Cortex, Expression)
- Facade pattern already in use (`OrthogonalProcessingUnit`)
- Modular design with distinct responsibilities

### Refactoring Opportunities

## 1. **Strategy Pattern** - Introspection Strategies

**Current Issue:** `AudioCortex` and `VisualCortex` have similar introspection logic but are separate classes.

**Refactoring:** Create an `IntrospectionStrategy` interface and concrete strategies.

```python
# core/introspection_strategy.py
from abc import ABC, abstractmethod
import numpy as np

class IntrospectionStrategy(ABC):
    """Abstract strategy for calculating surprise scores."""
    
    @abstractmethod
    def introspect(self, genomic_input):
        """Calculate surprise score from genomic input."""
        pass
    
    @abstractmethod
    def get_state(self):
        """Get current introspection state."""
        pass

class AudioIntrospectionStrategy(IntrospectionStrategy):
    """Strategy for audio introspection."""
    # Move AudioCortex logic here
    
class VisualIntrospectionStrategy(IntrospectionStrategy):
    """Strategy for visual introspection."""
    # Move VisualCortex logic here
```

**Benefits:**
- Easy to add new introspection types (e.g., `TactileIntrospectionStrategy`)
- Consistent interface across all introspection types
- Better testability

## 2. **Observer Pattern** - State Change Notifications

**Current Issue:** `CognitiveMapVisualizer` and `AestheticFeedbackLoop` poll state directly.

**Refactoring:** Make them observers of OPU state changes.

```python
# core/observer.py
from abc import ABC, abstractmethod

class OPUObserver(ABC):
    """Observer interface for OPU state changes."""
    
    @abstractmethod
    def on_state_changed(self, state):
        """Called when OPU state changes."""
        pass

# In OrthogonalProcessingUnit:
class OrthogonalProcessingUnit:
    def __init__(self):
        self._observers = []
        # ...
    
    def attach_observer(self, observer: OPUObserver):
        self._observers.append(observer)
    
    def _notify_observers(self, state):
        for observer in self._observers:
            observer.on_state_changed(state)
```

**Benefits:**
- Decouples visualization/expression from core logic
- Easy to add new observers (logging, metrics, etc.)
- Reactive updates instead of polling

## 3. **Factory Pattern** - Sense Creation

**Current Issue:** Sense labels are hardcoded strings, no centralized creation.

**Refactoring:** Create a `SenseFactory` for creating and managing senses.

```python
# core/sense_factory.py
from abc import ABC, abstractmethod
from config import AUDIO_SENSE, VIDEO_SENSE

class Sense(ABC):
    """Abstract sense interface."""
    
    @abstractmethod
    def perceive(self, raw_input):
        """Convert raw input to genomic bit."""
        pass
    
    @abstractmethod
    def get_label(self):
        """Return sense label (e.g., 'AUDIO_V1')."""
        pass

class AudioSense(Sense):
    def perceive(self, raw_input):
        from core.mic import perceive
        return perceive(raw_input)
    
    def get_label(self):
        return AUDIO_SENSE

class VisualSense(Sense):
    def perceive(self, raw_input):
        from core.camera import VisualPerception
        # ...
    
    def get_label(self):
        return VIDEO_SENSE

class SenseFactory:
    """Factory for creating sense instances."""
    
    _senses = {
        AUDIO_SENSE: AudioSense,
        VIDEO_SENSE: VisualSense,
    }
    
    @classmethod
    def create_sense(cls, sense_label):
        sense_class = cls._senses.get(sense_label)
        if sense_class:
            return sense_class()
        raise ValueError(f"Unknown sense: {sense_label}")
    
    @classmethod
    def register_sense(cls, label, sense_class):
        """Register a new sense type."""
        cls._senses[label] = sense_class
```

**Benefits:**
- Centralized sense creation
- Easy to add new senses (TOUCH_V1, TEMPERATURE_V1, etc.)
- Type-safe sense handling

## 4. **State Pattern** - Maturity Levels

**Current Issue:** Maturity logic is scattered with if/else statements.

**Refactoring:** Model maturity levels as states.

```python
# core/maturity_state.py
from abc import ABC, abstractmethod

class MaturityState(ABC):
    """Abstract maturity state."""
    
    @abstractmethod
    def get_pitch_multiplier(self):
        """Return pitch multiplier for this maturity level."""
        pass
    
    @abstractmethod
    def get_stability_threshold(self):
        """Return stability threshold for this level."""
        pass
    
    @abstractmethod
    def get_time_scale(self):
        """Return time scale name."""
        pass

class ChildState(MaturityState):  # Level 0
    def get_pitch_multiplier(self):
        return 1.0  # 440Hz
    
    def get_stability_threshold(self):
        return 3.0
    
    def get_time_scale(self):
        return "1 minute"

class SageState(MaturityState):  # Level 5
    def get_pitch_multiplier(self):
        return 0.25  # 110Hz
    
    def get_stability_threshold(self):
        return 8.0
    
    def get_time_scale(self):
        return "1 year"

class MaturityContext:
    """Context that maintains current maturity state."""
    
    def __init__(self):
        self._state = ChildState()
    
    def transition_to(self, state: MaturityState):
        self._state = state
    
    def get_pitch(self, base_pitch):
        return base_pitch * self._state.get_pitch_multiplier()
```

**Benefits:**
- Clear state transitions
- Easy to add new maturity levels
- Encapsulates state-specific behavior

## 5. **Template Method Pattern** - Processing Pipeline

**Current Issue:** `process_cycle()` has a fixed sequence that's hard to extend.

**Refactoring:** Define template method with hook points.

```python
# core/processing_pipeline.py
from abc import ABC, abstractmethod

class ProcessingPipeline(ABC):
    """Template method for OPU processing cycle."""
    
    def process(self):
        """Template method defining the processing steps."""
        # 1. Perception
        audio_data = self.capture_audio()
        visual_data = self.capture_visual()
        
        # 2. Genomic extraction
        audio_genomic = self.extract_audio_genomic(audio_data)
        visual_genomic = self.extract_visual_genomic(visual_data)
        
        # 3. Introspection
        s_audio = self.introspect_audio(audio_genomic)
        s_visual = self.introspect_visual(visual_genomic)
        
        # 4. Fusion (hook point - can be overridden)
        s_global = self.fuse_scores(s_audio, s_visual)
        
        # 5. Safety
        safe_score = self.apply_safety(s_global)
        
        # 6. Memory (hook point)
        self.store_memory(audio_genomic, safe_score)
        
        # 7. Expression (hook point)
        self.generate_expression(safe_score)
        
        return self.create_result(s_audio, s_visual, s_global)
    
    # Hook methods - can be overridden
    def fuse_scores(self, s_audio, s_visual):
        """Default fusion: max of both scores."""
        return max(s_audio, s_visual)
    
    @abstractmethod
    def capture_audio(self):
        pass
    
    @abstractmethod
    def capture_visual(self):
        pass
    # ... other abstract methods
```

**Benefits:**
- Clear processing flow
- Easy to customize specific steps
- Reusable pipeline structure

## 6. **Command Pattern** - Action Encapsulation

**Current Issue:** Actions (store memory, evolve character) are direct method calls.

**Refactoring:** Encapsulate actions as commands.

```python
# core/commands.py
from abc import ABC, abstractmethod

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

class StoreMemoryCommand(Command):
    def __init__(self, brain, genomic_bit, s_score, sense_label):
        self.brain = brain
        self.genomic_bit = genomic_bit
        self.s_score = s_score
        self.sense_label = sense_label
    
    def execute(self):
        self.brain.store_memory(
            self.genomic_bit,
            self.s_score,
            sense_label=self.sense_label
        )
    
    def undo(self):
        # Remove last memory entry
        pass

class CommandInvoker:
    """Invokes commands and maintains history."""
    
    def __init__(self):
        self.history = []
    
    def execute_command(self, command: Command):
        command.execute()
        self.history.append(command)
```

**Benefits:**
- Undo/redo capability
- Command queuing
- Logging/auditing of actions
- Macro commands (composite commands)

## 7. **Decorator Pattern** - Sense Processing Layers

**Current Issue:** All senses processed the same way, no middleware.

**Refactoring:** Add decorators for preprocessing/filtering.

```python
# core/sense_decorator.py
from abc import ABC, abstractmethod

class SenseDecorator(Sense):
    """Decorator base class."""
    
    def __init__(self, wrapped_sense: Sense):
        self._wrapped = wrapped_sense
    
    def perceive(self, raw_input):
        processed = self.preprocess(raw_input)
        result = self._wrapped.perceive(processed)
        return self.postprocess(result)
    
    def preprocess(self, raw_input):
        """Override to add preprocessing."""
        return raw_input
    
    def postprocess(self, result):
        """Override to add postprocessing."""
        return result

class NoiseGateDecorator(SenseDecorator):
    """Filters out low-amplitude noise."""
    
    def preprocess(self, raw_input):
        threshold = 0.1
        if np.max(np.abs(raw_input)) < threshold:
            return np.zeros_like(raw_input)
        return raw_input

class NormalizationDecorator(SenseDecorator):
    """Normalizes input amplitude."""
    
    def preprocess(self, raw_input):
        max_val = np.max(np.abs(raw_input))
        if max_val > 0:
            return raw_input / max_val
        return raw_input
```

**Benefits:**
- Composable processing pipelines
- Easy to add filters/transformations
- Maintains single responsibility

## 8. **Builder Pattern** - OPU Configuration

**Current Issue:** OPU initialization has many parameters scattered.

**Refactoring:** Use builder for complex initialization.

```python
# core/opu_builder.py
class OPUBuilder:
    """Builder for configuring OPU instances."""
    
    def __init__(self):
        self._brain_config = {}
        self._audio_config = {}
        self._visual_config = {}
        self._senses = []
    
    def with_brain_config(self, **kwargs):
        self._brain_config.update(kwargs)
        return self
    
    def with_audio_cortex(self, max_history=100):
        self._audio_config['max_history'] = max_history
        return self
    
    def add_sense(self, sense_label):
        self._senses.append(sense_label)
        return self
    
    def build(self):
        """Build configured OPU instance."""
        brain = Brain(**self._brain_config)
        audio = AudioCortex(**self._audio_config)
        # ...
        return OrthogonalProcessingUnit(brain, audio, ...)
```

**Benefits:**
- Flexible configuration
- Clear initialization
- Immutable configuration

## Priority Recommendations

### High Priority
1. **Strategy Pattern** for introspection - Makes adding new senses trivial
2. **Observer Pattern** for state changes - Decouples visualization/expression
3. **Factory Pattern** for senses - Centralizes sense creation

### Medium Priority
4. **Template Method** for processing pipeline - Makes pipeline extensible
5. **State Pattern** for maturity - Cleaner than if/else chains

### Low Priority (Nice to Have)
6. **Command Pattern** - Useful for undo/redo, but may be overkill
7. **Decorator Pattern** - Useful for preprocessing, but adds complexity
8. **Builder Pattern** - Nice for complex configs, but current init is simple

## Implementation Order

1. Start with **Strategy Pattern** for introspection (biggest impact)
2. Add **Observer Pattern** to decouple visualization
3. Implement **Factory Pattern** for senses
4. Refactor maturity with **State Pattern**
5. Apply **Template Method** to processing pipeline

This will make the codebase more extensible and maintainable while preserving all existing functionality.

