"""
Example: Using All GOF Design Patterns in OPU.

This example demonstrates how to use all the implemented design patterns:
1. Strategy Pattern - Introspection strategies
2. Observer Pattern - State change notifications
3. Factory Pattern - Sense creation
4. State Pattern - Maturity levels
5. Template Method Pattern - Processing pipeline
6. Command Pattern - Action encapsulation
7. Decorator Pattern - Sense preprocessing
8. Builder Pattern - OPU configuration
"""

import numpy as np
from core.patterns import (
    # Strategy Pattern
    AudioIntrospectionStrategy,
    VisualIntrospectionStrategy,
    # Observer Pattern
    OPUObserver,
    # Factory Pattern
    SenseFactory,
    AudioSense,
    VisualSense,
    # State Pattern
    MaturityContext,
    # Template Method Pattern
    ProcessingPipeline,
    # Command Pattern
    StoreMemoryCommand,
    CommandInvoker,
    # Decorator Pattern
    NoiseGateDecorator,
    NormalizationDecorator,
    # Builder Pattern
    OPUBuilder,
)
from core.opu import OrthogonalProcessingUnit
from core.brain import Brain
from core.genesis import GenesisKernel


# ============================================================================
# 1. STRATEGY PATTERN - Introspection Strategies
# ============================================================================

def example_strategy_pattern():
    """Example: Using introspection strategies."""
    print("\n=== Strategy Pattern Example ===")
    
    # Create strategies
    audio_strategy = AudioIntrospectionStrategy(max_history_size=1000)
    visual_strategy = VisualIntrospectionStrategy(max_history=50)
    
    # Use strategies
    audio_genomic = 0.5
    s_audio = audio_strategy.introspect(audio_genomic)
    print(f"Audio Strategy: s_score = {s_audio:.2f}")
    
    visual_vector = np.array([0.3, 0.4, 0.5])
    s_visual = visual_strategy.introspect(visual_vector)
    print(f"Visual Strategy: s_score = {s_visual:.2f}")


# ============================================================================
# 2. OBSERVER PATTERN - State Change Notifications
# ============================================================================

class LoggingObserver(OPUObserver):
    """Example observer that logs state changes."""
    
    def on_state_changed(self, state):
        """Log state changes."""
        print(f"[OBSERVER] State changed: s_score={state.get('s_score', 0):.2f}, "
              f"maturity={state.get('maturity', 0):.2f}")


def example_observer_pattern():
    """Example: Using observer pattern."""
    print("\n=== Observer Pattern Example ===")
    
    # Create OPU
    opu = OrthogonalProcessingUnit()
    
    # Attach observer
    observer = LoggingObserver()
    opu.attach_observer(observer)
    
    # State changes will notify observer
    opu.introspect(0.5)
    opu.introspect(1.2)


# ============================================================================
# 3. FACTORY PATTERN - Sense Creation
# ============================================================================

def example_factory_pattern():
    """Example: Using sense factory."""
    print("\n=== Factory Pattern Example ===")
    
    # Create senses using factory
    audio_sense = SenseFactory.create_sense("AUDIO_V1")
    visual_sense = SenseFactory.create_sense("VIDEO_V1")
    
    print(f"Audio Sense Label: {audio_sense.get_label()}")
    print(f"Visual Sense Label: {visual_sense.get_label()}")
    
    # Register new sense type
    class TemperatureSense(AudioSense):  # Inherit from AudioSense for simplicity
        def get_label(self):
            return "TEMPERATURE_V1"
    
    SenseFactory.register_sense("TEMPERATURE_V1", TemperatureSense)
    temp_sense = SenseFactory.create_sense("TEMPERATURE_V1")
    print(f"New Sense Label: {temp_sense.get_label()}")


# ============================================================================
# 4. STATE PATTERN - Maturity Levels
# ============================================================================

def example_state_pattern():
    """Example: Using maturity state context."""
    print("\n=== State Pattern Example ===")
    
    context = MaturityContext()
    
    # Start as child
    print(f"Level {context.get_level()}: {context.get_time_scale()}, "
          f"Pitch={context.get_pitch():.0f}Hz, "
          f"Threshold={context.get_stability_threshold():.1f}")
    
    # Transition to sage
    context.transition_to_level(5)
    print(f"Level {context.get_level()}: {context.get_time_scale()}, "
          f"Pitch={context.get_pitch():.0f}Hz, "
          f"Threshold={context.get_stability_threshold():.1f}")


# ============================================================================
# 5. TEMPLATE METHOD PATTERN - Processing Pipeline
# ============================================================================

class ExampleProcessingPipeline(ProcessingPipeline):
    """Example implementation of processing pipeline."""
    
    def __init__(self, opu, genesis):
        self.opu = opu
        self.genesis = genesis
        self.audio_data = None
        self.visual_data = None
    
    def capture_audio(self):
        """Capture audio (simulated)."""
        return np.random.randn(1024).astype(np.float32)
    
    def capture_visual(self):
        """Capture visual (simulated)."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def extract_audio_genomic(self, audio_data):
        """Extract audio genomic bit."""
        from core.mic import perceive
        result = perceive(audio_data)
        return result['genomic_bit']
    
    def extract_visual_genomic(self, visual_data):
        """Extract visual genomic vector."""
        if visual_data is None:
            return np.array([0.0, 0.0, 0.0])
        from core.camera import VisualPerception
        vp = VisualPerception()
        return vp.analyze_frame(visual_data)
    
    def introspect_audio(self, genomic_bit):
        """Introspect audio."""
        return self.opu.introspect(genomic_bit)
    
    def introspect_visual(self, visual_vector):
        """Introspect visual."""
        return self.opu.introspect_visual(visual_vector)
    
    def apply_safety(self, s_score, genomic_bit):
        """Apply safety kernel."""
        action_vector = np.array([s_score, genomic_bit])
        safe_vector = self.genesis.ethical_veto(action_vector)
        return safe_vector[0]  # Return safe s_score
    
    def store_memory(self, audio_genomic, s_score, visual_genomic, s_visual):
        """Store memory."""
        self.opu.store_memory(audio_genomic, s_score, sense_label="AUDIO_V1")
        if s_visual > 0.5:
            visual_genomic_bit = np.max(visual_genomic) if len(visual_genomic) > 0 else 0.0
            self.opu.store_memory(visual_genomic_bit, s_visual, sense_label="VIDEO_V1")
    
    def generate_expression(self, s_score):
        """Generate expression (placeholder)."""
        pass  # Would call AFL here


def example_template_method_pattern():
    """Example: Using template method pattern."""
    print("\n=== Template Method Pattern Example ===")
    
    opu = OrthogonalProcessingUnit()
    genesis = GenesisKernel()
    pipeline = ExampleProcessingPipeline(opu, genesis)
    
    # Process one cycle
    result = pipeline.process()
    print(f"Processing Result: s_audio={result['s_audio']:.2f}, "
          f"s_visual={result['s_visual']:.2f}, "
          f"s_global={result['s_global']:.2f}")


# ============================================================================
# 6. COMMAND PATTERN - Action Encapsulation
# ============================================================================

def example_command_pattern():
    """Example: Using command pattern."""
    print("\n=== Command Pattern Example ===")
    
    brain = Brain()
    invoker = CommandInvoker()
    
    # Create and execute commands
    cmd1 = StoreMemoryCommand(brain, 0.5, 1.2, "AUDIO_V1")
    invoker.execute_command(cmd1)
    print(f"Executed: {cmd1.get_description()}")
    
    # Undo command
    if invoker.undo():
        print("Command undone")
    
    # Redo command
    if invoker.redo():
        print("Command redone")


# ============================================================================
# 7. DECORATOR PATTERN - Sense Preprocessing
# ============================================================================

def example_decorator_pattern():
    """Example: Using decorator pattern for sense preprocessing."""
    print("\n=== Decorator Pattern Example ===")
    
    # Create base sense
    audio_sense = AudioSense()
    
    # Decorate with noise gate and normalization
    decorated_sense = NormalizationDecorator(
        NoiseGateDecorator(audio_sense, threshold=0.1),
        gain=1.0
    )
    
    # Use decorated sense
    raw_input = np.random.randn(1024).astype(np.float32) * 0.05  # Low amplitude
    result = decorated_sense.perceive(raw_input)
    print(f"Decorated Sense Result: genomic_bit={result['genomic_bit']:.4f}")


# ============================================================================
# 8. BUILDER PATTERN - OPU Configuration
# ============================================================================

def example_builder_pattern():
    """Example: Using builder pattern for OPU configuration."""
    print("\n=== Builder Pattern Example ===")
    
    # Build OPU with custom configuration
    opu = (OPUBuilder()
           .with_audio_cortex(max_history=5000)
           .with_visual_cortex(max_history=50)
           .with_genesis_kernel(g_empty_set=1.0)
           .add_sense("AUDIO_V1")
           .add_sense("VIDEO_V1")
           .with_maturity_context(enabled=True)
           .build())
    
    print(f"OPU built with {len(opu.genomic_bits_history)} audio history entries")
    print(f"Maturity context: {hasattr(opu, 'maturity_context')}")


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GOF Design Patterns Usage Examples")
    print("=" * 60)
    
    try:
        example_strategy_pattern()
        example_observer_pattern()
        example_factory_pattern()
        example_state_pattern()
        example_template_method_pattern()
        example_command_pattern()
        example_decorator_pattern()
        example_builder_pattern()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

