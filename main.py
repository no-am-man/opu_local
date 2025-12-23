"""
The Event Loop: The "Life" of the OPU.
Runs real-time audio processing, cognitive processing, and visualization.
Simulates the "Abstraction Cycle" by speeding up time.
"""

import numpy as np
import sounddevice as sd
import time
import sys
from datetime import datetime

from config import (
    SAMPLE_RATE, CHUNK_SIZE, ABSTRACTION_CYCLE_SECONDS,
    BASE_FREQUENCY
)
from core.genesis import GenesisKernel
from core.perception import perceive
from core.cortex import OrthogonalProcessingUnit
from core.expression import AestheticFeedbackLoop, PhonemeAnalyzer
from utils.visualization import CognitiveMapVisualizer


class OPUEventLoop:
    """
    Main event loop for the OPU.
    Coordinates all subsystems and runs the abstraction cycle.
    """
    
    def __init__(self):
        # Initialize subsystems
        self.genesis = GenesisKernel()
        self.cortex = OrthogonalProcessingUnit()
        self.afl = AestheticFeedbackLoop(base_pitch=BASE_FREQUENCY)
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.visualizer = CognitiveMapVisualizer()
        
        # Timing for abstraction cycle
        self.start_time = time.time()
        self.last_abstraction_time = time.time()
        self.day_counter = 0
        
        # Audio input
        self.use_microphone = False
        self.audio_stream = None
        
        print("[OPU] Initialized. Starting event loop...")
        print("[OPU] Press Ctrl+C to stop.")
    
    def setup_audio_input(self):
        """Setup audio input (microphone or simulation)."""
        try:
            # Try to open microphone
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                dtype=np.float32
            )
            self.audio_stream.start()
            self.use_microphone = True
            print("[OPU] Microphone input enabled.")
        except Exception as e:
            print(f"[OPU] Microphone not available: {e}")
            print("[OPU] Using simulated audio input.")
            self.use_microphone = False
    
    def get_audio_input(self):
        """
        Get audio input from microphone or generate simulated input.
        
        Returns:
            numpy array of audio samples
        """
        if self.use_microphone and self.audio_stream is not None:
            try:
                data, overflowed = self.audio_stream.read(CHUNK_SIZE)
                if overflowed:
                    print("[OPU] Audio buffer overflow!")
                return data.flatten()
            except Exception as e:
                print(f"[OPU] Error reading audio: {e}")
                return self.generate_simulated_input()
        else:
            return self.generate_simulated_input()
    
    def generate_simulated_input(self):
        """
        Generate simulated audio input for testing.
        Creates varying patterns to trigger different surprise levels.
        """
        # Create time vector
        t = np.linspace(0, CHUNK_SIZE / SAMPLE_RATE, CHUNK_SIZE)
        
        # Vary the pattern over time to create interesting dynamics
        current_time = time.time() - self.start_time
        
        # Mix different frequencies with varying amplitudes
        freq1 = 440 + 50 * np.sin(current_time * 0.1)
        freq2 = 220 + 30 * np.cos(current_time * 0.15)
        freq3 = 880 + 100 * np.sin(current_time * 0.05)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, CHUNK_SIZE)
        
        # Combine signals
        signal = (
            0.3 * np.sin(2 * np.pi * freq1 * t) +
            0.2 * np.sin(2 * np.pi * freq2 * t) +
            0.1 * np.sin(2 * np.pi * freq3 * t) +
            noise
        )
        
        # Occasionally add spikes (high surprise events)
        if np.random.random() < 0.05:
            spike = np.random.normal(0, 2.0, min(100, CHUNK_SIZE))
            signal[:len(spike)] += spike
        
        return signal.astype(np.float32)
    
    def process_cycle(self):
        """
        Process one cycle of the OPU pipeline:
        1. Capture audio input
        2. Perceive (scale-invariant)
        3. Introspect (calculate s_score)
        4. Apply ethical veto
        5. Store memory
        6. Generate expression (audio + phonemes)
        7. Update visualization
        """
        # 1. Capture input
        audio_input = self.get_audio_input()
        
        # 2. Perceive (scale-invariant perception)
        perception = perceive(audio_input)
        genomic_bit = perception['genomic_bit']
        
        # 3. Introspect (calculate surprise)
        s_score = self.cortex.introspect(genomic_bit)
        
        # 4. Apply ethical veto to action vector
        # (In this case, the "action" is the expression output)
        action_vector = np.array([s_score, genomic_bit])
        safe_action = self.genesis.ethical_veto(action_vector)
        safe_s_score = safe_action[0] if len(safe_action) > 0 else s_score
        
        # 5. Store memory
        self.cortex.store_memory(genomic_bit, safe_s_score)
        
        # 6. Get character state for expression
        character = self.cortex.get_character_state()
        self.afl.update_pitch(character['base_pitch'])
        
        # 7. Generate expression (audio feedback)
        # Play tone asynchronously (non-blocking)
        try:
            self.afl.play_tone(safe_s_score, duration=0.05)
        except:
            pass  # Don't block on audio errors
        
        # 8. Analyze phonemes
        current_pitch = self.afl.current_frequency
        phoneme = self.phoneme_analyzer.analyze(safe_s_score, current_pitch)
        if phoneme:
            print(f"[PHONEME] {phoneme} (s_score: {safe_s_score:.2f}, pitch: {current_pitch:.0f}Hz)")
        
        # 9. Update visualization
        state = self.cortex.get_current_state()
        self.visualizer.update_state(
            state['s_score'],
            state['coherence'],
            state['maturity']
        )
        self.visualizer.draw_cognitive_map()
        self.visualizer.refresh()
        
        return {
            's_score': safe_s_score,
            'genomic_bit': genomic_bit,
            'phoneme': phoneme,
            'maturity': character['maturity_index']
        }
    
    def check_abstraction_cycle(self):
        """
        Check if it's time for an abstraction cycle (simulated "Day").
        Every ABSTRACTION_CYCLE_SECONDS = 1 "Day" of maturity.
        """
        current_time = time.time()
        elapsed = current_time - self.last_abstraction_time
        
        if elapsed >= ABSTRACTION_CYCLE_SECONDS:
            self.day_counter += 1
            self.last_abstraction_time = current_time
            
            # Trigger memory consolidation at all levels
            for level in [2, 3]:
                if len(self.cortex.memory_levels[level]) > 0:
                    self.cortex.consolidate_memory(level)
            
            # Print day summary
            state = self.cortex.get_current_state()
            phoneme_stats = self.phoneme_analyzer.get_phoneme_statistics()
            
            print(f"\n[DAY {self.day_counter}] Abstraction Cycle Complete")
            print(f"  Maturity: {state['maturity']:.2f}")
            print(f"  Phonemes Learned: {phoneme_stats['total']}")
            print(f"    Vowels: {phoneme_stats['vowels']}, "
                  f"Fricatives: {phoneme_stats['fricatives']}, "
                  f"Plosives: {phoneme_stats['plosives']}")
            print(f"  Memory Levels: "
                  f"L0={len(self.cortex.memory_levels[0])}, "
                  f"L1={len(self.cortex.memory_levels[1])}, "
                  f"L2={len(self.cortex.memory_levels[2])}, "
                  f"L3={len(self.cortex.memory_levels[3])}\n")
    
    def run(self):
        """Main event loop."""
        # Setup
        self.setup_audio_input()
        self.visualizer.show()
        
        try:
            cycle_count = 0
            while True:
                # Process one cycle
                result = self.process_cycle()
                cycle_count += 1
                
                # Check for abstraction cycle
                self.check_abstraction_cycle()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
                # Periodic status
                if cycle_count % 100 == 0:
                    state = self.cortex.get_current_state()
                    print(f"[CYCLE {cycle_count}] "
                          f"s_score: {state['s_score']:.2f}, "
                          f"coherence: {state['coherence']:.2f}, "
                          f"maturity: {state['maturity']:.2f}")
        
        except KeyboardInterrupt:
            print("\n[OPU] Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
        print("[OPU] Cleanup complete.")


def main():
    """Entry point."""
    print("=" * 60)
    print("Orthogonal Processing Unit (OPU)")
    print("Process-Centric AI Architecture")
    print("=" * 60)
    print()
    
    opu = OPUEventLoop()
    opu.run()


if __name__ == "__main__":
    main()

