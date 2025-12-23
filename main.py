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
    BASE_FREQUENCY, STATE_FILE, MATURITY_LEVEL_TIMES, TIME_SCALE_MULTIPLIER
)
from core.genesis import GenesisKernel
from core.perception import perceive
from core.cortex import OrthogonalProcessingUnit
from core.expression import AestheticFeedbackLoop, PhonemeAnalyzer
from utils.visualization import CognitiveMapVisualizer
from utils.persistence import OPUPersistence


class OPUEventLoop:
    """
    Main event loop for the OPU.
    Coordinates all subsystems and runs the abstraction cycle.
    """
    
    def __init__(self, state_file=None):
        # Initialize subsystems
        self.genesis = GenesisKernel()
        self.cortex = OrthogonalProcessingUnit()
        self.afl = AestheticFeedbackLoop(base_pitch=BASE_FREQUENCY)
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.visualizer = CognitiveMapVisualizer()
        
        # Initialize persistence
        self.persistence = OPUPersistence(state_file=state_file or STATE_FILE)
        
        # Timing for abstraction cycles (6 maturity levels)
        self.start_time = time.time()
        self.last_abstraction_times = {level: time.time() for level in range(6)}  # Track each level separately
        self.day_counter = 0
        self.maturity_level_times = MATURITY_LEVEL_TIMES
        
        # Audio input
        self.use_microphone = False
        self.audio_stream = None
        
        # Load saved state if available
        self._load_state()
        
        print("[OPU] Initialized. Starting event loop...")
        print("[OPU] Press Ctrl+C to stop.")
    
    def _load_state(self):
        """Load saved OPU state from disk."""
        success, day_counter = self.persistence.load_state(
            self.cortex,
            self.phoneme_analyzer
        )
        if success:
            self.day_counter = day_counter
            # Update AFL pitch based on loaded character state
            character = self.cortex.get_character_state()
            self.afl.update_pitch(character['base_pitch'])
    
    def _save_state(self):
        """Save current OPU state to disk."""
        self.persistence.save_state(
            self.cortex,
            self.phoneme_analyzer,
            self.day_counter
        )
    
    def setup_audio_input(self):
        """Setup audio input (microphone or simulation)."""
        try:
            # Try to open microphone with larger buffer to prevent overflow
            # Use even larger blocksize and buffer to handle processing delays
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE * 4,  # Much larger buffer
                dtype=np.float32,
                latency='high'  # Higher latency for stability
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
        Reads ALL available data aggressively to prevent buffer overflow.
        
        Returns:
            numpy array of audio samples
        """
        if self.use_microphone and self.audio_stream is not None:
            try:
                # Read ALL available data to prevent overflow
                available = self.audio_stream.read_available
                if available > 0:
                    # Always read everything available to clear the buffer
                    read_size = available
                    data, overflowed = self.audio_stream.read(read_size)
                    
                    if overflowed:
                        # Only print occasionally to avoid spam
                        if not hasattr(self, '_last_overflow_warn') or \
                           time.time() - self._last_overflow_warn > 5.0:
                            print(f"[OPU] Audio buffer overflow detected! (available: {available})")
                            self._last_overflow_warn = time.time()
                    
                    # Take the most recent CHUNK_SIZE samples for processing
                    # This ensures we always process the latest audio
                    data_flat = data.flatten()
                    if len(data_flat) >= CHUNK_SIZE:
                        return data_flat[-CHUNK_SIZE:]
                    else:
                        # Pad if we got less than CHUNK_SIZE
                        padded = np.zeros(CHUNK_SIZE, dtype=np.float32)
                        padded[:len(data_flat)] = data_flat
                        return padded
                else:
                    # No data available, return zeros
                    return np.zeros(CHUNK_SIZE, dtype=np.float32)
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
        
        # Use more chaotic patterns to generate higher variance
        # Random walk for frequencies to create more surprise
        if not hasattr(self, 'freq_state'):
            self.freq_state = {'f1': 440, 'f2': 220, 'f3': 880, 'amp': 1.0}
        
        # Random walk for frequencies (more unpredictable)
        self.freq_state['f1'] += np.random.normal(0, 20)
        self.freq_state['f2'] += np.random.normal(0, 15)
        self.freq_state['f3'] += np.random.normal(0, 30)
        self.freq_state['amp'] = 0.5 + 0.5 * np.sin(current_time * 0.3)
        
        # Clamp frequencies to reasonable ranges
        freq1 = np.clip(self.freq_state['f1'], 200, 1000)
        freq2 = np.clip(self.freq_state['f2'], 100, 500)
        freq3 = np.clip(self.freq_state['f3'], 400, 2000)
        
        # Add more varied noise (sometimes high, sometimes low)
        noise_level = 0.05 + 0.15 * abs(np.sin(current_time * 0.7))
        noise = np.random.normal(0, noise_level, CHUNK_SIZE)
        
        # Combine signals with varying amplitudes
        signal = (
            self.freq_state['amp'] * 0.4 * np.sin(2 * np.pi * freq1 * t) +
            0.3 * np.sin(2 * np.pi * freq2 * t) +
            0.2 * np.sin(2 * np.pi * freq3 * t) +
            noise
        )
        
        # More frequent and varied spikes (high surprise events)
        spike_prob = 0.15  # Increased from 0.05
        if np.random.random() < spike_prob:
            spike_magnitude = np.random.uniform(1.5, 4.0)
            spike_length = np.random.randint(50, 200)
            spike = np.random.normal(0, spike_magnitude, min(spike_length, CHUNK_SIZE))
            spike_start = np.random.randint(0, max(1, CHUNK_SIZE - spike_length))
            signal[spike_start:spike_start+len(spike)] += spike
        
        # Occasionally add silence (also creates surprise)
        if np.random.random() < 0.03:
            silence_start = np.random.randint(0, CHUNK_SIZE // 2)
            silence_length = np.random.randint(CHUNK_SIZE // 4, CHUNK_SIZE // 2)
            signal[silence_start:silence_start+silence_length] *= 0.1
        
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
        
        # 8. Analyze phonemes (use original s_score, not clamped, for pattern recognition)
        current_pitch = self.afl.current_frequency
        phoneme = self.phoneme_analyzer.analyze(s_score, current_pitch)  # Use original s_score
        if phoneme:
            print(f"[PHONEME] {phoneme} (s_score: {s_score:.2f}, pitch: {current_pitch:.0f}Hz)")
        
        # 9. Update visualization
        state = self.cortex.get_current_state()
        character = self.cortex.get_character_state()
        self.visualizer.update_state(
            state['s_score'],
            state['coherence'],
            state['maturity'],
            character.get('maturity_level', 0)
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
        Check if it's time for abstraction cycles at any maturity level.
        Each level has its own time scale (1 minute to 1 year).
        """
        current_time = time.time()
        time_scales = {
            0: "1 minute",
            1: "1 hour",
            2: "1 day",
            3: "1 week",
            4: "1 month",
            5: "1 year"
        }
        
        # Check each maturity level for its abstraction cycle
        for level in range(6):
            elapsed = current_time - self.last_abstraction_times[level]
            level_time = self.maturity_level_times[level]
            
            if elapsed >= level_time:
                self.last_abstraction_times[level] = current_time
                
                # Only consolidate if we have memories at this level
                if len(self.cortex.memory_levels[level]) > 0:
                    self.cortex.consolidate_memory(level)
                    
                    # Print level-specific summary
                    state = self.cortex.get_current_state()
                    character = self.cortex.get_character_state()
                    time_scale = time_scales[level]
                    
                    print(f"\n[MATURITY LEVEL {level} - {time_scale.upper()}] Abstraction Cycle")
                    print(f"  Maturity Level: {character['maturity_level']} | Index: {state['maturity']:.2f}")
                    print(f"  Memory Distribution: " + 
                          " | ".join([f"L{i}={len(self.cortex.memory_levels[i])}" 
                                     for i in range(6)]))
                    
                    # Save state after any abstraction cycle (all levels)
                    # This ensures state is saved regularly
                    self._save_state()
                
                # Increment day counter only for Level 2 (1 day)
                if level == 2:
                    self.day_counter += 1
    
    def run(self):
        """Main event loop."""
        # Setup
        self.setup_audio_input()
        self.visualizer.show()
        
        try:
            cycle_count = 0
            last_cycle_time = time.time()
            
            while True:
                # Process one cycle
                result = self.process_cycle()
                cycle_count += 1
                
                # Check for abstraction cycle
                self.check_abstraction_cycle()
                
                # Calculate proper timing based on audio chunk size
                # Each chunk should take CHUNK_SIZE / SAMPLE_RATE seconds
                chunk_duration = CHUNK_SIZE / SAMPLE_RATE
                current_time = time.time()
                elapsed = current_time - last_cycle_time
                
                # Only sleep if we're processing too fast
                # But don't sleep if we're already behind (to prevent buffer overflow)
                if elapsed < chunk_duration * 0.5:  # Only sleep if we're way ahead
                    time.sleep(max(0, chunk_duration * 0.5 - elapsed))
                
                last_cycle_time = time.time()
                
                # Periodic status with s_score info
                if cycle_count % 100 == 0:
                    state = self.cortex.get_current_state()
                    print(f"[CYCLE {cycle_count}] "
                          f"s_score: {state['s_score']:.2f}, "
                          f"coherence: {state['coherence']:.2f}, "
                          f"maturity: {state['maturity']:.2f}, "
                          f"genomic_bit: {state.get('g_now', 0):.4f}")
                    
                    # Show memory distribution (all 6 levels)
                    mem_dist = {k: len(v) for k, v in self.cortex.memory_levels.items()}
                    print(f"  Memory: " + " | ".join([f"L{i}={mem_dist[i]}" for i in range(6)]))
        
        except KeyboardInterrupt:
            print("\n[OPU] Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources and save state."""
        # Save state before shutdown
        print("[OPU] Saving state...")
        self._save_state()
        
        # Cleanup audio input
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Cleanup audio output
        self.afl.cleanup()
        
        print("[OPU] Cleanup complete.")


def main():
    """Entry point."""
    from config import OPU_VERSION
    
    print("=" * 60)
    print("Orthogonal Processing Unit (OPU)")
    print(f"Version {OPU_VERSION} - MIT License")
    print("Process-Centric AI Architecture")
    print("=" * 60)
    print()
    
    opu = OPUEventLoop()
    opu.run()


if __name__ == "__main__":
    main()

