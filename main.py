"""
The Event Loop: The "Life" of the OPU.
Runs real-time audio processing, cognitive processing, and visualization.
Simulates the "Abstraction Cycle" by speeding up time.
"""

# CRITICAL: Set environment variable BEFORE any tkinter imports on macOS
# This must be done at the very top, before any other imports
import os
import platform
if platform.system() == 'Darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

import numpy as np
import sounddevice as sd
import time
import sys
from datetime import datetime

# Optional cv2 import for visual display
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[OPU] Warning: opencv-python not installed. Visual display disabled.")

from config import (
    SAMPLE_RATE, CHUNK_SIZE, ABSTRACTION_CYCLE_SECONDS,
    BASE_FREQUENCY, STATE_FILE, MATURITY_LEVEL_TIMES, TIME_SCALE_MULTIPLIER,
    AUDIO_SENSE, VIDEO_SENSE
)
from core.genesis import GenesisKernel
from core.mic import perceive
from core.opu import OrthogonalProcessingUnit
from core.expression import AestheticFeedbackLoop, PhonemeAnalyzer
from core.camera import VisualPerception
from core.object_detection import ObjectDetector
from utils.visualization import CognitiveMapVisualizer
from utils.persistence import OPUPersistence

# Optional log window - may not work on all systems (e.g., macOS with Python 3.13+)
# NOTE: On macOS, you MUST use ./run_opu.sh launcher script to set TK_SILENCE_DEPRECATION
# before Python starts. This prevents the NSApplication crash.
try:
    from utils.log_window import OPULogWindow
    LOG_WINDOW_AVAILABLE = True
except Exception:
    OPULogWindow = None
    LOG_WINDOW_AVAILABLE = False


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
        
        # Initialize Visual Perception (NEW: Multi-Modal Integration)
        self.visual_perception = VisualPerception(camera_index=0)
        
        # Initialize Object Detection (NEW: Visual Object Recognition)
        self.object_detector = ObjectDetector(use_dnn=False, confidence_threshold=0.5)
        
        # Initialize persistence
        self.persistence = OPUPersistence(state_file=state_file or STATE_FILE)
        
        # Initialize log window (must be after other initializations)
        # This will redirect stdout/stderr to the log window
        # NOTE: On macOS, the environment variable TK_SILENCE_DEPRECATION must be set
        # BEFORE Python starts (use ./run_opu.sh launcher script)
        self.log_window = None
        if LOG_WINDOW_AVAILABLE:
            # Try to create the log window
            try:
                self.log_window = OPULogWindow(title="OPU Log - Real-time Output")
                self.log_window.start()
                print("[OPU] Log window enabled - all output will appear in the log window")
            except Exception as e:
                # If log window fails, continue without it
                error_msg = str(e)
                if platform.system() == 'Darwin' and 'NSApplication' in error_msg:
                    print("[OPU] ERROR: Log window crashed due to macOS tkinter issue.")
                    print("[OPU] SOLUTION: Use the launcher script instead:")
                    print("[OPU]   ./run_opu.sh")
                    print("[OPU] This sets the required environment variable before Python starts.")
                else:
                    print(f"[OPU] Note: Log window unavailable (using terminal output): {type(e).__name__}: {error_msg}")
                self.log_window = None
        else:
            print("[OPU] Note: Log window module unavailable (using terminal output)")
        
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
            # Use low latency and explicit blocksize to minimize buffer buildup
            # Low latency = smaller internal buffer = less overflow risk
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=CHUNK_SIZE,
                latency='low'  # Low latency = smaller buffer
            )
            self.audio_stream.start()
            self.use_microphone = True
            print("[OPU] Microphone input enabled (low latency mode).")
        except Exception as e1:
            # If that fails, try without latency setting
            try:
                self.audio_stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    blocksize=CHUNK_SIZE,
                    dtype=np.float32
                )
                self.audio_stream.start()
                self.use_microphone = True
                print("[OPU] Microphone input enabled.")
            except Exception as e2:
                # If both fail, fall back to simulated input
                print(f"[OPU] Microphone not available: {e1}")
                print("[OPU] Using simulated audio input.")
                self.use_microphone = False
    
    def get_audio_input(self):
        """
        Get audio input from microphone or generate simulated input.
        Reads ALL available data aggressively to prevent buffer overflow.
        Uses non-blocking read to avoid delays.
        
        FEEDBACK PREVENTION: When OPU is actively speaking (outputting audio),
        we mute the microphone input to prevent acoustic feedback loops.
        The microphone picks up the speaker output, which causes a feedback loop.
        
        Returns:
            numpy array of audio samples
        """
        if self.use_microphone and self.audio_stream is not None:
            # FEEDBACK PREVENTION: If OPU is speaking, mute microphone input
            # This prevents the microphone from picking up the speaker output
            if self.afl.is_active():
                # OPU is speaking - mute microphone to prevent feedback
                # Aggressively drain buffer to prevent overflow
                try:
                    available = self.audio_stream.read_available
                    if available > 0:
                        # Read and discard ALL available data to prevent overflow
                        # Read in large chunks to drain quickly
                        while available > 0:
                            read_size = min(available, CHUNK_SIZE * 8)
                            try:
                                self.audio_stream.read(read_size, blocking=False)
                            except:
                                break
                            available = self.audio_stream.read_available
                            if available <= 0:
                                break
                    return np.zeros(CHUNK_SIZE, dtype=np.float32)
                except Exception:
                    return np.zeros(CHUNK_SIZE, dtype=np.float32)
            
            try:
                # AGGRESSIVE BUFFER DRAINING: Read ALL available data to prevent overflow
                available = self.audio_stream.read_available
                
                # If buffer is getting full, drain it aggressively
                if available > CHUNK_SIZE * 8:  # Buffer getting full
                    # Emergency: drain everything to prevent overflow
                    if not hasattr(self, '_last_overflow_warn') or \
                       time.time() - self._last_overflow_warn > 2.0:
                        print(f"[OPU] Audio buffer draining (available: {available})")
                        self._last_overflow_warn = time.time()
                    
                    # Drain buffer completely
                    while self.audio_stream.read_available > CHUNK_SIZE:
                        try:
                            drain_size = min(self.audio_stream.read_available, CHUNK_SIZE * 8)
                            self.audio_stream.read(drain_size, blocking=False)
                        except:
                            break
                    available = self.audio_stream.read_available
                
                if available > 0:
                    # Read available data (non-blocking)
                    # Read more than we need to keep buffer from filling up
                    read_size = min(available, CHUNK_SIZE * 2)
                    try:
                        data, overflowed = self.audio_stream.read(read_size, blocking=False)
                    except:
                        # Fallback: try smaller read
                        try:
                            data, overflowed = self.audio_stream.read(CHUNK_SIZE, blocking=False)
                        except:
                            # Last resort: return zeros
                            return np.zeros(CHUNK_SIZE, dtype=np.float32)
                    
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
                # Silently fall back to simulated input on errors
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
        The Synesthesia Loop: Multi-Modal Processing
        
        Process one cycle of the OPU pipeline:
        1. Capture audio input (Hear)
        2. Capture visual input (See)
        3. Perceive both streams (scale-invariant)
        4. Introspect on both (calculate s_scores)
        5. Sensory Fusion (S_global = max(S_audio, S_visual))
        6. Apply ethical veto
        7. Store memory
        8. Generate expression (audio + phonemes) - Synesthesia
        9. Update visualization
        """
        # --- 1. AUDIO PERCEPTION ---
        audio_input = self.get_audio_input()
        perception_a = perceive(audio_input)
        genomic_bit_audio = perception_a['genomic_bit']
        s_audio = self.cortex.introspect(genomic_bit_audio)
        
        # --- 2. VISUAL PERCEPTION (RECURSIVE: OPU sees its own thoughts) ---
        # Step 2a: Capture raw frame
        raw_frame = self.visual_perception.capture_frame()
        
        # Step 2b: Generate graphics FIRST (before OPU perceives)
        # This creates the "Cybernetic Frame" - video + OpenCV annotations
        # The OPU will analyze this annotated frame, seeing its own bounding boxes
        # as part of visual reality. This enables recursive feedback loops.
        processed_frame = None
        detections = []
        
        if raw_frame is not None:
            # Run object detection on raw frame
            detections = self.object_detector.detect_objects(raw_frame)
            
            # Draw detections (bounding boxes, labels) onto the frame
            # This annotated frame is what the OPU will "see"
            processed_frame = self.object_detector.draw_detections(raw_frame.copy(), detections)
        
        # Step 2c: OPU analyzes the PROCESSED frame (with graphics)
        # The yellow bounding boxes, text labels, etc. become part of visual entropy
        # This is Recursive Perception: The OPU sees its own thoughts as reality
        visual_vector = self.visual_perception.analyze_frame(processed_frame)
        s_visual, channel_scores = self.cortex.introspect_visual(visual_vector)
        
        # --- 3. SENSORY FUSION (Synesthesia) ---
        # The OPU reacts to the most intense reality, whether light or sound.
        # If you wave a red flag (High S_visual), the OPU will "Scream".
        # If you scream (High S_audio), the OPU will also "Scream".
        # Now: If the OPU draws a red bounding box (High Red Entropy),
        # it will perceive that red as "danger" and potentially react to it.
        s_global = max(s_audio, s_visual)
        
        # --- 4. APPLY ETHICAL VETO ---
        # Apply safety kernel to the fused score
        action_vector = np.array([s_global, genomic_bit_audio])
        safe_action = self.genesis.ethical_veto(action_vector)
        safe_s_score = safe_action[0] if len(safe_action) > 0 else s_global
        
        # --- 5. STORE MEMORY (with sense labels) ---
        # Store memories with sense labels (AUDIO_V1, VIDEO_V1, etc.)
        # This makes the OPU extensible - future senses can be plugged in
        # Each sense is tracked independently, allowing the OPU to learn from multiple modalities
        
        # Store audio memory (primary sense)
        self.cortex.store_memory(genomic_bit_audio, safe_s_score, sense_label=AUDIO_SENSE)
        
        # Store visual memory if significant (multi-modal learning)
        # Visual contributes to s_global, and we store it separately with VIDEO_SENSE label
        if s_visual > 0.5:  # Only store if visual surprise is meaningful
            # Use the maximum channel entropy as the visual genomic bit
            visual_genomic_bit = max(visual_vector) if len(visual_vector) > 0 else 0.0
            self.cortex.store_memory(visual_genomic_bit, s_visual, sense_label=VIDEO_SENSE)
        
        # --- 6. GET CHARACTER STATE ---
        character = self.cortex.get_character_state()
        self.afl.update_pitch(character['base_pitch'])
        
        # --- 7. GENERATE EXPRESSION (Synesthesia) ---
        # The voice pitch is now driven by the Global Score.
        # Visual chaos creates audio response (true synesthesia).
        # Now: If graphics turn red (panic), the OPU sees red entropy,
        # which confirms the danger and amplifies the response (adrenaline loop).
        try:
            self.afl.play_tone(safe_s_score, duration=0.05)
        except:
            pass  # Don't block on audio errors
        
        # --- 8. ANALYZE PHONEMES ---
        # Phonemes still primarily driven by Audio structure
        # (We don't want visual noise to create "false phonemes")
        current_pitch = self.afl.current_frequency
        phoneme = self.phoneme_analyzer.analyze(s_audio, current_pitch)  # Use audio s_score
        if phoneme:
            print(f"[PHONEME] {phoneme} (s_score: {s_audio:.2f}, pitch: {current_pitch:.0f}Hz)")
        
        # --- 9. FINAL DISPLAY (HUD for human user) ---
        # Overlay cognitive state HUD on top of the processed frame
        # Note: We could feed this back into the loop, but that might create
        # infinite feedback. For now, HUD is display-only (not analyzed by OPU).
        if processed_frame is not None:
            self.display_visual_cortex(processed_frame, s_global, s_visual, s_audio, channel_scores, detections)
        
        # --- 10. UPDATE COGNITIVE MAP ---
        state = self.cortex.get_current_state()
        character = self.cortex.get_character_state()
        self.visualizer.update_state(
            state['s_score'],
            state['coherence'],
            state['maturity'],
            character.get('maturity_level', 0)
        )
        # Draw visualization (with error handling for matplotlib threading issues)
        try:
            self.visualizer.draw_cognitive_map()
            self.visualizer.refresh()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Silently ignore visualization errors (e.g., window closed, threading issues)
            pass
        
        return {
            's_score': safe_s_score,
            's_audio': s_audio,
            's_visual': s_visual,
            's_global': s_global,
            'genomic_bit': genomic_bit_audio,
            'phoneme': phoneme,
            'maturity': character['maturity_index'],
            'channel_scores': channel_scores
        }
    
    def display_visual_cortex(self, frame, s_global, s_visual, s_audio, channel_scores, detections=None):
        """
        Displays a dedicated webcam preview window with object detection graphics.
        
        The window shows:
        - Live webcam feed
        - Object detection bounding boxes (faces, etc.) - drawn in yellow
        - OPU cognitive state HUD (R, G, B entropy bars, surprise scores)
        
        Window is resized to 75% of capture resolution for a compact preview.
        
        NOTE: This HUD is for human display only. It is NOT fed back into
        the OPU's perception loop to avoid infinite feedback where drawing
        the score changes the score. The object detections (bounding boxes)
        ARE analyzed by the OPU (see process_cycle), but the HUD stats are not.
        
        Args:
            frame: Processed BGR frame (already has object detection graphics drawn)
            s_global: Global surprise score (max of audio/visual)
            s_visual: Visual surprise score
            s_audio: Audio surprise score
            channel_scores: dict with {'R': float, 'G': float, 'B': float}
            detections: List of detected objects (already drawn on frame, passed for reference)
        """
        if not CV2_AVAILABLE or frame is None:
            return
        
        # Create a HUD overlay (frame already has detections drawn)
        display = frame.copy()
        h, w, _ = display.shape
        
        # Color map for channels
        c_map = {'R': (0, 0, 255), 'G': (0, 255, 0), 'B': (255, 0, 0)}
        y_pos = 30
        
        # 1. Bar Charts for R, G, B Entropy
        # We draw bars proportional to the Surprise in each channel
        for chan in ['R', 'G', 'B']:
            score = channel_scores.get(chan, 0.0)
            bar_len = int(min(score, 5.0) * 50)  # Scale for display (max 250px)
            color = c_map[chan]
            
            # Label
            cv2.putText(display, f"{chan}: {score:.2f}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Bar
            cv2.rectangle(display, (100, y_pos-15), (100+bar_len, y_pos+5), color, -1)
            y_pos += 30

        # 2. Global State Overlay
        # Color changes based on Attention (Green=Calm, Yellow=Interest, Red=Panic)
        if s_global > 3.0:
            status_color = (0, 0, 255)  # RED - ALERT
        elif s_global > 1.5:
            status_color = (0, 255, 255)  # YELLOW - INTEREST
        else:
            status_color = (0, 255, 0)  # GREEN - CALM
        
        cv2.putText(display, f"GLOBAL SURPRISE: {s_global:.2f}", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 3. Audio/Visual Split
        cv2.putText(display, f"A: {s_audio:.2f} | V: {s_visual:.2f}", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Show Window - Small preview with object detection
        # Resize to a smaller, more compact preview window
        preview_scale = 0.75  # Make it 75% of original size for compact preview
        preview_h = int(h * preview_scale)
        preview_w = int(w * preview_scale)
        preview_display = cv2.resize(display, (preview_w, preview_h))
        
        cv2.imshow('OPU WebCam Preview', preview_display)
        cv2.waitKey(1)  # Non-blocking, allows other processing
    
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
                
                # Save state after ANY abstraction cycle trigger (even if no memories)
                # This ensures state is saved regularly regardless of memory state
                # State includes character profile, history, phonemes, etc. - not just memories
                self._save_state()
                
                # Increment day counter only for Level 2 (1 day)
                if level == 2:
                    self.day_counter += 1
    
    def _drain_audio_buffer(self):
        """
        Aggressively drain audio buffer to prevent overflow.
        Called frequently to keep buffer from filling up.
        """
        if not self.use_microphone or self.audio_stream is None:
            return
        
        try:
            available = self.audio_stream.read_available
            # If buffer is getting full, drain it
            if available > CHUNK_SIZE * 4:
                # Drain aggressively
                while available > CHUNK_SIZE:
                    try:
                        drain_size = min(available, CHUNK_SIZE * 8)
                        self.audio_stream.read(drain_size, blocking=False)
                        available = self.audio_stream.read_available
                        if available <= 0:
                            break
                    except:
                        break
        except Exception:
            pass  # Silently ignore errors
    
    def run(self):
        """Main event loop."""
        # Setup
        self.setup_audio_input()
        self.visualizer.show()
        
        try:
            cycle_count = 0
            last_cycle_time = time.time()
            
            while True:
                # Drain audio buffer frequently to prevent overflow
                # Do this BEFORE process_cycle to keep buffer clear
                self._drain_audio_buffer()
                
                # Process one cycle
                result = self.process_cycle()
                cycle_count += 1
                
                # Update log window (if enabled) - call periodically to process messages
                if hasattr(self, 'log_window') and self.log_window is not None:
                    try:
                        self.log_window.update()
                    except Exception:
                        pass  # Silently continue if log window has issues
                
                # Check for abstraction cycle
                self.check_abstraction_cycle()
                
                # Calculate proper timing based on audio chunk size
                # Each chunk should take CHUNK_SIZE / SAMPLE_RATE seconds
                chunk_duration = CHUNK_SIZE / SAMPLE_RATE
                current_time = time.time()
                elapsed = current_time - last_cycle_time
                
                # Don't sleep if we're behind - prioritize clearing audio buffer
                # With visual processing, we need to be more aggressive about timing
                # Only sleep if we're significantly ahead (reduced threshold)
                if elapsed < chunk_duration * 0.3:  # Only sleep if we're way ahead
                    time.sleep(max(0, chunk_duration * 0.2 - elapsed))
                
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
            print("\n[OPU] Shutting down gracefully...")
        except Exception as e:
            # Catch any other exceptions to ensure cleanup
            print(f"\n[OPU] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Cleanup visual perception
        self.visual_perception.cleanup()
        
        # Cleanup object detector
        if hasattr(self, 'object_detector'):
            self.object_detector.cleanup()
        
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        
        # Cleanup log window (restore stdout/stderr)
        if hasattr(self, 'log_window') and self.log_window is not None:
            try:
                self.log_window.stop()
            except Exception:
                pass
        
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

