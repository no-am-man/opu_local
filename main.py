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
import time
import sys
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

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
    AUDIO_SENSE, VIDEO_SENSE, USE_COLOR_CONSTANCY,
    VISUAL_SURPRISE_THRESHOLD, AUDIO_TONE_DURATION_SECONDS,
    DITHERING_NOISE_SIGMA, BUFFER_DRAIN_MULTIPLIER, BUFFER_READ_MULTIPLIER,
    OVERFLOW_WARNING_INTERVAL_SECONDS, BUFFER_DRAIN_WARNING_INTERVAL_SECONDS,
    BUFFER_FULL_THRESHOLD_MULTIPLIER,
    PREVIEW_SCALE, MAX_BAR_SCORE, BAR_SCALE_FACTOR, BAR_SPACING,
    BAR_LABEL_X, BAR_START_X, BAR_HEIGHT, ALERT_THRESHOLD, INTEREST_THRESHOLD,
    FONT_SCALE_SMALL, FONT_SCALE_MEDIUM, FONT_THICKNESS_THIN, FONT_THICKNESS_THICK,
    TEXT_COLOR_GRAY, STATUS_COLOR_RED, STATUS_COLOR_YELLOW, STATUS_COLOR_GREEN,
    CHANNEL_COLOR_RED, CHANNEL_COLOR_GREEN, CHANNEL_COLOR_BLUE,
    SIMULATED_FREQ_BASE_1, SIMULATED_FREQ_BASE_2, SIMULATED_FREQ_BASE_3,
    SIMULATED_FREQ_WALK_STD_1, SIMULATED_FREQ_WALK_STD_2, SIMULATED_FREQ_WALK_STD_3,
    SIMULATED_FREQ_MIN_1, SIMULATED_FREQ_MAX_1, SIMULATED_FREQ_MIN_2, SIMULATED_FREQ_MAX_2,
    SIMULATED_FREQ_MIN_3, SIMULATED_FREQ_MAX_3, SIMULATED_AMP_BASE, SIMULATED_AMP_RANGE,
    SIMULATED_AMP_FREQ, SIMULATED_SIGNAL_AMP_1, SIMULATED_SIGNAL_AMP_2, SIMULATED_SIGNAL_AMP_3,
    SIMULATED_NOISE_BASE, SIMULATED_NOISE_RANGE, SIMULATED_NOISE_FREQ,
    SIMULATED_SPIKE_PROBABILITY, SIMULATED_SPIKE_MAGNITUDE_MIN, SIMULATED_SPIKE_MAGNITUDE_MAX,
    SIMULATED_SPIKE_LENGTH_MIN, SIMULATED_SPIKE_LENGTH_MAX,
    SIMULATED_SILENCE_PROBABILITY, SIMULATED_SILENCE_START_RATIO,
    SIMULATED_SILENCE_LENGTH_MIN_RATIO, SIMULATED_SILENCE_LENGTH_MAX_RATIO,
    SIMULATED_SILENCE_ATTENUATION,
    MATURITY_TIME_SCALES, DAY_COUNTER_LEVEL
)
from core.genesis import GenesisKernel
from core.mic import perceive
from core.opu import OrthogonalProcessingUnit
from core.expression import AestheticFeedbackLoop, PhonemeAnalyzer
from core.camera import VisualPerception
from core.object_detection import ObjectDetector
from core.audio_input_handler import AudioInputHandler
from utils.visualization import CognitiveMapVisualizer
from utils.persistence import OPUPersistence
from utils.file_logger import FileLogger

# Optional log window - may not work on all systems (e.g., macOS with Python 3.13+)
# NOTE: On macOS, you MUST use ./run_opu.sh launcher script to set TK_SILENCE_DEPRECATION
# before Python starts. This prevents the NSApplication crash.
try:
    from utils.log_window import OPULogWindow
    LOG_WINDOW_AVAILABLE = True
except Exception:
    OPULogWindow = None
    LOG_WINDOW_AVAILABLE = False


@dataclass
class VisualHUDParams:
    """Parameters for visual HUD overlay display."""
    frame: np.ndarray
    s_global: float
    s_visual: float
    s_audio: float
    channel_scores: Dict[str, float]
    detections: Optional[list] = None


@dataclass
class ChannelBarParams:
    """Parameters for drawing a single channel bar."""
    display: np.ndarray
    channel: str
    score: float
    bar_length: int
    color: Tuple[int, int, int]
    y_position: int


class OPUEventLoop:
    """
    Main event loop for the OPU.
    Coordinates all subsystems and runs the abstraction cycle.
    """
    
    def __init__(self, state_file=None, log_file=None, enable_log_window=True):
        """
        Initialize the OPU event loop.
        
        Args:
            state_file: Path to state file (defaults to config.STATE_FILE)
            log_file: Path to log file (None to disable file logging)
            enable_log_window: Whether to enable the GUI log window
        """
        self.genesis = GenesisKernel()
        self.cortex = OrthogonalProcessingUnit()
        self.afl = AestheticFeedbackLoop(base_pitch=BASE_FREQUENCY)
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.visualizer = CognitiveMapVisualizer()
        self.visual_perception = VisualPerception(camera_index=0, use_color_constancy=USE_COLOR_CONSTANCY)
        self.object_detector = ObjectDetector(use_dnn=False, confidence_threshold=0.5)
        self.persistence = OPUPersistence(state_file=state_file or STATE_FILE)
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        self.log_window = None
        if enable_log_window and LOG_WINDOW_AVAILABLE:
            try:
                self.log_window = OPULogWindow(title="OPU Log - Real-time Output")
                self.log_window.start()
                print("[OPU] Log window enabled - all output will appear in the log window")
            except Exception as e:
                error_msg = str(e)
                if platform.system() == 'Darwin' and 'NSApplication' in error_msg:
                    print("[OPU] ERROR: Log window crashed due to macOS tkinter issue.")
                    print("[OPU] SOLUTION: Use the launcher script instead:")
                    print("[OPU]   ./run_opu.sh")
                    print("[OPU] This sets the required environment variable before Python starts.")
                else:
                    print(f"[OPU] Note: Log window unavailable (using terminal output): {type(e).__name__}: {error_msg}")
                self.log_window = None
        elif not enable_log_window:
            print("[OPU] Log window disabled (use --log-window to enable)")
        else:
            print("[OPU] Note: Log window module unavailable (using terminal output)")
        
        self.file_logger = None
        if log_file:
            chain_target = sys.stdout if self.log_window else self.original_stdout
            self.file_logger = FileLogger(log_file, chain_to=chain_target)
            if self.file_logger.enabled:
                sys.stdout = self.file_logger
                sys.stderr = self.file_logger
                print(f"[OPU] File logging enabled: {self.file_logger.get_log_path()}")
        
        self.start_time = time.time()
        self.last_abstraction_times = {level: time.time() for level in range(8)}
        self.day_counter = 0
        self.maturity_level_times = MATURITY_LEVEL_TIMES
        self.audio_handler = AudioInputHandler(self.afl, self.start_time)
        
        self._load_state()
        
        print("[OPU] Initialized. Starting event loop...")
        print("[OPU] Press Ctrl+C to stop.")
    
    def _load_state(self):
        """Load saved OPU state from disk."""
        success, day_counter, last_abstraction_times = self.persistence.load_state(
            self.cortex,
            self.phoneme_analyzer
        )
        if success:
            self.day_counter = day_counter
            # Restore abstraction cycle timers if available
            if last_abstraction_times:
                for level, timestamp in last_abstraction_times.items():
                    if 0 <= level <= 7:
                        self.last_abstraction_times[level] = timestamp
            # Update AFL pitch based on loaded character state
            character = self.cortex.get_character_state()
            self.afl.update_pitch(character['base_pitch'])
    
    def _save_state(self):
        """Save current OPU state to disk."""
        self.persistence.save_state(
            self.cortex,
            self.phoneme_analyzer,
            self.day_counter,
            self.last_abstraction_times  # Save abstraction cycle timers
        )
    
    def process_cycle(self):
        """Process one cycle of the OPU multi-modal perception pipeline."""
        audio_result = self._process_audio_perception()
        self._last_audio_result = audio_result
        
        visual_result = self._process_visual_perception()
        
        fused_score = self._fuse_sensory_scores(audio_result['surprise'], visual_result['surprise'])
        safe_score = self._apply_ethical_veto(fused_score, audio_result['genomic_bit'])
        
        self._store_memories(audio_result, visual_result, safe_score)
        self._update_expression(safe_score)
        self._analyze_phonemes(audio_result['surprise'])
        self._display_visual_hud(visual_result)
        self._update_visualization()
    
    def _process_audio_perception(self):
        """Process audio input and calculate surprise score."""
        audio_input = self.audio_handler.get_audio_input()
        perception = perceive(audio_input)
        genomic_bit = perception['genomic_bit']
        surprise = self.cortex.introspect(genomic_bit)
        return {'genomic_bit': genomic_bit, 'surprise': surprise}
    
    def _process_visual_perception(self):
        """Process visual input with recursive perception (OPU sees its own annotations)."""
        raw_frame = self.visual_perception.capture_frame()
        
        if raw_frame is None:
            return self._create_empty_visual_result()
        
        detections = self.object_detector.detect_objects(raw_frame)
        processed_frame = self.object_detector.draw_detections(raw_frame.copy(), detections)
        visual_vector = self.visual_perception.analyze_frame(processed_frame)
        surprise, channel_scores = self.cortex.introspect_visual(visual_vector)
        emotion = self._extract_emotion_from_detections(detections)
        
        return {
            'surprise': surprise,
            'vector': visual_vector,
            'detections': detections,
            'emotion': emotion,
            'processed_frame': processed_frame,
            'channel_scores': channel_scores
        }
    
    def _create_empty_visual_result(self):
        """Create empty visual result when no frame is available."""
        return {
            'surprise': 0.0,
            'vector': np.array([0.0, 0.0, 0.0]),
            'detections': [],
            'emotion': None,
            'processed_frame': None,
            'channel_scores': {}
        }
    
    def _extract_emotion_from_detections(self, detections):
        """Extract emotion from face detections."""
        if not detections:
            return None
        
        for detection in detections:
            if detection.get('label') == 'face' and 'emotion' in detection:
                return detection['emotion']
        return None
    
    def _fuse_sensory_scores(self, audio_surprise, visual_surprise):
        """Fuse audio and visual surprise scores using maximum intensity."""
        return max(audio_surprise, visual_surprise)
    
    def _apply_ethical_veto(self, global_score, genomic_bit):
        """Apply ethical safety kernel to the global surprise score."""
        action_vector = np.array([global_score, genomic_bit])
        safe_action = self.genesis.ethical_veto(action_vector)
        return safe_action[0] if len(safe_action) > 0 else global_score
    
    def _store_memories(self, audio_result, visual_result, safe_score):
        """Store memories from both audio and visual perception."""
        self.cortex.store_memory(
            audio_result['genomic_bit'],
            safe_score,
            sense_label=AUDIO_SENSE
        )
        
        if visual_result['surprise'] > VISUAL_SURPRISE_THRESHOLD:
            visual_genomic_bit = self._extract_visual_genomic_bit(visual_result['vector'])
            self.cortex.store_memory(
                visual_genomic_bit,
                visual_result['surprise'],
                sense_label=VIDEO_SENSE,
                emotion=visual_result['emotion']
            )
    
    def _extract_visual_genomic_bit(self, visual_vector):
        """Extract genomic bit from visual vector (maximum channel entropy)."""
        return max(visual_vector) if len(visual_vector) > 0 else 0.0
    
    def _update_expression(self, safe_score):
        """Update audio expression based on surprise score."""
        character = self.cortex.get_character_state()
        self.afl.update_pitch(character['base_pitch'])
        self._play_audio_tone(safe_score)
    
    def _play_audio_tone(self, surprise_score):
        """Play audio tone for expression, handling errors gracefully."""
        try:
            self.afl.play_tone(surprise_score, duration=AUDIO_TONE_DURATION_SECONDS)
        except Exception:
            pass
    
    def _analyze_phonemes(self, surprise_score):
        """Analyze phonemes from audio surprise score."""
        current_pitch = self.afl.current_frequency
        phoneme = self.phoneme_analyzer.analyze(surprise_score, current_pitch)
        if phoneme:
            print(f"[PHONEME] {phoneme} (s_score: {surprise_score:.2f}, pitch: {current_pitch:.0f}Hz)")
    
    def _update_visualization(self):
        """Update cognitive map visualization with current state."""
        state = self.cortex.get_current_state()
        character = self.cortex.get_character_state()
        
        self.visualizer.update_state(
            state['s_score'],
            state['coherence'],
            character['maturity_index'],
            character.get('maturity_level', 0),
            self.afl.current_frequency
        )
        self._draw_visualization()
    
    def _draw_visualization(self):
        """Draw cognitive map visualization, handling errors gracefully."""
        try:
            self.visualizer.draw_cognitive_map()
            self.visualizer.refresh()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            pass
    
    def _display_visual_hud(self, visual_result):
        """Display visual HUD overlay if frame is available."""
        processed_frame = visual_result.get('processed_frame')
        if processed_frame is None:
            return
        
        audio_result = self._get_last_audio_result()
        global_score = self._calculate_global_score(audio_result, visual_result)
        
        self.display_visual_cortex(
            processed_frame,
            global_score,
            visual_result['surprise'],
            audio_result.get('surprise', 0.0) if audio_result else 0.0,
            visual_result.get('channel_scores', {}),
            visual_result['detections']
        )
    
    def _get_last_audio_result(self):
        """Get the last audio processing result (for HUD display)."""
        if not hasattr(self, '_last_audio_result'):
            return None
        return self._last_audio_result
    
    def _calculate_global_score(self, audio_result, visual_result):
        """Calculate global surprise score from audio and visual."""
        audio_surprise = audio_result.get('surprise', 0.0) if audio_result else 0.0
        visual_surprise = visual_result.get('surprise', 0.0)
        return max(audio_surprise, visual_surprise)
    
    def display_visual_cortex(self, frame, s_global, s_visual, s_audio, channel_scores, detections=None):
        """Display webcam preview with OPU cognitive state HUD overlay."""
        params = VisualHUDParams(
            frame=frame,
            s_global=s_global,
            s_visual=s_visual,
            s_audio=s_audio,
            channel_scores=channel_scores,
            detections=detections
        )
        
        if not self._can_display_visual(params.frame):
            return
        
        display = self._create_hud_overlay(params)
        self._show_preview_window(display)
    
    def _can_display_visual(self, frame):
        """Check if visual display is available and frame is valid."""
        return CV2_AVAILABLE and frame is not None
    
    def _create_hud_overlay(self, params: VisualHUDParams):
        """Create HUD overlay with cognitive state information."""
        display = params.frame.copy()
        height, width, _ = display.shape
        
        y_position = BAR_SPACING
        self._draw_channel_bars(display, params.channel_scores, y_position)
        self._draw_global_status(display, params.s_global, height)
        self._draw_audio_visual_split(display, params.s_audio, params.s_visual, height)
        
        return display
    
    def _draw_channel_bars(self, display, channel_scores, start_y):
        """Draw R, G, B channel entropy bars."""
        channel_colors = {
            'R': CHANNEL_COLOR_RED,
            'G': CHANNEL_COLOR_GREEN,
            'B': CHANNEL_COLOR_BLUE
        }
        
        y_position = start_y
        for channel in ['R', 'G', 'B']:
            score = channel_scores.get(channel, 0.0)
            bar_length = self._calculate_bar_length(score)
            color = channel_colors[channel]
            
            bar_params = ChannelBarParams(
                display=display,
                channel=channel,
                score=score,
                bar_length=bar_length,
                color=color,
                y_position=y_position
            )
            self._draw_channel_bar(bar_params)
            y_position += BAR_SPACING
    
    def _calculate_bar_length(self, score):
        """Calculate bar length in pixels from score."""
        clamped_score = min(score, MAX_BAR_SCORE)
        return int(clamped_score * BAR_SCALE_FACTOR)
    
    def _draw_channel_bar(self, params: ChannelBarParams):
        """Draw a single channel bar with label."""
        label_text = f"{params.channel}: {params.score:.2f}"
        cv2.putText(
            params.display, label_text, (BAR_LABEL_X, params.y_position),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, params.color, FONT_THICKNESS_THICK
        )
        
        bar_top = params.y_position - (BAR_HEIGHT // 2)
        bar_bottom = params.y_position + (BAR_HEIGHT // 2)
        bar_right = BAR_START_X + params.bar_length
        
        cv2.rectangle(
            params.display, (BAR_START_X, bar_top), (bar_right, bar_bottom), params.color, -1
        )
    
    def _draw_global_status(self, display, s_global, height):
        """Draw global surprise status with color-coded alert level."""
        status_color = self._get_status_color(s_global)
        status_text = f"GLOBAL SURPRISE: {s_global:.2f}"
        y_position = height - 20
        
        cv2.putText(
            display, status_text, (BAR_LABEL_X, y_position),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, status_color, FONT_THICKNESS_THICK
        )
    
    def _get_status_color(self, s_global):
        """Get status color based on surprise score threshold."""
        if s_global > ALERT_THRESHOLD:
            return STATUS_COLOR_RED
        elif s_global > INTEREST_THRESHOLD:
            return STATUS_COLOR_YELLOW
        else:
            return STATUS_COLOR_GREEN
    
    def _draw_audio_visual_split(self, display, s_audio, s_visual, height):
        """Draw audio/visual score split display."""
        split_text = f"A: {s_audio:.2f} | V: {s_visual:.2f}"
        y_position = height - 50
        
        cv2.putText(
            display, split_text, (BAR_LABEL_X, y_position),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, TEXT_COLOR_GRAY, FONT_THICKNESS_THIN
        )
    
    def _show_preview_window(self, display):
        """Show preview window with scaled display."""
        height, width, _ = display.shape
        preview_height = int(height * PREVIEW_SCALE)
        preview_width = int(width * PREVIEW_SCALE)
        preview_display = cv2.resize(display, (preview_width, preview_height))
        
        cv2.imshow('OPU WebCam Preview', preview_display)
        cv2.waitKey(1)
    
    def check_abstraction_cycle(self):
        """Check if it's time for abstraction cycles at any maturity level."""
        current_time = time.time()
        
        for level in range(8):
            if self._should_trigger_abstraction(level, current_time):
                self._process_abstraction_cycle(level, current_time)
    
    def _should_trigger_abstraction(self, level, current_time):
        """Check if abstraction cycle should trigger for given level."""
        elapsed = current_time - self.last_abstraction_times[level]
        level_time = self.maturity_level_times[level]
        return elapsed >= level_time
    
    def _process_abstraction_cycle(self, level, current_time):
        """Process abstraction cycle for a specific level."""
        self.last_abstraction_times[level] = current_time
        
        if self._has_memories_at_level(level):
            self._consolidate_and_report(level)
        
        self._save_state()
        self._increment_day_counter_if_needed(level)
    
    def _has_memories_at_level(self, level):
        """Check if there are memories at the given level."""
        return len(self.cortex.memory_levels[level]) > 0
    
    def _consolidate_and_report(self, level):
        """Consolidate memory and print level-specific summary."""
        self.cortex.consolidate_memory(level)
        self._print_abstraction_summary(level)
    
    def _print_abstraction_summary(self, level):
        """Print abstraction cycle summary for a level."""
        state = self.cortex.get_current_state()
        character = self.cortex.get_character_state()
        time_scale = MATURITY_TIME_SCALES[level]
        
        print(f"\n[MATURITY LEVEL {level} - {time_scale.upper()}] Abstraction Cycle")
        print(f"  Maturity Level: {character['maturity_level']} | Index: {state['maturity']:.2f}")
        print(f"  Memory Distribution: " + self._format_memory_distribution())
    
    def _format_memory_distribution(self):
        """Format memory distribution across all levels."""
        return " | ".join([f"L{i}={len(self.cortex.memory_levels[i])}" for i in range(8)])
    
    def _increment_day_counter_if_needed(self, level):
        """Increment day counter if this is the day-level abstraction."""
        if level == DAY_COUNTER_LEVEL:
            self.day_counter += 1
    
    def run(self):
        """Main event loop."""
        # Setup
        self.audio_handler.setup_audio_input()
        self.visualizer.show()
        
        try:
            cycle_count = 0
            last_cycle_time = time.time()
            
            while True:
                self.audio_handler.drain_audio_buffer()
                result = self.process_cycle()
                cycle_count += 1
                
                if hasattr(self, 'log_window') and self.log_window is not None:
                    try:
                        self.log_window.update()
                    except Exception:
                        pass  # Silently continue if log window has issues
                
                self.check_abstraction_cycle()
                
                chunk_duration = CHUNK_SIZE / SAMPLE_RATE
                current_time = time.time()
                elapsed = current_time - last_cycle_time
                
                if elapsed < chunk_duration * 0.3:
                    time.sleep(max(0, chunk_duration * 0.2 - elapsed))
                
                last_cycle_time = time.time()
                
                if cycle_count % 100 == 0:
                    state = self.cortex.get_current_state()
                    print(f"[CYCLE {cycle_count}] "
                          f"s_score: {state['s_score']:.2f}, "
                          f"coherence: {state['coherence']:.2f}, "
                          f"maturity: {state['maturity']:.2f}, "
                          f"genomic_bit: {state.get('g_now', 0):.4f}")
                    
                    mem_dist = {k: len(v) for k, v in self.cortex.memory_levels.items()}
                    print(f"  Memory: " + " | ".join([f"L{i}={mem_dist[i]}" for i in range(8)]))
        
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
        
        if hasattr(self, 'audio_handler') and self.audio_handler.audio_stream is not None:
            self.audio_handler.audio_stream.stop()
            self.audio_handler.audio_stream.close()
        
        self.afl.cleanup()
        self.visual_perception.cleanup()
        if hasattr(self, 'object_detector'):
            self.object_detector.cleanup()
        
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        
        # Cleanup file logger first (restore stdout/stderr to log window or original)
        if self.file_logger:
            sys.stdout = self.file_logger.original_stdout
            sys.stderr = self.file_logger.original_stderr
            self.file_logger.close()
        
        # Cleanup log window (restore to original stdout/stderr)
        if hasattr(self, 'log_window') and self.log_window is not None:
            try:
                self.log_window.stop()
            except Exception:
                pass
        
        # Final restore to original stdout/stderr
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        
        print("[OPU] Cleanup complete.")


def main():
    """Entry point."""
    from config import OPU_VERSION
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Orthogonal Processing Unit (OPU) - Process-Centric AI Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default settings
  python main.py --log-file opu.log       # Enable file logging
  python main.py --no-log-window          # Disable GUI log window
  python main.py --log-file debug.log --no-log-window  # File logging only
        """
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (enables file logging). Default: opu_debug.log'
    )
    parser.add_argument(
        '--no-log-window',
        action='store_true',
        help='Disable the GUI log window'
    )
    parser.add_argument(
        '--state-file',
        type=str,
        default=None,
        help='Path to OPU state file (default: from config.py)'
    )
    
    args = parser.parse_args()
    
    # Default log file if --log-file is specified without a path
    log_file = args.log_file if args.log_file else ('opu_debug.log' if args.log_file is not None else None)
    
    print("=" * 60)
    print("Orthogonal Processing Unit (OPU)")
    print(f"Version {OPU_VERSION} - MIT License")
    print("Process-Centric AI Architecture")
    print("=" * 60)
    print()
    
    opu = OPUEventLoop(
        state_file=args.state_file,
        log_file=log_file,
        enable_log_window=not args.no_log_window
    )
    opu.run()


if __name__ == "__main__":
    main()

