"""
The Event Loop: The "Life" of the OPU.
Runs real-time audio processing, cognitive processing, and visualization.
Uses Multiprocessing Queue to send real-time graphs to the viewer.
"""

# CRITICAL: Set environment variables BEFORE any imports on macOS
import os
import platform
if platform.system() == 'Darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sys
import argparse
import multiprocessing  # Process isolation
import queue            # For Queue exceptions
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Optional cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[OPU] Warning: opencv-python not installed. Visual display disabled.")

from config import (
    SAMPLE_RATE, CHUNK_SIZE, BASE_FREQUENCY, STATE_FILE, MATURITY_LEVEL_TIMES,
    AUDIO_SENSE, VIDEO_SENSE, USE_COLOR_CONSTANCY,
    VISUAL_SURPRISE_THRESHOLD, AUDIO_TONE_DURATION_SECONDS,
    PREVIEW_SCALE, MAX_BAR_SCORE, BAR_SCALE_FACTOR, BAR_SPACING,
    BAR_LABEL_X, BAR_START_X, BAR_HEIGHT, ALERT_THRESHOLD, INTEREST_THRESHOLD,
    FONT_SCALE_SMALL, FONT_SCALE_MEDIUM, FONT_THICKNESS_THIN, FONT_THICKNESS_THICK,
    TEXT_COLOR_GRAY, STATUS_COLOR_RED, STATUS_COLOR_YELLOW, STATUS_COLOR_GREEN,
    CHANNEL_COLOR_RED, CHANNEL_COLOR_GREEN, CHANNEL_COLOR_BLUE,
    MATURITY_TIME_SCALES, DAY_COUNTER_LEVEL,
    MAIN_DEFAULT_CONFIDENCE_THRESHOLD, MAIN_DEFAULT_SURPRISE_SCORE,
    MAIN_EMPTY_VISUAL_VECTOR
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

# --- HELPER PROCESS ---
def launch_state_viewer_process(state_file_path, image_queue):
    """
    Runs the State Viewer in a standalone process.
    Receives real-time images via the image_queue.
    """
    viewer = None
    try:
        # Import inside process to avoid GIL conflicts
        from utils.state_viewer import OPUStateViewer
        import tkinter as tk
        import signal
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            if viewer:
                viewer.running = False
                try:
                    viewer.root.quit()
                    viewer.root.destroy()
                except Exception:
                    pass
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initialize with the queue
        viewer = OPUStateViewer(state_file=state_file_path, image_queue=image_queue)
        viewer.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[PROCESS] State Viewer crashed: {e}")
    finally:
        if viewer:
            viewer.running = False
            try:
                viewer.root.quit()
                viewer.root.destroy()
            except Exception:
                pass

@dataclass
class VisualHUDParams:
    frame: np.ndarray
    s_global: float
    s_visual: float
    s_audio: float
    channel_scores: Dict[str, float]
    detections: Optional[list] = None

@dataclass
class ChannelBarParams:
    display: np.ndarray
    channel: str
    score: float
    bar_length: int
    color: Tuple[int, int, int]
    y_position: int

class OPUEventLoop:
    def __init__(self, state_file=None, log_file=None, enable_state_viewer=True):
        self.genesis = GenesisKernel()
        self.cortex = OrthogonalProcessingUnit()
        self.afl = AestheticFeedbackLoop(base_pitch=BASE_FREQUENCY)
        self.phoneme_analyzer = PhonemeAnalyzer()
        self.visualizer = CognitiveMapVisualizer()
        self.visual_perception = VisualPerception(camera_index=0, use_color_constancy=USE_COLOR_CONSTANCY)
        self.object_detector = ObjectDetector(use_dnn=False, confidence_threshold=MAIN_DEFAULT_CONFIDENCE_THRESHOLD)
        self.persistence = OPUPersistence(state_file=state_file or STATE_FILE)
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.opencv_preview_enabled = True
        
        # --- 1. SHARED QUEUE (The Bridge) ---
        # This allows sending images from Main Process -> Viewer Process
        self.image_queue = multiprocessing.Queue(maxsize=2)
        
        self.viewer_process = None
        if enable_state_viewer:
            state_file_path = state_file or STATE_FILE
            
            # --- 2. LAUNCH VIEWER PROCESS ---
            self.viewer_process = multiprocessing.Process(
                target=launch_state_viewer_process, 
                args=(state_file_path, self.image_queue), # Pass the queue
                daemon=True 
            )
            self.viewer_process.start()
            print("[OPU] State viewer launched in background process.")
        
        # File Logging (Required for Viewer to see logs)
        self.file_logger = None
        if log_file is None:
            log_file = "opu.log" # Default log file so viewer works
            
        self.file_logger = FileLogger(log_file, chain_to=self.original_stdout)
        sys.stdout = self.file_logger
        sys.stderr = self.file_logger
        print(f"[OPU] Logging active: {log_file}")
        
        self.start_time = time.time()
        self.last_abstraction_times = {level: time.time() for level in range(8)}
        self.day_counter = 0
        self.maturity_level_times = MATURITY_LEVEL_TIMES
        self.audio_handler = AudioInputHandler(self.afl, self.start_time)
        
        self._load_state()
        print("[OPU] Initialized. Starting event loop...")
    
    def _load_state(self):
        success, day_counter, last_abstraction_times = self.persistence.load_state(
            self.cortex, self.phoneme_analyzer
        )
        if success:
            self.day_counter = day_counter
            if last_abstraction_times:
                for level, timestamp in last_abstraction_times.items():
                    if 0 <= level <= 7:
                        self.last_abstraction_times[level] = timestamp
            character = self.cortex.get_character_state()
            self.afl.update_pitch(character['base_pitch'])
    
    def _save_state(self):
        self.persistence.save_state(
            self.cortex,
            self.phoneme_analyzer,
            self.day_counter,
            self.last_abstraction_times
        )
    
    def process_cycle(self):
        audio_result = self._process_audio_perception()
        self._last_audio_result = audio_result
        visual_result = self._process_visual_perception()
        
        fused_score = max(audio_result['surprise'], visual_result['surprise'])
        safe_score = self._apply_ethical_veto(fused_score, audio_result['genomic_bit'])
        
        self._store_memories(audio_result, visual_result, safe_score)
        self._update_expression(safe_score)
        self._analyze_phonemes(audio_result['surprise'])
        self._display_visual_hud(visual_result)
        
        # Update and Send Graph
        self._update_visualization()
        
        self._process_opencv_events()
    
    def _process_audio_perception(self):
        audio_input = self.audio_handler.get_audio_input()
        perception = perceive(audio_input)
        return {'genomic_bit': perception['genomic_bit'], 'surprise': self.cortex.introspect(perception['genomic_bit'])}
    
    def _process_visual_perception(self):
        try:
            raw_frame = self.visual_perception.capture_frame()
            if raw_frame is None: return self._create_empty_visual_result()
            
            detections = self.object_detector.detect_objects(raw_frame)
            processed_frame = self.object_detector.draw_detections(raw_frame.copy(), detections)
            visual_vector = self.visual_perception.analyze_frame(processed_frame)
            surprise, channel_scores = self.cortex.introspect_visual(visual_vector)
            emotion = self._extract_emotion(detections)
            
            return {
                'surprise': surprise, 'vector': visual_vector, 'detections': detections,
                'emotion': emotion, 'processed_frame': processed_frame, 'channel_scores': channel_scores
            }
        except Exception:
            return self._create_empty_visual_result()
    
    def _create_empty_visual_result(self):
        return {
            'surprise': MAIN_DEFAULT_SURPRISE_SCORE, 'vector': np.array(MAIN_EMPTY_VISUAL_VECTOR),
            'detections': [], 'emotion': None, 'processed_frame': None, 'channel_scores': {}
        }
    
    def _extract_emotion(self, detections):
        if not detections: return None
        for d in detections:
            if d.get('label') == 'face' and 'emotion' in d: return d['emotion']
        return None
    
    def _apply_ethical_veto(self, score, bit):
        action = self.genesis.ethical_veto(np.array([score, bit]))
        return action[0] if len(action) > 0 else score
    
    def _store_memories(self, audio, visual, score):
        self.cortex.store_memory(audio['genomic_bit'], score, sense_label=AUDIO_SENSE)
        if visual['surprise'] > VISUAL_SURPRISE_THRESHOLD:
            v_bit = max(visual['vector']) if len(visual['vector']) > 0 else 0
            self.cortex.store_memory(v_bit, visual['surprise'], sense_label=VIDEO_SENSE, emotion=visual['emotion'])
    
    def _update_expression(self, score):
        char = self.cortex.get_character_state()
        self.afl.update_pitch(char['base_pitch'])
        try: self.afl.play_tone(score, duration=AUDIO_TONE_DURATION_SECONDS)
        except: pass
    
    def _analyze_phonemes(self, score):
        phoneme = self.phoneme_analyzer.analyze(score, self.afl.current_frequency)
        if phoneme: print(f"[PHONEME] {phoneme} (s_score: {score:.2f})")
    
    def _update_visualization(self):
        """Update local cognitive map and PUSH TO QUEUE for viewer."""
        state = self.cortex.get_current_state()
        char = self.cortex.get_character_state()
        
        # 1. Update Plot Data
        self.visualizer.update_state(
            state['s_score'], state['coherence'],
            char['maturity_index'], char.get('maturity_level', 0)
        )
        self.visualizer.draw_cognitive_map()
        
        # 2. Render to Image (BGR)
        graph_image_bgr = self.visualizer.render_to_image()
        
        # 3. PUSH TO VIEWER (IPC)
        if graph_image_bgr is not None:
            try:
                # Convert BGR -> RGB for Tkinter
                graph_image_rgb = cv2.cvtColor(graph_image_bgr, cv2.COLOR_BGR2RGB)
                # Send as tuple: ('cognitive_map', image)
                self.image_queue.put_nowait(('cognitive_map', graph_image_rgb))
            except queue.Full:
                pass # Viewer is slow, drop frame
            except Exception:
                pass

        # 4. Display Local OpenCV Window (DISABLED - now shown in State Viewer)
        # self._display_cognitive_map(graph_image_bgr)  # Removed: cognitive map is in State Viewer
    
    def _display_cognitive_map(self, graph_image):
        # DISABLED: Cognitive map is now displayed in State Viewer, not in separate OpenCV window
        pass
    
    def _display_visual_hud(self, visual_result):
        if visual_result.get('processed_frame') is None: return
        audio = self._last_audio_result
        s_glob = max(audio.get('surprise', 0) if audio else 0, visual_result['surprise'])
        self.display_visual_cortex(
            visual_result['processed_frame'], s_glob, visual_result['surprise'],
            audio.get('surprise', 0) if audio else 0,
            visual_result.get('channel_scores', {}), visual_result['detections']
        )
    
    def display_visual_cortex(self, frame, s_glob, s_vis, s_aud, scores, dets):
        try:
            params = VisualHUDParams(frame, s_glob, s_vis, s_aud, scores, dets)
            if not CV2_AVAILABLE or frame is None: return
            display = self._create_hud_overlay(params)
            # Send to State Viewer instead of separate window
            self._send_webcam_to_viewer(display)
        except: pass

    def _create_hud_overlay(self, params):
        display = params.frame.copy()
        h, w = display.shape[:2]
        y = BAR_SPACING
        self._draw_channel_bars(display, params.channel_scores, y)
        self._draw_global_status(display, params.s_global, h)
        return display

    def _draw_channel_bars(self, display, scores, y):
        colors = {'R': CHANNEL_COLOR_RED, 'G': CHANNEL_COLOR_GREEN, 'B': CHANNEL_COLOR_BLUE}
        for ch in ['R', 'G', 'B']:
            score = scores.get(ch, 0)
            length = int(min(score, MAX_BAR_SCORE) * BAR_SCALE_FACTOR)
            color = colors[ch]
            cv2.putText(display, f"{ch}: {score:.2f}", (BAR_LABEL_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SMALL, color, FONT_THICKNESS_THICK)
            cv2.rectangle(display, (BAR_START_X, y-BAR_HEIGHT//2), (BAR_START_X+length, y+BAR_HEIGHT//2), color, -1)
            y += BAR_SPACING

    def _draw_global_status(self, display, score, h):
        color = STATUS_COLOR_RED if score > ALERT_THRESHOLD else (STATUS_COLOR_YELLOW if score > INTEREST_THRESHOLD else STATUS_COLOR_GREEN)
        cv2.putText(display, f"GLOBAL SURPRISE: {score:.2f}", (BAR_LABEL_X, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MEDIUM, color, FONT_THICKNESS_THICK)

    def _show_preview_window(self, display):
        # DISABLED: Webcam preview is now shown in State Viewer, not in separate OpenCV window
        pass
    
    def _send_webcam_to_viewer(self, display):
        """Send annotated webcam frame to State Viewer via queue."""
        if self.image_queue is None:
            return
        try:
            # Convert BGR -> RGB for Tkinter
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            # Send as tuple: ('webcam', image)
            self.image_queue.put_nowait(('webcam', display_rgb))
        except queue.Full:
            pass  # Viewer is slow, drop frame
        except Exception:
            pass

    def _process_opencv_events(self):
        if not self.opencv_preview_enabled: return
        try:
            if cv2.waitKey(1) & 0xFF == ord('q'): raise KeyboardInterrupt
        except: pass

    def _get_last_audio_result(self):
        return getattr(self, '_last_audio_result', None)
    
    def check_abstraction_cycle(self):
        now = time.time()
        for lvl in range(8):
            if now - self.last_abstraction_times[lvl] >= self.maturity_level_times[lvl]:
                self.last_abstraction_times[lvl] = now
                if len(self.cortex.memory_levels[lvl]) > 0:
                    self.cortex.consolidate_memory(lvl)
                    self._print_abstraction_summary(lvl)
                self._save_state()
                if lvl == DAY_COUNTER_LEVEL: self.day_counter += 1

    def _print_abstraction_summary(self, lvl):
        char = self.cortex.get_character_state()
        name = MATURITY_TIME_SCALES[lvl]
        print(f"\n[MATURITY {lvl} - {name.upper()}] Abstraction | Maturity: {char['maturity_level']}")
        print("  Memory: " + " | ".join([f"L{i}={len(self.cortex.memory_levels[i])}" for i in range(8)]))
    
    def run(self):
        self.audio_handler.setup_audio_input()
        print("[OPU] Main loop running...")
        try:
            cycle = 0
            last_t = time.time()
            while True:
                try:
                    self.audio_handler.drain_audio_buffer()
                    self.process_cycle()
                    self.check_abstraction_cycle()
                except Exception as e:
                    print(f"[OPU] Cycle error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Timing
                elapsed = time.time() - last_t
                target = CHUNK_SIZE / SAMPLE_RATE
                if elapsed < target * 0.3: time.sleep(target * 0.2 - elapsed)
                last_t = time.time()
                
                cycle += 1
                if cycle % 100 == 0: self._print_status(cycle)
        except KeyboardInterrupt:
            print("\n[OPU] Shutting down...")
        finally:
            self.cleanup()
    
    def _print_status(self, cycle):
        s = self.cortex.get_current_state()
        print(f"[CYCLE {cycle}] s_score: {s['s_score']:.2f}, maturity: {s['maturity']:.2f}")
    
    def cleanup(self):
        print("[OPU] Saving state...")
        self._save_state()
        
        # Terminate viewer process first
        if self.viewer_process and self.viewer_process.is_alive():
            print("[OPU] Terminating State Viewer...")
            self.viewer_process.terminate()
            self.viewer_process.join(timeout=3.0)
            if self.viewer_process.is_alive():
                print("[OPU] Force killing State Viewer...")
                self.viewer_process.kill()
                self.viewer_process.join(timeout=1.0)
        
        # Cleanup other resources
        self.afl.cleanup()
        self.visual_perception.cleanup()
        if hasattr(self, 'object_detector'): self.object_detector.cleanup()
        if CV2_AVAILABLE: cv2.destroyAllWindows()
        if self.file_logger:
            sys.stdout = self.file_logger.original_stdout
            sys.stderr = self.file_logger.original_stderr
            self.file_logger.close()
        
        print("[OPU] Cleanup complete.")

        # Force exit if we're in the main process
        import os
        os._exit(0)  # Force exit to ensure all threads/processes terminate

def main():
    from config import OPU_VERSION
    
    parser = argparse.ArgumentParser(
        description='Orthogonal Processing Unit (OPU) - Process-Centric AI Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default settings
  python main.py --log-file opu.log       # Enable file logging
  python main.py --no-state-viewer       # Disable GUI state viewer
  python main.py --log-file debug.log --no-state-viewer  # File logging only
        """
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (enables file logging). Default: opu_debug.log'
    )
    parser.add_argument(
        '--no-state-viewer',
        action='store_true',
        help='Disable the GUI state viewer'
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
        enable_state_viewer=not args.no_state_viewer
    )
    opu.run()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
