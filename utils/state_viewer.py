"""
OPU State Viewer: Visual representation of opu_state.json
Displays the OPU's current state in a beautiful, informative GUI.
"""

# CRITICAL: Set environment variable BEFORE importing tkinter on macOS
import os
import platform
if platform.system() == 'Darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
from pathlib import Path
from datetime import datetime
import threading
import time
import sys
import queue
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple

# PIL for image display
try:
    from PIL import Image, ImageTk
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageTk = None

# OpenCV for image conversion (BGR to RGB)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from config import (
    STATE_VIEWER_DEFAULT_MATURITY_INDEX, STATE_VIEWER_DEFAULT_PITCH,
    STATE_VIEWER_DEFAULT_STABILITY_THRESHOLD, STATE_VIEWER_DEFAULT_S_SCORE,
    STATE_VIEWER_DEFAULT_COHERENCE, STATE_VIEWER_DEFAULT_G_NOW,
    STATE_VIEWER_DEFAULT_CONFIDENCE, STATE_VIEWER_DEFAULT_SPEECH_THRESHOLD,
    STATE_VIEWER_MEMORY_LEVELS_COUNT
)


# Constants
@dataclass
class ViewerConfig:
    """Configuration constants for the state viewer."""
    WINDOW_WIDTH = 2000  # Wider for 3 columns with larger cognitive map
    WINDOW_HEIGHT = 800  # Taller for better cognitive map visibility
    UPDATE_INTERVAL = 0.5
    DEFAULT_STATE_FILE = "opu_state.json"
    LOG_PANEL_WIDTH = 500
    COGNITIVE_MAP_PANEL_WIDTH = 700  # Wider for cognitive map column (increased from 500)
    WEBCAM_PREVIEW_HEIGHT = 200  # Fixed height for webcam preview section
    
    # Colors
    BG_DARK = '#1e1e1e'
    BG_SECTION = '#2d2d2d'
    TEXT_WHITE = '#ffffff'
    TEXT_GRAY = '#cccccc'
    TEXT_LIGHT_GRAY = '#888888'
    
    # Value colors
    COLOR_MATURITY = "#4CAF50"
    COLOR_LEVEL = "#2196F3"
    COLOR_PITCH = "#FF9800"
    COLOR_STABILITY = "#9C27B0"
    COLOR_MEMORY = "#607D8B"
    COLOR_S_SCORE = "#F44336"
    COLOR_COHERENCE = "#00BCD4"
    COLOR_G_NOW = "#E91E63"
    COLOR_EMOTION_COUNT = "#FF5722"
    COLOR_EMOTION_DOMINANT = "#795548"
    COLOR_EMOTION_CONFIDENCE = "#9E9E9E"
    COLOR_PHONEME_COUNT = "#3F51B5"
    COLOR_PHONEME_THRESHOLD = "#009688"
    
    # Fonts
    FONT_TITLE = ('Arial', 24, 'bold')
    FONT_SECTION = ('Arial', 16, 'bold')
    FONT_LABEL = ('Arial', 12)
    FONT_VALUE = ('Arial', 13, 'bold')
    
    # Layout
    LABEL_WIDTH = 30
    PADX = 12
    PADY = 8
    PADY_ROW = 4


@dataclass
class TimeScale:
    """Time scale mapping for maturity levels."""
    SCALES = {
        0: "1s", 1: "1m", 2: "1h", 3: "1d",
        4: "1w", 5: "1mo", 6: "1y", 7: "10y"
    }
    
    @classmethod
    def get_scale(cls, level: int) -> str:
        """Get time scale string for a maturity level."""
        return cls.SCALES.get(level, "Unknown")


class OPUStateViewer:
    """
    Visual representation of OPU state from opu_state.json.
    Updates in real-time to show current cognitive state.
    """
    
    def __init__(self, state_file: str = ViewerConfig.DEFAULT_STATE_FILE, 
                 image_queue=None,
                 update_interval: float = ViewerConfig.UPDATE_INTERVAL):
        """
        Initialize the state viewer.
        
        Args:
            state_file: Path to opu_state.json
            image_queue: multiprocessing.Queue for real-time cognitive map images
            update_interval: How often to refresh (seconds)
        """
        self.state_file = Path(state_file)
        self.image_queue = image_queue  # Store the queue
        self.log_file = Path("opu.log")  # Log file to tail
        self.update_interval = update_interval
        self.running = True
        self.config = ViewerConfig()
        
        # Log capture setup (for file tailing)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        self._create_window()
        self._create_split_layout()
        self._create_scrollable_container()  # Left: State
        self._create_cognitive_map_panel()    # Middle: Cognitive Map
        self._create_log_panel()              # Right: Log
        self._create_all_sections()
        self._start_update_thread()
        self._start_log_tail_thread()  # NEW: Tail the log file
        
        # Start image polling if queue is available
        if self.image_queue:
            self._start_image_poll_thread()
    
    def _create_window(self):
        """Create and configure the main window."""
        self.root = tk.Tk()
        self.root.title("OPU State Viewer")
        
        # Center the window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate center coordinates
        x = (screen_width - self.config.WINDOW_WIDTH) // 2
        y = (screen_height - self.config.WINDOW_HEIGHT) // 2
        
        # Set geometry with offset
        self.root.geometry(f"{self.config.WINDOW_WIDTH}x{self.config.WINDOW_HEIGHT}+{x}+{y}")
        
        self.root.configure(bg=self.config.BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Bring window to front (macOS)
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(lambda: self.root.attributes('-topmost', False))
    
    def _create_split_layout(self):
        """Create split paned window for three columns: State | Cognitive Map | Log."""
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_scrollable_container(self):
        """Create scrollable container for state content (Left column)."""
        # Left panel for state
        left_frame = tk.Frame(self.paned, bg=self.config.BG_DARK)
        self.paned.add(left_frame, weight=1)  # Reduced from 2 to give more space to cognitive map
        
        canvas = tk.Canvas(left_frame, bg=self.config.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.config.BG_DARK)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.container = scrollable_frame
    
    def _create_cognitive_map_panel(self):
        """Create cognitive map panel (Middle column) with webcam preview above."""
        # Middle panel for webcam preview and cognitive map
        map_frame = tk.Frame(self.paned, bg=self.config.BG_DARK)
        self.paned.add(map_frame, weight=2)  # Increased from 1 to make cognitive map bigger
        
        # --- WEBCAM PREVIEW SECTION (Top) ---
        webcam_header = tk.Label(
            map_frame,
            text="WebCam Preview (Real-Time)",
            font=self.config.FONT_SECTION,
            fg=self.config.TEXT_WHITE,
            bg=self.config.BG_SECTION,
            anchor='w'
        )
        webcam_header.pack(fill=tk.X, padx=5, pady=5)
        
        webcam_container = tk.Frame(map_frame, bg=self.config.BG_DARK, height=self.config.WEBCAM_PREVIEW_HEIGHT)
        webcam_container.pack(fill=tk.X, expand=False, padx=5, pady=5)
        webcam_container.pack_propagate(False)  # Prevent container from resizing
        
        # Create label for webcam preview image
        self.webcam_label = tk.Label(
            webcam_container,
            text="Waiting for camera...",
            bg=self.config.BG_DARK,
            fg=self.config.TEXT_GRAY,
            font=self.config.FONT_LABEL
        )
        self.webcam_label.pack(fill=tk.BOTH, expand=True)
        
        # --- COGNITIVE MAP SECTION (Bottom) ---
        map_header = tk.Label(
            map_frame,
            text="Cognitive Map (Real-Time)",
            font=self.config.FONT_SECTION,
            fg=self.config.TEXT_WHITE,
            bg=self.config.BG_SECTION,
            anchor='w'
        )
        map_header.pack(fill=tk.X, padx=5, pady=5)
        
        # Image container with scrollable frame
        map_container = tk.Frame(map_frame, bg=self.config.BG_DARK)
        map_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create label for cognitive map image
        self.map_label = tk.Label(
            map_container,
            text="Waiting for Visual Cortex...",
            bg=self.config.BG_DARK,
            fg=self.config.TEXT_GRAY,
            font=self.config.FONT_LABEL
        )
        self.map_label.pack(fill=tk.BOTH, expand=True)
    
    def _create_log_panel(self):
        """Create log panel on the right side."""
        # Right panel for logs
        log_frame = tk.Frame(self.paned, bg=self.config.BG_DARK)
        self.paned.add(log_frame, weight=1)
        
        # Log header
        log_header = tk.Label(
            log_frame,
            text="OPU Log",
            font=self.config.FONT_SECTION,
            fg=self.config.TEXT_WHITE,
            bg=self.config.BG_SECTION,
            anchor='w'
        )
        log_header.pack(fill=tk.X, padx=5, pady=5)
        
        # Log text widget with scrollbar
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='#ffffff',
            selectbackground='#264f78',
            font=('Courier', 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure text tags for styling
        self.log_text.tag_config('timestamp', foreground='#888888')
        self.log_text.tag_config('info', foreground='#4ec9b0')
        self.log_text.tag_config('error', foreground='#f48771')
        self.log_text.tag_config('warning', foreground='#dcdcaa')
        self.log_text.tag_config('normal', foreground='#d4d4d4')
        
        # Add initial message
        self._append_log("=" * 60 + "\n", 'normal')
        self._append_log("OPU Log Viewer\n", 'info')
        self._append_log("=" * 60 + "\n", 'normal')
        self._append_log(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n", 'timestamp')
        
        # Note: stdout/stderr redirection is handled by main.py after initialization
        # to allow proper chaining with file_logger if it exists
    
    def _create_all_sections(self):
        """Create all display sections (in State column, left side)."""
        self._create_header()
        self._create_character_section()
        # Note: Cognitive map is now in its own column (middle), not here
        self._create_memory_section()
        self._create_current_state_section()
        self._create_emotion_section()
        self._create_phoneme_section()
    
    def _create_header(self):
        """Create header section with title and last update time."""
        header_frame = tk.Frame(self.container, bg=self.config.BG_DARK, pady=self.config.PADY * 2)
        header_frame.pack(fill=tk.X, padx=self.config.PADX)
        
        title = tk.Label(
            header_frame,
            text="OPU State Visualization",
            font=self.config.FONT_TITLE,
            fg=self.config.TEXT_WHITE,
            bg=self.config.BG_DARK
        )
        title.pack()
        
        self.last_update_label = tk.Label(
            header_frame,
            text="Loading...",
            font=self.config.FONT_LABEL,
            fg=self.config.TEXT_LIGHT_GRAY,
            bg=self.config.BG_DARK
        )
        self.last_update_label.pack()
    
    def _create_character_section(self):
        """Create character profile section."""
        section = self._create_section("Character Profile")
        
        self.maturity_frame = self._create_info_row(
            section, "Maturity Index", "0.000", self.config.COLOR_MATURITY
        )
        self.maturity_level_label = self._create_info_row(
            section, "Maturity Level", "L0 (1 second)", self.config.COLOR_LEVEL
        )
        self.pitch_label = self._create_info_row(
            section, "Voice Pitch", "440.0 Hz", self.config.COLOR_PITCH
        )
        self.stability_label = self._create_info_row(
            section, "Stability Threshold", "3.0", self.config.COLOR_STABILITY
        )
    
    def _create_memory_section(self):
        """Create memory levels section."""
        section = self._create_section("Memory Levels (8-Layer Fractal)")
        
        self.memory_labels = {}
        for level in range(STATE_VIEWER_MEMORY_LEVELS_COUNT):
            time_scale = TimeScale.get_scale(level)
            label = self._create_info_row(
                section, f"Level {level} ({time_scale})", "0 items", self.config.COLOR_MEMORY
            )
            self.memory_labels[level] = label
    
    def _create_current_state_section(self):
        """Create current cognitive state section."""
        section = self._create_section("Current Cognitive State")
        
        self.s_score_label = self._create_info_row(
            section, "Surprise Score (s_score)", "0.000", self.config.COLOR_S_SCORE
        )
        self.coherence_label = self._create_info_row(
            section, "Coherence", "0.000", self.config.COLOR_COHERENCE
        )
        self.g_now_label = self._create_info_row(
            section, "Genomic Bit (g_now)", "0.000", self.config.COLOR_G_NOW
        )
    
    def _create_emotion_section(self):
        """Create emotion history section."""
        section = self._create_section("Emotion History")
        
        self.emotion_count_label = self._create_info_row(
            section, "Total Emotions Detected", "0", self.config.COLOR_EMOTION_COUNT
        )
        self.dominant_emotion_label = self._create_info_row(
            section, "Dominant Emotion", "None", self.config.COLOR_EMOTION_DOMINANT
        )
        self.avg_confidence_label = self._create_info_row(
            section, "Average Confidence", "0.000", self.config.COLOR_EMOTION_CONFIDENCE
        )
    
    def _create_phoneme_section(self):
        """Create phoneme section."""
        section = self._create_section("Phoneme Analysis")
        
        self.phoneme_count_label = self._create_info_row(
            section, "Phonemes Detected", "0", self.config.COLOR_PHONEME_COUNT
        )
        self.speech_threshold_label = self._create_info_row(
            section, "Speech Threshold", "0.000", self.config.COLOR_PHONEME_THRESHOLD
        )
    
    # Note: _create_cognitive_map_section() removed - cognitive map is now in its own column (middle)
    
    def _create_section(self, title: str) -> tk.Frame:
        """Create a section container with title."""
        section_frame = tk.Frame(
            self.container, 
            bg=self.config.BG_SECTION, 
            relief=tk.RAISED, 
            bd=1, 
            pady=self.config.PADY
        )
        section_frame.pack(fill=tk.X, padx=self.config.PADX, pady=self.config.PADY)
        
        title_label = tk.Label(
            section_frame,
            text=title,
            font=self.config.FONT_SECTION,
            fg=self.config.TEXT_WHITE,
            bg=self.config.BG_SECTION,
            anchor='w'
        )
        title_label.pack(fill=tk.X, padx=self.config.PADX, pady=self.config.PADY)
        
        content_frame = tk.Frame(section_frame, bg=self.config.BG_SECTION)
        content_frame.pack(fill=tk.X, padx=self.config.PADX, pady=self.config.PADY)
        
        return content_frame
    
    def _create_info_row(self, parent: tk.Frame, label_text: str, 
                        value_text: str, value_color: str) -> tk.Label:
        """Create a row with label and value."""
        row = tk.Frame(parent, bg=self.config.BG_SECTION)
        row.pack(fill=tk.X, pady=self.config.PADY_ROW)
        
        label = tk.Label(
            row,
            text=f"{label_text}:",
            font=self.config.FONT_LABEL,
            fg=self.config.TEXT_GRAY,
            bg=self.config.BG_SECTION,
            anchor='w',
            width=self.config.LABEL_WIDTH
        )
        label.pack(side=tk.LEFT)
        
        value = tk.Label(
            row,
            text=value_text,
            font=self.config.FONT_VALUE,
            fg=value_color,
            bg=self.config.BG_SECTION,
            anchor='w'
        )
        value.pack(side=tk.LEFT, padx=self.config.PADX)
        
        return value
    
    def _load_state(self) -> Optional[Dict[str, Any]]:
        """Load state from JSON file."""
        if not self.state_file.exists():
            return None
        
        return self._read_state_file()
    
    def _read_state_file(self) -> Optional[Dict[str, Any]]:
        """Read and parse state file."""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _update_display(self, state: Dict[str, Any]):
        """Update the display with current state (must be called from main thread)."""
        if not self._can_update_display(state):
            return
        
        try:
            cortex = self._extract_cortex(state)
            self._update_all_sections(cortex, state)
            # Note: Cognitive map is updated via image polling thread, not here
        except Exception:
            # Silently handle errors during display update (window might be closing)
            pass
    
    def _can_update_display(self, state: Dict[str, Any]) -> bool:
        """Check if display can be updated."""
        return state is not None and self.running
    
    def _extract_cortex(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cortex data from state."""
        return state.get('cortex', {})
    
    def _update_all_sections(self, cortex: Dict[str, Any], state: Dict[str, Any]):
        """Update all display sections."""
        self._update_character_profile(cortex.get('character_profile', {}))
        self._update_memory_levels(cortex.get('memory_levels', {}))
        self._update_current_state(cortex.get('current_state', {}))
        self._update_emotion_history(cortex.get('emotion_history', []))
        self._update_phonemes(state.get('phonemes', {}))
        self._update_timestamp()
    
    def _update_character_profile(self, character: Dict[str, Any]):
        """Update character profile section."""
        maturity = character.get('maturity_index', STATE_VIEWER_DEFAULT_MATURITY_INDEX)
        self.maturity_frame.config(text=f"{maturity:.3f}")
        
        maturity_level = character.get('maturity_level', 0)
        time_scale = TimeScale.get_scale(maturity_level)
        self.maturity_level_label.config(text=f"L{maturity_level} ({time_scale})")
        
        pitch = character.get('base_pitch', STATE_VIEWER_DEFAULT_PITCH)
        self.pitch_label.config(text=f"{pitch:.1f} Hz")
        
        stability = character.get('stability_threshold', STATE_VIEWER_DEFAULT_STABILITY_THRESHOLD)
        self.stability_label.config(text=f"{stability:.1f}")
    
    def _update_memory_levels(self, memory_levels: Dict[str, List]):
        """Update memory levels section."""
        for level in range(STATE_VIEWER_MEMORY_LEVELS_COUNT):
            level_data = memory_levels.get(str(level), [])
            count = self._count_memory_items(level_data)
            self.memory_labels[level].config(text=f"{count} items")
    
    def _count_memory_items(self, level_data: Any) -> int:
        """Count memory items in level data."""
        return len(level_data) if isinstance(level_data, list) else 0
    
    def _update_current_state(self, current_state: Dict[str, float]):
        """Update current cognitive state section."""
        s_score = current_state.get('s_score', STATE_VIEWER_DEFAULT_S_SCORE)
        self.s_score_label.config(text=f"{s_score:.3f}")
        
        coherence = current_state.get('coherence', STATE_VIEWER_DEFAULT_COHERENCE)
        self.coherence_label.config(text=f"{coherence:.3f}")
        
        g_now = current_state.get('g_now', STATE_VIEWER_DEFAULT_G_NOW)
        self.g_now_label.config(text=f"{g_now:.3f}")
    
    def _update_emotion_history(self, emotion_history: List[Dict[str, Any]]):
        """Update emotion history section."""
        emotion_count = len(emotion_history)
        self.emotion_count_label.config(text=f"{emotion_count}")
        
        if self._has_no_emotions(emotion_history):
            self._set_empty_emotion_state()
            return
        
        emotion_stats = self._calculate_emotion_statistics(emotion_history)
        self._set_emotion_state_from_stats(emotion_stats)
    
    def _has_no_emotions(self, emotion_history: List[Dict[str, Any]]) -> bool:
        """Check if emotion history is empty."""
        return not emotion_history
    
    def _set_emotion_state_from_stats(self, emotion_stats: Optional[Dict[str, Any]]):
        """Set emotion state from statistics or empty state if None."""
        if emotion_stats:
            self._set_emotion_state(emotion_stats)
        else:
            self._set_empty_emotion_state()
    
    def _calculate_emotion_statistics(self, emotion_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate emotion statistics from history."""
        emotion_counts = self._count_emotions(emotion_history)
        if not emotion_counts:
            return None
        
        total_confidence = self._calculate_total_confidence(emotion_history)
        dominant = self._find_dominant_emotion(emotion_counts)
        avg_confidence = self._calculate_average_confidence(total_confidence, len(emotion_history))
        
        return {
            'dominant_name': dominant[0],
            'dominant_count': dominant[1],
            'average_confidence': avg_confidence
        }
    
    def _count_emotions(self, emotion_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count occurrences of each emotion."""
        emotion_counts = {}
        for em in emotion_history:
            if not isinstance(em, dict):
                continue
            em_name = self._extract_emotion_name(em)
            emotion_counts[em_name] = emotion_counts.get(em_name, 0) + 1
        return emotion_counts
    
    def _extract_emotion_name(self, emotion: Dict[str, Any]) -> str:
        """Extract emotion name from emotion dictionary."""
        return emotion.get('emotion') or emotion.get('label', 'unknown')
    
    def _calculate_total_confidence(self, emotion_history: List[Dict[str, Any]]) -> float:
        """Calculate total confidence from emotion history."""
        total_confidence = STATE_VIEWER_DEFAULT_CONFIDENCE
        for em in emotion_history:
            if self._is_valid_emotion_dict(em):
                confidence = self._extract_confidence(em)
                total_confidence += confidence
        return total_confidence
    
    def _is_valid_emotion_dict(self, emotion: Any) -> bool:
        """Check if emotion is a valid dictionary."""
        return isinstance(emotion, dict)
    
    def _extract_confidence(self, emotion: Dict[str, Any]) -> float:
        """Extract confidence value from emotion dictionary."""
        return emotion.get('confidence') or emotion.get('intensity', STATE_VIEWER_DEFAULT_CONFIDENCE)
    
    def _find_dominant_emotion(self, emotion_counts: Dict[str, int]) -> Tuple[str, int]:
        """Find the most frequently occurring emotion."""
        return max(emotion_counts.items(), key=lambda x: x[1])
    
    def _calculate_average_confidence(self, total_confidence: float, history_length: int) -> float:
        """Calculate average confidence from total and history length."""
        if history_length == 0:
            return STATE_VIEWER_DEFAULT_CONFIDENCE
        return total_confidence / history_length
    
    def _set_emotion_state(self, stats: Dict[str, Any]):
        """Set emotion display with calculated statistics."""
        self.dominant_emotion_label.config(
            text=f"{stats['dominant_name']} ({stats['dominant_count']}x)"
        )
        self.avg_confidence_label.config(
            text=f"{stats['average_confidence']:.3f}"
        )
    
    def _set_empty_emotion_state(self):
        """Set emotion display to empty state."""
        self.dominant_emotion_label.config(text="None")
        self.avg_confidence_label.config(text=f"{STATE_VIEWER_DEFAULT_CONFIDENCE:.3f}")
    
    def _update_phonemes(self, phonemes: Dict[str, Any]):
        """Update phoneme section."""
        phoneme_history = phonemes.get('history', [])
        phoneme_count = self._count_phoneme_items(phoneme_history)
        self.phoneme_count_label.config(text=f"{phoneme_count}")
        
        speech_threshold = self._extract_speech_threshold(phonemes)
        self.speech_threshold_label.config(text=f"{speech_threshold:.3f}")
    
    def _count_phoneme_items(self, phoneme_history: Any) -> int:
        """Count phoneme items in history."""
        return len(phoneme_history) if isinstance(phoneme_history, list) else 0
    
    def _extract_speech_threshold(self, phonemes: Dict[str, Any]) -> float:
        """Extract speech threshold from phonemes dictionary."""
        return phonemes.get('speech_threshold', STATE_VIEWER_DEFAULT_SPEECH_THRESHOLD)
    
    def _update_timestamp(self):
        """Update last update timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.last_update_label.config(text=f"Last updated: {timestamp}")
    
    def _start_update_thread(self):
        """Start the background update thread."""
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Background thread that updates the display."""
        while self.running:
            try:
                if not self.running:  # Check before processing
                    break
                state = self._load_state()
                if state and self.running:  # Check again after load
                    # Schedule update on main thread (thread-safe)
                    try:
                        self.root.after(0, self._update_display, state)
                    except RuntimeError:
                        # Main loop not running or window closed
                        break
            except Exception:
                if not self.running:
                    break
            if not self.running:  # Check before sleep
                break
            time.sleep(self.update_interval)
    
    def write(self, message):
        """Write method for stdout/stderr redirection."""
        if not self.running:
            return
        
        if self._should_filter_message(message):
            return
        
        tag = self._determine_message_tag(message)
        self.message_queue.put((message, tag))
    
    def _should_filter_message(self, message: str) -> bool:
        """Check if message should be filtered out."""
        message_lower = message.lower()
        return ('tensorflow' in message_lower and 
                (self._is_tensorflow_cpu_message(message_lower) or 
                 'AVX2 FMA' in message or 
                 'optimized to use available CPU' in message))
    
    def _is_tensorflow_cpu_message(self, message_lower: str) -> bool:
        """Check if message is a TensorFlow CPU feature guard message."""
        return 'cpu_feature_guard' in message_lower
    
    def _determine_message_tag(self, message: str) -> str:
        """Determine log tag based on message content."""
        if self._is_error_message(message):
            return 'error'
        elif self._is_warning_message(message):
            return 'warning'
        elif self._is_info_message(message):
            return 'info'
        return 'normal'
    
    def _is_error_message(self, message: str) -> bool:
        """Check if message is an error."""
        return '[ERROR]' in message or 'Error' in message or 'Traceback' in message
    
    def _is_warning_message(self, message: str) -> bool:
        """Check if message is a warning."""
        return '[WARNING]' in message or 'Warning' in message
    
    def _is_info_message(self, message: str) -> bool:
        """Check if message is an info message."""
        return '[INFO]' in message or '[OPU]' in message or '[VISION]' in message
    
    def flush(self):
        """Flush method required for stdout/stderr redirection."""
        pass
    
    def _append_log(self, message, tag='normal'):
        """Append text to log widget (must be called from main thread)."""
        self.log_text.insert(tk.END, message, tag)
        self.log_text.see(tk.END)  # Auto-scroll to bottom
        
        # Limit log size to prevent memory issues (keep last 10000 lines)
        lines = self.log_text.get('1.0', tk.END).split('\n')
        if len(lines) > 10000:
            self.log_text.delete('1.0', f'{len(lines) - 10000}.0')
    
    # --- IMAGE HANDLING (Real-time from Queue) ---
    
    def _start_image_poll_thread(self):
        """Start thread to poll images from shared memory."""
        self.image_thread = threading.Thread(target=self._image_poll_loop, daemon=True)
        self.image_thread.start()
    
    def _image_poll_loop(self):
        """Poll the queue for new frames (webcam or cognitive map)."""
        while self.running:
            try:
                if not self.running:  # Check before blocking
                    break
                # Blocking get with timeout allows checking self.running
                # We expect tuples: ('webcam', image) or ('cognitive_map', image)
                item = self.image_queue.get(timeout=0.1)
                
                if not self.running:  # Check after getting item
                    break
                    
                if isinstance(item, tuple) and len(item) == 2:
                    image_type, image_array = item
                    if image_type == 'webcam':
                        # Schedule GUI update on main thread for webcam
                        try:
                            self.root.after(0, self._update_webcam_image, image_array)
                        except RuntimeError:
                            break
                    elif image_type == 'cognitive_map':
                        # Schedule GUI update on main thread for cognitive map
                        try:
                            self.root.after(0, self._update_map_image, image_array)
                        except RuntimeError:
                            break
                else:
                    # Backward compatibility: if it's just an array, assume cognitive map
                    try:
                        self.root.after(0, self._update_map_image, item)
                    except RuntimeError:
                        break
                
            except queue.Empty:
                continue
            except Exception:
                if not self.running:
                    break
    
    def _update_webcam_image(self, image_array):
        """Convert numpy array to PhotoImage and display in webcam preview."""
        try:
            # Convert Numpy -> PIL Image
            img = Image.fromarray(image_array)
            
            # Resize to fit the webcam preview container (fixed height, maintain aspect ratio)
            target_width = self.config.COGNITIVE_MAP_PANEL_WIDTH - 20
            target_height = self.config.WEBCAM_PREVIEW_HEIGHT - 20  # Account for padding
            
            # Calculate scaling to fit both width and height while maintaining aspect ratio
            width_ratio = target_width / img.width
            height_ratio = target_height / img.height
            ratio = min(width_ratio, height_ratio)  # Use smaller ratio to fit both dimensions
            
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(img_resized)
            
            # Update Label
            self.webcam_label.config(image=photo, text="")
            self.webcam_label.image = photo  # Keep reference!
            
        except Exception:
            # Silently handle image update errors
            pass
    
    def _update_map_image(self, image_array):
        """Convert numpy array to PhotoImage and display in cognitive map."""
        try:
            # Convert Numpy -> PIL Image
            img = Image.fromarray(image_array)
            
            # Resize to fit the cognitive map panel width (minus padding)
            # Use larger size for better visibility
            target_width = self.config.COGNITIVE_MAP_PANEL_WIDTH - 20  # Reduced padding for larger image
            
            if img.width > target_width:
                ratio = target_width / img.width
                target_height = int(img.height * ratio)
                img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            else:
                img_resized = img
            
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(img_resized)
            
            # Update Label
            self.map_label.config(image=photo, text="")
            self.map_label.image = photo  # Keep reference!
            
        except Exception:
            # Silently handle image update errors
            pass
    
    # --- LOGGING SYSTEM (FILE TAILING) ---
    
    def _start_log_tail_thread(self):
        """Start thread to tail the log file."""
        self.log_tail_thread = threading.Thread(target=self._log_tail_loop, daemon=True)
        self.log_tail_thread.start()
    
    def _log_tail_loop(self):
        """Continuously read new lines from opu.log."""
        # Wait for file to exist
        while not self.log_file.exists() and self.running:
            time.sleep(0.5)  # Check more frequently
            if not self.running:
                return
        
        if not self.running:
            return
        
        try:
            with open(self.log_file, "r", encoding='utf-8', errors='ignore') as f:
                # Go to end of file initially (tail mode)
                f.seek(0, 2)
                
                while self.running:
                    if not self.running:  # Check before read
                        break
                    line = f.readline()
                    if line:
                        # Schedule GUI update on main thread
                        try:
                            self.root.after(0, self._append_log_safe, line)
                        except RuntimeError:
                            # Window closed
                            break
                    else:
                        if not self.running:  # Check before sleep
                            break
                        time.sleep(0.1)  # Wait for new data
        except Exception:
            # Silently handle errors
            pass
    
    def _append_log_safe(self, message):
        """Append log (Main Thread Safe)."""
        try:
            tag = self._determine_message_tag(message)
            self.log_text.insert(tk.END, message, tag)
            self.log_text.see(tk.END)
            
            # Limit buffer (keep last 5000 lines)
            line_count = int(self.log_text.index('end-1c').split('.')[0])
            if line_count > 5000:
                self.log_text.delete('1.0', '2.0')
        except Exception:
            # Silently handle errors
            pass
    
    def on_closing(self):
        """Handle window close event."""
        self.running = False
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Destroy window and quit mainloop
        try:
            self.root.quit()  # Exit mainloop
            self.root.destroy()  # Destroy window
        except Exception:
            pass
    
    def run(self):
        """Start the GUI main loop (blocks until window is closed)."""
        try:
            self.root.mainloop()
        except Exception:
            pass  # Window was closed
        finally:
            # Ensure all threads stop
            self.running = False
            # Ensure stdout/stderr are restored
            if sys.stdout == self:
                sys.stdout = self.original_stdout
            if sys.stderr == self:
                sys.stderr = self.original_stderr
            # Destroy window if still exists
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
