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

# Import emotion visualizer with error handling
try:
    from utils.emotion_visualizer import generate_all_visualizations
    EMOTION_VIZ_AVAILABLE = True
except ImportError as e:
    EMOTION_VIZ_AVAILABLE = False
    print(f"[VIEWER] Warning: Emotion visualizer not available: {e}", file=sys.stderr)
    def generate_all_visualizations(*args, **kwargs):
        return {'pie_chart': None, 'timeline': None, 'confidence_distribution': None}

# PIL for image display
try:
    from PIL import Image, ImageTk
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageTk = None
    np = None  # Set np to None when not available

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
    STATE_VIEWER_MEMORY_LEVELS_COUNT,
    STATE_VIEWER_EMOTION_HISTORY_MAX_DISPLAY, STATE_VIEWER_EMOTION_TIMESTAMP_FORMAT,
    STATE_VIEWER_EMOTION_UNKNOWN_TIMESTAMP
)


# Constants
@dataclass
class ViewerConfig:
    """Configuration constants for the state viewer."""
    WINDOW_WIDTH = 1200  # Smaller window with 3 evenly distributed columns
    WINDOW_HEIGHT = 700  # Compact height
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
    
    # Visualization sizes
    VIZ_PIE_WIDTH = 300
    VIZ_PIE_HEIGHT = 200
    VIZ_TIMELINE_WIDTH = 300
    VIZ_TIMELINE_HEIGHT = 200
    VIZ_CONFIDENCE_WIDTH = 300
    VIZ_CONFIDENCE_HEIGHT = 200
    VIZ_MAX_DISPLAY_WIDTH = 300
    VIZ_MAX_DISPLAY_HEIGHT = 200
    
    # Valid emotion tags for color coding
    VALID_EMOTION_TAGS = {
        'emotion_angry', 'emotion_happy', 'emotion_sad', 'emotion_fear',
        'emotion_surprise', 'emotion_disgust', 'emotion_neutral', 'emotion_unknown'
    }


@dataclass
class TimeScale:
    """Time scale mapping for maturity levels."""
    SCALES = {
        0: "1s", 1: "1m", 2: "1h", 3: "1d",
        4: "1w", 5: "1mo", 6: "1y", 7: "4y"
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
        
        try:
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
        except Exception as e:
            # Log initialization errors
            import traceback
            error_msg = f"[VIEWER] Initialization error: {e}\n{traceback.format_exc()}"
            try:
                print(error_msg, file=sys.stderr, flush=True)
                # Create error log file if it doesn't exist
                error_log_path = Path("viewer_error.log")
                error_log_path.touch(exist_ok=True)  # Create file if it doesn't exist
                with open(error_log_path, "a") as f:
                    f.write(f"{error_msg}\n")
            except Exception:
                pass
            # Re-raise to let the process know initialization failed
            raise
    
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
        # Ensure window stays open - prevent auto-closing
        self.root.deiconify()  # Make sure window is visible
        self.root.update_idletasks()  # Process pending events
    
    def _create_split_layout(self):
        """Create split paned window for three columns: State | Cognitive Map | Log."""
        # Use tk.PanedWindow (not ttk) to support minsize parameter
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.config.BG_DARK, sashwidth=3)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_scrollable_container(self):
        """Create scrollable container for state content (Left column)."""
        # Left panel for state
        left_frame = tk.Frame(self.paned, bg=self.config.BG_DARK)
        self.paned.add(left_frame, minsize=self.config.WINDOW_WIDTH // 3)  # Equal width distribution
        
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
        self.paned.add(map_frame, minsize=self.config.WINDOW_WIDTH // 3)  # Equal width distribution
        
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
        """Create log panel on the right side with emotional history above."""
        # Right panel container
        right_frame = tk.Frame(self.paned, bg=self.config.BG_DARK)
        self.paned.add(right_frame, minsize=self.config.WINDOW_WIDTH // 3)  # Equal width distribution
        
        # Create vertical paned window to split emotional history and log
        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- EMOTIONAL HISTORY SECTION (Top) ---
        self._create_emotion_history_panel(right_paned)
        
        # --- LOG SECTION (Bottom) ---
        self._create_log_section(right_paned)
    
    def _create_emotion_history_panel(self, parent):
        """Create emotional history panel with visualizations above the log."""
        # Emotion history frame
        emotion_frame = tk.Frame(parent, bg=self.config.BG_DARK)
        parent.add(emotion_frame)  # PanedWindow.add() doesn't support weight parameter
        
        # Header
        emotion_header = tk.Label(
            emotion_frame,
            text="Emotional History (Real-Time)",
            font=self.config.FONT_SECTION,
            fg=self.config.TEXT_WHITE,
            bg=self.config.BG_SECTION,
            anchor='w'
        )
        emotion_header.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a horizontal paned window for visualizations and text
        viz_paned = tk.PanedWindow(emotion_frame, orient=tk.HORIZONTAL, bg=self.config.BG_DARK, sashwidth=3)
        viz_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side: Visualizations
        viz_container = tk.Frame(viz_paned, bg=self.config.BG_DARK)
        viz_paned.add(viz_container, minsize=200)  # PanedWindow.add() doesn't accept weight parameter
        
        # Create visualization labels (will be updated with images)
        self.pie_chart_label = tk.Label(
            viz_container,
            text="Pie Chart\n(Loading...)",
            bg=self.config.BG_DARK,
            fg=self.config.TEXT_GRAY,
            font=('Arial', 9)
        )
        self.pie_chart_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.timeline_label = tk.Label(
            viz_container,
            text="Timeline\n(Loading...)",
            bg=self.config.BG_DARK,
            fg=self.config.TEXT_GRAY,
            font=('Arial', 9)
        )
        self.timeline_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.confidence_label = tk.Label(
            viz_container,
            text="Confidence\n(Loading...)",
            bg=self.config.BG_DARK,
            fg=self.config.TEXT_GRAY,
            font=('Arial', 9)
        )
        self.confidence_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Right side: Text history
        text_frame = tk.Frame(viz_paned, bg=self.config.BG_DARK)
        viz_paned.add(text_frame, minsize=150)  # PanedWindow.add() doesn't accept weight parameter
        
        # Scrollable text widget for emotion history
        self.emotion_text = scrolledtext.ScrolledText(
            text_frame,
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='#ffffff',
            selectbackground='#264f78',
            font=('Courier', 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            height=15  # Fixed height for emotion history
        )
        self.emotion_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure text tags for emotion colors
        self.emotion_text.tag_config('timestamp', foreground='#888888')
        self.emotion_text.tag_config('emotion_angry', foreground='#f48771')
        self.emotion_text.tag_config('emotion_happy', foreground='#4ec9b0')
        self.emotion_text.tag_config('emotion_sad', foreground='#569cd6')
        self.emotion_text.tag_config('emotion_fear', foreground='#dcdcaa')
        self.emotion_text.tag_config('emotion_surprise', foreground='#ce9178')
        self.emotion_text.tag_config('emotion_disgust', foreground='#c586c0')
        self.emotion_text.tag_config('emotion_neutral', foreground='#d4d4d4')
        self.emotion_text.tag_config('emotion_unknown', foreground='#808080')
        self.emotion_text.tag_config('confidence', foreground='#9e9e9e')
        
        # Add initial message
        self._append_emotion("=" * 60 + "\n", 'emotion_neutral')
        self._append_emotion("OPU Emotional History\n", 'emotion_neutral')
        self._append_emotion("=" * 60 + "\n", 'emotion_neutral')
        
        # Store last emotion count to detect new entries
        self.last_emotion_count = 0
        
        # Force initial visualization update to replace "Loading..." text
        # This ensures labels are updated immediately, even if emotion_history is empty
        try:
            self._update_emotion_visualizations([])
        except Exception as e:
            # Log error but don't break initialization - update will happen on first state file read
            self._log_error("initial emotion visualization update", e)
    
    def _create_log_section(self, parent):
        """Create log section below emotional history."""
        # Log frame
        log_frame = tk.Frame(parent, bg=self.config.BG_DARK)
        parent.add(log_frame)  # PanedWindow.add() doesn't support weight parameter
        
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
        except (json.JSONDecodeError, IOError) as e:
            self._log_error(f"reading state file {self.state_file}", e)
            return None
    
    def _update_display(self, state: Dict[str, Any]):
        """Update the display with current state (must be called from main thread)."""
        if not self._can_update_display(state):
            return
        
        try:
            cortex = self._extract_cortex(state)
            self._update_all_sections(cortex, state)
            # Note: Cognitive map is updated via image polling thread, not here
        except Exception as e:
            # Log errors but don't break the UI
            self._log_error("updating display", e)
            # Still try to update timestamp even if other updates fail
            try:
                self._update_timestamp()
            except Exception:
                pass
    
    def _can_update_display(self, state: Dict[str, Any]) -> bool:
        """Check if display can be updated."""
        return state is not None and self.running
    
    def _extract_cortex(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cortex data from state."""
        return state.get('cortex', {})
    
    def _update_all_sections(self, cortex: Dict[str, Any], state: Dict[str, Any]):
        """Update all display sections."""
        try:
            self._update_character_profile(cortex.get('character_profile', {}))
        except Exception as e:
            self._log_error("updating character profile", e)
        
        try:
            self._update_memory_levels(cortex.get('memory_levels', {}))
        except Exception as e:
            self._log_error("updating memory levels", e)
        
        try:
            self._update_current_state(cortex.get('current_state', {}))
        except Exception as e:
            self._log_error("updating current state", e)
        
        try:
            self._update_emotion_history(cortex.get('emotion_history', []))
        except Exception as e:
            self._log_error("updating emotion history", e)
        
        try:
            self._update_phonemes(state.get('phonemes', {}))
        except Exception as e:
            self._log_error("updating phonemes", e)
        
        try:
            self._update_timestamp()
        except Exception as e:
            self._log_error("updating timestamp", e)
    
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
        """Update emotion history section (both left panel stats and right panel real-time list)."""
        emotion_count = len(emotion_history)
        self.emotion_count_label.config(text=f"{emotion_count}")
        
        if self._has_no_emotions(emotion_history):
            self._set_empty_emotion_state()
            # Still update panel to show "No data" in visualizations instead of "Loading..."
            self._update_emotion_history_panel(emotion_history)
            return
        
        emotion_stats = self._calculate_emotion_statistics(emotion_history)
        self._set_emotion_state_from_stats(emotion_stats)
        
        # Update real-time emotional history panel (right side)
        self._update_emotion_history_panel(emotion_history)
    
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
        # Handle nested emotion structure: {'emotion': {'emotion': 'angry', 'confidence': 0.8}, ...}
        emotion_data = emotion.get('emotion')
        if isinstance(emotion_data, dict):
            return emotion_data.get('emotion', 'unknown')
        # Handle flat structure: {'emotion': 'angry', ...}
        if isinstance(emotion_data, str):
            return emotion_data
        # Fallback to label or unknown
        return emotion.get('label', 'unknown')
    
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
        # Handle nested emotion structure: {'emotion': {'emotion': 'angry', 'confidence': 0.8}, ...}
        emotion_data = emotion.get('emotion')
        if isinstance(emotion_data, dict):
            return emotion_data.get('confidence', STATE_VIEWER_DEFAULT_CONFIDENCE)
        # Handle flat structure: {'emotion': 'angry', 'confidence': 0.8, ...}
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
    
    def _update_emotion_history_panel(self, emotion_history: List[Dict[str, Any]]):
        """Update the real-time emotional history panel with new emotions and visualizations."""
        if not hasattr(self, 'emotion_text'):
            return
        
        current_count = len(emotion_history)
        
        # Initialize last_emotion_count if not set
        if not hasattr(self, 'last_emotion_count'):
            self.last_emotion_count = 0
        
        # Always update visualizations first (even if emotion_history is empty)
        # This ensures "No data" is shown instead of staying at "Loading..."
        # Force update even if labels don't exist yet (they should be created by now)
        try:
            self._update_emotion_visualizations(emotion_history)
        except Exception as e:
            # Log but don't break the UI
            self._log_error("in _update_emotion_history_panel", e)
            # Even on error, try to update labels to show error state instead of "Loading..."
            try:
                if self._has_visualization_labels():
                    self._set_all_labels_error(e)
            except Exception:
                pass
        
        # Check if this is the first update or if there are new emotions
        is_first_update = (self.last_emotion_count == 0 and current_count > 0)
        has_new_emotions = current_count > self.last_emotion_count
        
        if not is_first_update and not has_new_emotions:
            return
        
        # On first update with existing emotions, show the most recent ones
        # Otherwise, show only new emotions
        max_display = STATE_VIEWER_EMOTION_HISTORY_MAX_DISPLAY
        if is_first_update:
            # First update: show the most recent emotions (up to max_display)
            if current_count > max_display:
                display_emotions = emotion_history[-max_display:]
                self._handle_emotion_history_overflow(emotion_history, current_count, max_display)
            else:
                display_emotions = emotion_history
                # Clear any initial messages
                self.emotion_text.delete('1.0', tk.END)
        else:
            # Subsequent updates: show only new emotions
            new_emotions = emotion_history[self.last_emotion_count:]
            if current_count > max_display:
                # If we're over the limit, show only the most recent
                display_emotions = emotion_history[-max_display:]
                self._handle_emotion_history_overflow(emotion_history, current_count, max_display)
            else:
                display_emotions = new_emotions
        
        # Update the count
        self.last_emotion_count = current_count
        
        # Append emotions to display
        for entry in display_emotions:
            self._append_emotion_entry(entry)
        
        # Auto-scroll to bottom
        self.emotion_text.see(tk.END)
    
    def _update_emotion_visualizations(self, emotion_history: List[Dict[str, Any]]):
        """Update emotion visualization images in real-time."""
        try:
            # Check if labels exist - if not, we can't update them
            if not self._has_visualization_labels():
                # Labels should exist by now, but if they don't, log it
                self._log_error("updating emotion visualizations", 
                               Exception("Visualization labels not found"))
                return
            
            # Check for missing libraries and update labels accordingly
            missing_library = self._check_required_libraries()
            if missing_library:
                self._set_all_labels_unavailable(missing_library)
                return
            
            # Generate visualizations and update labels
            visualizations = self._generate_emotion_visualizations(emotion_history)
            self._update_all_visualization_labels(visualizations)
        except Exception as e:
            self._log_error("updating emotion visualizations", e)
            # Always try to update labels to show error state instead of "Loading..."
            try:
                if self._has_visualization_labels():
                    self._set_all_labels_error(e)
            except Exception:
                pass
    
    def _has_visualization_labels(self) -> bool:
        """Check if visualization labels exist."""
        return (hasattr(self, 'pie_chart_label') and 
                hasattr(self, 'timeline_label') and 
                hasattr(self, 'confidence_label'))
    
    def _check_required_libraries(self) -> Optional[str]:
        """Check if required libraries are available. Returns missing library name or None."""
        if not PIL_AVAILABLE:
            return "PIL"
        if not CV2_AVAILABLE:
            return "OpenCV"
        if not EMOTION_VIZ_AVAILABLE:
            return "Visualizer"
        return None
    
    def _set_all_labels_unavailable(self, library_name: str):
        """Set all visualization labels to show library unavailable message."""
        message = f"({library_name} not available)"
        self.pie_chart_label.config(text=f"Pie Chart\n{message}", image='')
        self.timeline_label.config(text=f"Timeline\n{message}", image='')
        self.confidence_label.config(text=f"Confidence\n{message}", image='')
    
    def _generate_emotion_visualizations(self, emotion_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate all emotion visualizations."""
        return generate_all_visualizations(
            emotion_history,
            pie_width=self.config.VIZ_PIE_WIDTH,
            pie_height=self.config.VIZ_PIE_HEIGHT,
            timeline_width=self.config.VIZ_TIMELINE_WIDTH,
            timeline_height=self.config.VIZ_TIMELINE_HEIGHT,
            confidence_width=self.config.VIZ_CONFIDENCE_WIDTH,
            confidence_height=self.config.VIZ_CONFIDENCE_HEIGHT
        )
    
    def _update_all_visualization_labels(self, visualizations: Dict[str, Any]):
        """Update all visualization labels with generated images or 'No data' message."""
        self._update_single_visualization(
            self.pie_chart_label, visualizations['pie_chart'], "Pie Chart"
        )
        self._update_single_visualization(
            self.timeline_label, visualizations['timeline'], "Timeline"
        )
        self._update_single_visualization(
            self.confidence_label, visualizations['confidence_distribution'], "Confidence"
        )
    
    def _update_single_visualization(self, label: tk.Label, image_array: Any, label_name: str):
        """Update a single visualization label with image or 'No data' message."""
        if image_array is not None:
            self._update_visualization_image(label, image_array)
        else:
            label.config(text=f"{label_name}\n(No data)", image='')
    
    def _set_all_labels_error(self, error: Exception):
        """Set all visualization labels to show error state."""
        error_msg = str(error)[:30]
        try:
            self.pie_chart_label.config(text=f"Pie Chart\n(Error: {error_msg})", image='')
            self.timeline_label.config(text=f"Timeline\n(Error: {error_msg})", image='')
            self.confidence_label.config(text=f"Confidence\n(Error: {error_msg})", image='')
        except Exception:
            pass
    
    def _log_error(self, context: str, error: Exception):
        """Log error to stderr and error log file."""
        import traceback
        error_msg = f"[VIEWER] Error {context}: {error}\n{traceback.format_exc()}"
        try:
            print(error_msg, file=sys.stderr, flush=True)
            # Create error log file if it doesn't exist
            error_log_path = Path("viewer_error.log")
            error_log_path.touch(exist_ok=True)  # Create file if it doesn't exist
            with open(error_log_path, "a") as f:
                f.write(f"{error_msg}\n")
        except Exception:
            pass
    
    def _update_visualization_image(self, label: tk.Label, image_array: Any):
        """Update a label with a numpy array image."""
        try:
            if image_array is None:
                return
            
            image_resized = self._resize_image_for_display(image_array)
            photo = self._convert_array_to_photoimage(image_resized)
            
            # Update label
            label.config(image=photo, text='')
            label.image = photo  # Keep a reference to prevent garbage collection
        except Exception as e:
            self._log_error("updating visualization image", e)
    
    def _resize_image_for_display(self, image_array: Any) -> Any:
        """Resize image array to fit display constraints."""
        height, width = image_array.shape[:2]
        max_width, max_height = self.config.VIZ_MAX_DISPLAY_WIDTH, self.config.VIZ_MAX_DISPLAY_HEIGHT
        
        # Calculate scaling
        scale = min(max_width / width, max_height / height, 1.0)
        
        if scale >= 1.0:
            return image_array
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _convert_array_to_photoimage(self, image_array: Any):
        """Convert numpy array to Tkinter PhotoImage."""
        from PIL import Image
        pil_image = Image.fromarray(image_array)
        return ImageTk.PhotoImage(image=pil_image)
    
    def _handle_emotion_history_overflow(self, emotion_history: List[Dict[str, Any]], 
                                         current_count: int, max_display: int) -> List[Dict[str, Any]]:
        """Handle emotion history overflow by clearing and showing only recent entries."""
        self.emotion_text.delete('1.0', tk.END)
        display_emotions = emotion_history[-max_display:]
        self._append_emotion("=" * 60 + "\n", 'emotion_neutral')
        self._append_emotion(f"Showing last {max_display} emotions (Total: {current_count})\n", 'emotion_neutral')
        self._append_emotion("=" * 60 + "\n", 'emotion_neutral')
        return display_emotions
    
    def _append_emotion_entry(self, entry: Dict[str, Any]):
        """Append a single emotion entry to the history panel."""
        if not self._is_valid_emotion_entry(entry):
            return
        
        emotion_name, confidence, timestamp = self._extract_emotion_data(entry)
        time_str = self._format_emotion_timestamp(timestamp)
        emotion_tag = self._get_emotion_tag(emotion_name)
        
        self._insert_emotion_text(time_str, emotion_name, confidence, emotion_tag)
    
    def _is_valid_emotion_entry(self, entry: Dict[str, Any]) -> bool:
        """Check if emotion entry is valid."""
        if not isinstance(entry, dict):
            return False
        emotion_dict = entry.get('emotion', {})
        return isinstance(emotion_dict, dict)
    
    def _extract_emotion_data(self, entry: Dict[str, Any]) -> Tuple[str, float, float]:
        """Extract emotion name, confidence, and timestamp from entry."""
        emotion_dict = entry.get('emotion', {})
        emotion_name = emotion_dict.get('emotion', 'unknown')
        confidence = emotion_dict.get('confidence', 0.0)
        timestamp = entry.get('timestamp', 0)
        return emotion_name, confidence, timestamp
    
    def _format_emotion_timestamp(self, timestamp: float) -> str:
        """Format timestamp for emotion display."""
        if timestamp > 0:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime(STATE_VIEWER_EMOTION_TIMESTAMP_FORMAT)
        return STATE_VIEWER_EMOTION_UNKNOWN_TIMESTAMP
    
    def _get_emotion_tag(self, emotion_name: str) -> str:
        """Get emotion tag for color coding."""
        emotion_tag = f'emotion_{emotion_name.lower()}'
        if emotion_tag not in self.config.VALID_EMOTION_TAGS:
            return 'emotion_unknown'
        return emotion_tag
    
    def _insert_emotion_text(self, time_str: str, emotion_name: str, 
                            confidence: float, emotion_tag: str):
        """Insert formatted emotion text into the display."""
        self.emotion_text.insert(tk.END, f"[{time_str}] ", 'timestamp')
        self.emotion_text.insert(tk.END, f"{emotion_name.upper():10s} ", emotion_tag)
        self.emotion_text.insert(tk.END, f"(conf: {confidence:.3f})\n", 'confidence')
    
    def _append_emotion(self, text: str, tag: str = 'emotion_neutral'):
        """Append text to emotion history panel."""
        if hasattr(self, 'emotion_text'):
            self.emotion_text.insert(tk.END, text, tag)
            self.emotion_text.see(tk.END)
    
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
                    except RuntimeError as e:
                        # Main loop not running or window closed
                        self._log_error("scheduling display update (RuntimeError)", e)
                        break
            except Exception as e:
                self._log_error("in update loop", e)
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
                        except RuntimeError as e:
                            self._log_error("scheduling webcam image update (RuntimeError)", e)
                            break
                    elif image_type == 'cognitive_map':
                        # Schedule GUI update on main thread for cognitive map
                        try:
                            self.root.after(0, self._update_map_image, image_array)
                        except RuntimeError as e:
                            self._log_error("scheduling cognitive map image update (RuntimeError)", e)
                            break
                else:
                    # Backward compatibility: if it's just an array, assume cognitive map
                    try:
                        self.root.after(0, self._update_map_image, item)
                    except RuntimeError as e:
                        self._log_error("scheduling map image update (RuntimeError)", e)
                        break
                
            except queue.Empty:
                continue
            except Exception as e:
                self._log_error("in image poll loop", e)
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
            
        except Exception as e:
            self._log_error("updating webcam image", e)
    
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
            
        except Exception as e:
            self._log_error("updating webcam image", e)
    
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
                        except RuntimeError as e:
                            # Window closed
                            self._log_error("scheduling log append (RuntimeError)", e)
                            break
                    else:
                        if not self.running:  # Check before sleep
                            break
                        time.sleep(0.1)  # Wait for new data
        except Exception as e:
            self._log_error("in log tail loop", e)
    
    def _append_log_safe(self, message):
        """Append log (Main Thread Safe)."""
        try:
            tag = self._determine_message_tag(message)
            self.log_text.insert(tk.END, message, tag)
            self.log_text.see(tk.END)
            
            # Limit buffer (keep last 5000 lines)
            try:
                line_count = int(self.log_text.index('end-1c').split('.')[0])
                if line_count > 5000:
                    self.log_text.delete('1.0', '2.0')
            except (ValueError, tk.TclError):
                # Ignore errors from line counting/deletion - not critical
                pass
        except Exception as e:
            self._log_error("appending log safely", e)
    
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
        import traceback
        try:
            # Ensure window is visible before starting mainloop
            self.root.update_idletasks()
            self.root.deiconify()  # Make sure window is visible
            self.root.mainloop()
        except Exception as e:
            # Log the error instead of silently failing
            error_msg = f"[VIEWER] Mainloop error: {e}\n{traceback.format_exc()}"
            try:
                print(error_msg, file=sys.stderr, flush=True)
                # Create error log file if it doesn't exist
                error_log_path = Path("viewer_error.log")
                error_log_path.touch(exist_ok=True)  # Create file if it doesn't exist
                with open(error_log_path, "a") as f:
                    f.write(f"{error_msg}\n")
            except Exception:
                pass
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
