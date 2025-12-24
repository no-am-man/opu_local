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
from tkinter import ttk
import json
from pathlib import Path
from datetime import datetime
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
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
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 700
    UPDATE_INTERVAL = 0.5
    DEFAULT_STATE_FILE = "opu_state.json"
    
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
    FONT_TITLE = ('Arial', 20, 'bold')
    FONT_SECTION = ('Arial', 14, 'bold')
    FONT_LABEL = ('Arial', 10)
    FONT_VALUE = ('Arial', 10, 'bold')
    
    # Layout
    LABEL_WIDTH = 25
    PADX = 10
    PADY = 5
    PADY_ROW = 2


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
                 update_interval: float = ViewerConfig.UPDATE_INTERVAL):
        """
        Initialize the state viewer.
        
        Args:
            state_file: Path to opu_state.json
            update_interval: How often to refresh (seconds)
        """
        self.state_file = Path(state_file)
        self.update_interval = update_interval
        self.running = True
        self.config = ViewerConfig()
        
        self._create_window()
        self._create_scrollable_container()
        self._create_all_sections()
        self._start_update_thread()
    
    def _create_window(self):
        """Create and configure the main window."""
        self.root = tk.Tk()
        self.root.title("OPU State Viewer")
        self.root.geometry(f"{self.config.WINDOW_WIDTH}x{self.config.WINDOW_HEIGHT}")
        self.root.configure(bg=self.config.BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_scrollable_container(self):
        """Create scrollable container for content."""
        canvas = tk.Canvas(self.root, bg=self.config.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
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
    
    def _create_all_sections(self):
        """Create all display sections."""
        self._create_header()
        self._create_character_section()
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
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _update_display(self, state: Dict[str, Any]):
        """Update the display with current state."""
        if state is None:
            return
        
        cortex = state.get('cortex', {})
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
            count = len(level_data) if isinstance(level_data, list) else 0
            self.memory_labels[level].config(text=f"{count} items")
    
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
        
        if not emotion_history:
            self._set_empty_emotion_state()
            return
        
        emotion_stats = self._calculate_emotion_statistics(emotion_history)
        if emotion_stats:
            self._set_emotion_state(emotion_stats)
        else:
            self._set_empty_emotion_state()
    
    def _calculate_emotion_statistics(self, emotion_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate emotion statistics from history."""
        emotion_counts = {}
        total_confidence = STATE_VIEWER_DEFAULT_CONFIDENCE
        
        for em in emotion_history:
            if not isinstance(em, dict):
                continue
            
            em_name = em.get('emotion', em.get('label', 'unknown'))
            emotion_counts[em_name] = emotion_counts.get(em_name, 0) + 1
            total_confidence += em.get('confidence', em.get('intensity', STATE_VIEWER_DEFAULT_CONFIDENCE))
        
        if not emotion_counts:
            return None
        
        dominant = max(emotion_counts.items(), key=lambda x: x[1])
        avg_confidence = total_confidence / len(emotion_history) if emotion_history else STATE_VIEWER_DEFAULT_CONFIDENCE
        
        return {
            'dominant_name': dominant[0],
            'dominant_count': dominant[1],
            'average_confidence': avg_confidence
        }
    
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
        phoneme_count = len(phoneme_history) if isinstance(phoneme_history, list) else 0
        self.phoneme_count_label.config(text=f"{phoneme_count}")
        
        speech_threshold = phonemes.get('speech_threshold', STATE_VIEWER_DEFAULT_SPEECH_THRESHOLD)
        self.speech_threshold_label.config(text=f"{speech_threshold:.3f}")
    
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
            state = self._load_state()
            if state:
                self.root.after(0, self._update_display, state)
            time.sleep(self.update_interval)
    
    def on_closing(self):
        """Handle window close event."""
        self.running = False
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop (blocks until window is closed)."""
        try:
            self.root.mainloop()
        except Exception:
            pass  # Window was closed
