"""
Real-time Emotion Visualization Generator

Generates emotion visualizations as in-memory numpy arrays for display in the State Viewer.
Based on visualize_emotions.py but returns images instead of saving files.
"""

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2
from config import (
    EMOTION_VIZ_DPI, EMOTION_VIZ_SECONDS_TO_MINUTES, EMOTION_VIZ_SCATTER_ALPHA,
    EMOTION_VIZ_SCATTER_SIZE, EMOTION_VIZ_BOX_ALPHA, EMOTION_VIZ_GRID_ALPHA,
    EMOTION_VIZ_CONFIDENCE_MAX, EMOTION_VIZ_PIE_START_ANGLE, EMOTION_VIZ_PIE_TITLE_PAD,
    EMOTION_VIZ_TITLE_FONTSIZE, EMOTION_VIZ_AXIS_FONTSIZE, EMOTION_VIZ_LEGEND_FONTSIZE,
    EMOTION_VIZ_TICK_FONTSIZE
)


def analyze_emotions(emotion_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze emotion history and return statistics."""
    if not emotion_history:
        return {
            'emotion_counts': {},
            'total_emotions': 0,
            'emotions_by_time': [],
            'confidence_by_emotion': defaultdict(list),
            'timeline': []
        }
    
    emotion_counts = Counter()
    emotions_by_time = []
    confidence_by_emotion = defaultdict(list)
    timeline = []
    
    for entry in emotion_history:
        emotion_name, confidence, timestamp = _parse_emotion_entry(entry)
        
        if emotion_name is None:
            continue
        
        emotion_counts[emotion_name] += 1
        confidence_by_emotion[emotion_name].append(confidence)
        
        if timestamp:
            emotions_by_time.append({
                'emotion': emotion_name,
                'confidence': confidence,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp) if timestamp > 0 else None
            })
            timeline.append((timestamp, emotion_name, confidence))
    
    # Sort timeline by timestamp
    timeline.sort(key=lambda x: x[0])
    emotions_by_time.sort(key=lambda x: x['timestamp'] if x['timestamp'] else 0)
    
    return {
        'emotion_counts': dict(emotion_counts),
        'total_emotions': len(emotion_history),
        'emotions_by_time': emotions_by_time,
        'confidence_by_emotion': dict(confidence_by_emotion),
        'timeline': timeline
    }


def _parse_emotion_entry(entry: Dict[str, Any]) -> Tuple[Optional[str], float, float]:
    """Parse emotion entry and extract emotion name, confidence, and timestamp."""
    if not isinstance(entry, dict):
        return None, 0.0, 0.0
    
    emotion_dict = entry.get('emotion', {})
    if not isinstance(emotion_dict, dict):
        return None, 0.0, 0.0
    
    emotion_name = emotion_dict.get('emotion', 'unknown')
    confidence = emotion_dict.get('confidence', 0.0)
    timestamp = entry.get('timestamp', 0)
    
    return emotion_name, confidence, timestamp


def _figure_to_rgb_array(fig) -> np.ndarray:
    """Convert matplotlib figure to RGB numpy array."""
    fig.canvas.draw()
    
    # Handle different matplotlib versions
    # Newer versions (3.5+) use buffer_rgba() instead of tostring_rgb()
    try:
        # Try modern API first (matplotlib 3.5+)
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        buf = buf.reshape((height, width, 4))  # RGBA
        # Convert RGBA to RGB by dropping alpha channel
        buf = buf[:, :, :3]  # Take only RGB channels
    except AttributeError:
        # Fallback to older API (matplotlib < 3.5)
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Last resort: use renderer
            renderer = fig.canvas.get_renderer()
            buf = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV compatibility, then back to RGB for Tkinter
    image_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def _create_figure(width: int, height: int) -> Tuple[plt.Figure, plt.Axes]:
    """Create a matplotlib figure with specified dimensions."""
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=EMOTION_VIZ_DPI)
    return fig, ax


def _enhance_pie_chart_text(autotexts):
    """Enhance pie chart autotext for better visibility."""
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')


def create_pie_chart_image(emotion_counts: Dict[str, int], 
                           width: int = 400, 
                           height: int = 300) -> Optional[np.ndarray]:
    """Create pie chart of emotion distribution as numpy array (RGB)."""
    if not emotion_counts:
        return None
    
    fig, ax = _create_figure(width, height)
    
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=EMOTION_VIZ_PIE_START_ANGLE
    )
    
    _enhance_pie_chart_text(autotexts)
    
    ax.set_title('Emotion Distribution', 
                 fontsize=EMOTION_VIZ_TITLE_FONTSIZE, 
                 fontweight='bold', 
                 pad=EMOTION_VIZ_PIE_TITLE_PAD)
    
    plt.tight_layout()
    return _figure_to_rgb_array(fig)


def _extract_timeline_data(emotions_by_time: List[Dict[str, Any]]) -> Tuple[List[float], List[str], List[float]]:
    """Extract timestamps, emotions, and confidences from emotion history."""
    timestamps = [e['timestamp'] for e in emotions_by_time if e['timestamp']]
    emotions = [e['emotion'] for e in emotions_by_time if e['timestamp']]
    confidences = [e['confidence'] for e in emotions_by_time if e['timestamp']]
    return timestamps, emotions, confidences


def _normalize_timestamps(timestamps: List[float]) -> List[float]:
    """Normalize timestamps to start from 0 and convert to minutes."""
    if not timestamps:
        return []
    start_time = min(timestamps)
    return [(t - start_time) / EMOTION_VIZ_SECONDS_TO_MINUTES for t in timestamps]


def _create_emotion_color_map(emotions: List[str]) -> Dict[str, Tuple[float, ...]]:
    """Create color map for unique emotions."""
    unique_emotions = list(set(emotions))
    return {emotion: plt.cm.tab10(i) for i, emotion in enumerate(unique_emotions)}


def _plot_emotion_scatter(ax, unique_emotions: List[str], emotions: List[str],
                          normalized_times: List[float], confidences: List[float],
                          emotion_colors: Dict[str, Tuple[float, ...]]):
    """Plot scatter points for each emotion type."""
    for emotion in unique_emotions:
        mask = [e == emotion for e in emotions]
        times = [normalized_times[i] for i in range(len(normalized_times)) if mask[i]]
        confs = [confidences[i] for i in range(len(confidences)) if mask[i]]
        ax.scatter(times, confs, label=emotion, color=emotion_colors[emotion],
                   alpha=EMOTION_VIZ_SCATTER_ALPHA, s=EMOTION_VIZ_SCATTER_SIZE)


def create_timeline_image(emotions_by_time: List[Dict[str, Any]],
                         width: int = 600,
                         height: int = 300) -> Optional[np.ndarray]:
    """Create timeline visualization of emotions over time as numpy array (RGB)."""
    if not emotions_by_time:
        return None
    
    timestamps, emotions, confidences = _extract_timeline_data(emotions_by_time)
    if not timestamps:
        return None
    
    normalized_times = _normalize_timestamps(timestamps)
    unique_emotions = list(set(emotions))
    emotion_colors = _create_emotion_color_map(emotions)
    
    fig, ax = _create_figure(width, height)
    
    _plot_emotion_scatter(ax, unique_emotions, emotions, normalized_times, confidences, emotion_colors)
    
    ax.set_xlabel('Time (minutes)', fontsize=EMOTION_VIZ_AXIS_FONTSIZE)
    ax.set_ylabel('Confidence', fontsize=EMOTION_VIZ_AXIS_FONTSIZE)
    ax.set_title('Emotion Timeline', fontsize=EMOTION_VIZ_TITLE_FONTSIZE, fontweight='bold')
    ax.legend(loc='upper right', fontsize=EMOTION_VIZ_LEGEND_FONTSIZE)
    ax.grid(True, alpha=EMOTION_VIZ_GRID_ALPHA)
    ax.set_ylim(0, EMOTION_VIZ_CONFIDENCE_MAX)
    
    plt.tight_layout()
    return _figure_to_rgb_array(fig)


def _colorize_boxplot(bp, num_boxes: int):
    """Apply colors to boxplot boxes."""
    colors = plt.cm.Set3(np.linspace(0, 1, num_boxes))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(EMOTION_VIZ_BOX_ALPHA)


def create_confidence_distribution_image(confidence_by_emotion: Dict[str, List[float]],
                                         width: int = 500,
                                         height: int = 250) -> Optional[np.ndarray]:
    """Create box plot of confidence distribution by emotion as numpy array (RGB)."""
    if not confidence_by_emotion:
        return None
    
    fig, ax = _create_figure(width, height)
    
    emotions = list(confidence_by_emotion.keys())
    data = [confidence_by_emotion[emotion] for emotion in emotions]
    
    bp = ax.boxplot(data, tick_labels=emotions, patch_artist=True)
    _colorize_boxplot(bp, len(emotions))
    
    ax.set_ylabel('Confidence', fontsize=EMOTION_VIZ_AXIS_FONTSIZE)
    ax.set_title('Confidence Distribution', fontsize=EMOTION_VIZ_TITLE_FONTSIZE, fontweight='bold')
    ax.grid(True, alpha=EMOTION_VIZ_GRID_ALPHA, axis='y')
    ax.set_ylim(0, EMOTION_VIZ_CONFIDENCE_MAX)
    
    plt.xticks(rotation=45, ha='right', fontsize=EMOTION_VIZ_TICK_FONTSIZE)
    plt.tight_layout()
    
    return _figure_to_rgb_array(fig)


def generate_all_visualizations(emotion_history: List[Dict[str, Any]],
                                pie_width: int = 400,
                                pie_height: int = 300,
                                timeline_width: int = 600,
                                timeline_height: int = 300,
                                confidence_width: int = 500,
                                confidence_height: int = 250) -> Dict[str, Optional[np.ndarray]]:
    """
    Generate all emotion visualizations as numpy arrays.
    
    Returns:
        Dictionary with keys: 'pie_chart', 'timeline', 'confidence_distribution'
        Values are numpy arrays (RGB) or None if no data available.
    """
    analysis = analyze_emotions(emotion_history)
    
    pie_chart = None
    timeline = None
    confidence_dist = None
    
    if analysis['emotion_counts']:
        pie_chart = create_pie_chart_image(analysis['emotion_counts'], pie_width, pie_height)
    
    if analysis['emotions_by_time']:
        timeline = create_timeline_image(analysis['emotions_by_time'], timeline_width, timeline_height)
    
    if analysis['confidence_by_emotion']:
        confidence_dist = create_confidence_distribution_image(
            analysis['confidence_by_emotion'], confidence_width, confidence_height
        )
    
    return {
        'pie_chart': pie_chart,
        'timeline': timeline,
        'confidence_distribution': confidence_dist
    }

