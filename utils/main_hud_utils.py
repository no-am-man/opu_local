"""
HUD Utilities for Main OPU Event Loop.
Contains functions for drawing HUD overlays on webcam frames.
"""

import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from config import (
    BAR_SPACING, BAR_LABEL_X, BAR_START_X, BAR_HEIGHT,
    MAX_BAR_SCORE, BAR_SCALE_FACTOR,
    FONT_SCALE_SMALL, FONT_SCALE_MEDIUM, FONT_THICKNESS_THICK,
    STATUS_COLOR_RED, STATUS_COLOR_YELLOW, STATUS_COLOR_GREEN,
    CHANNEL_COLOR_RED, CHANNEL_COLOR_GREEN, CHANNEL_COLOR_BLUE,
    ALERT_THRESHOLD, INTEREST_THRESHOLD
)


@dataclass
class MainHUDParams:
    """Parameters for main OPU HUD overlay."""
    frame: np.ndarray
    s_global: float
    s_visual: float
    s_audio: float
    channel_scores: Dict[str, float]
    detections: Optional[list] = None


def draw_main_hud(params: MainHUDParams) -> np.ndarray:
    """
    Draws a HUD overlay on the main OPU webcam frame.
    
    Args:
        params: MainHUDParams containing frame and scores
        
    Returns:
        Annotated frame with HUD overlay
    """
    if params.frame is None:
        return params.frame
    
    display = params.frame.copy()
    h, w = display.shape[:2]
    
    # Draw channel bars
    y = BAR_SPACING
    y = draw_channel_bars(display, params.channel_scores, y)
    
    # Draw global status
    draw_global_status(display, params.s_global, h)
    
    return display


def draw_channel_bars(display: np.ndarray, scores: Dict[str, float], start_y: int) -> int:
    """
    Draw channel score bars (R, G, B) on the display.
    
    Args:
        display: Frame to draw on
        scores: Dictionary of channel scores {'R': float, 'G': float, 'B': float}
        start_y: Starting Y position for bars
        
    Returns:
        Final Y position after drawing all bars
    """
    colors = {
        'R': CHANNEL_COLOR_RED,
        'G': CHANNEL_COLOR_GREEN,
        'B': CHANNEL_COLOR_BLUE
    }
    
    y = start_y
    for ch in ['R', 'G', 'B']:
        score = scores.get(ch, 0.0)
        length = int(min(score, MAX_BAR_SCORE) * BAR_SCALE_FACTOR)
        color = colors[ch]
        
        # Draw label
        cv2.putText(
            display,
            f"{ch}: {score:.2f}",
            (BAR_LABEL_X, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE_SMALL,
            color,
            FONT_THICKNESS_THICK
        )
        
        # Draw bar
        cv2.rectangle(
            display,
            (BAR_START_X, y - BAR_HEIGHT // 2),
            (BAR_START_X + length, y + BAR_HEIGHT // 2),
            color,
            -1
        )
        
        y += BAR_SPACING
    
    return y


def draw_global_status(display: np.ndarray, score: float, frame_height: int):
    """
    Draw global surprise score at the bottom of the frame.
    
    Args:
        display: Frame to draw on
        score: Global surprise score
        frame_height: Height of the frame
    """
    # Determine color based on score
    if score > ALERT_THRESHOLD:
        color = STATUS_COLOR_RED
    elif score > INTEREST_THRESHOLD:
        color = STATUS_COLOR_YELLOW
    else:
        color = STATUS_COLOR_GREEN
    
    # Draw text
    cv2.putText(
        display,
        f"GLOBAL SURPRISE: {score:.2f}",
        (BAR_LABEL_X, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE_MEDIUM,
        color,
        FONT_THICKNESS_THICK
    )

