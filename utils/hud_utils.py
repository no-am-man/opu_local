"""
HUD Drawing Utilities: Common HUD overlay functions for video frames
"""

from typing import Optional, Tuple
from config import (
    YOUTUBE_HUD_POS_X, YOUTUBE_HUD_POS_Y_LINE1, YOUTUBE_HUD_POS_Y_LINE2, YOUTUBE_HUD_POS_Y_LINE3,
    YOUTUBE_HUD_FONT_SCALE_LARGE, YOUTUBE_HUD_FONT_SCALE_SMALL,
    YOUTUBE_HUD_FONT_THICKNESS, YOUTUBE_HUD_FONT_THICKNESS_THIN,
    STATUS_COLOR_GREEN, TEXT_COLOR_WHITE, TEXT_COLOR_GRAY
)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


def draw_youtube_hud(
    frame,
    safe_score: float,
    s_audio: float,
    s_visual: float,
    title: str,
    frame_count: int,
    fps: float
) -> Optional[any]:
    """
    Draw HUD overlay on YouTube video frame.
    
    Args:
        frame: OpenCV frame (BGR format)
        safe_score: Current safe score
        s_audio: Audio surprise score
        s_visual: Visual surprise score
        title: Video title
        frame_count: Current frame count
        fps: Current FPS
        
    Returns:
        Frame with HUD overlay (or None if cv2 not available)
    """
    if not CV2_AVAILABLE or frame is None:
        return frame
    
    # Main status line
    cv2.putText(
        frame,
        f"s_score: {safe_score:.4f} | Audio: {s_audio:.4f} | Visual: {s_visual:.4f}",
        (YOUTUBE_HUD_POS_X, YOUTUBE_HUD_POS_Y_LINE1),
        cv2.FONT_HERSHEY_SIMPLEX,
        YOUTUBE_HUD_FONT_SCALE_LARGE,
        STATUS_COLOR_GREEN,
        YOUTUBE_HUD_FONT_THICKNESS
    )
    
    # Title line
    title_short = title[:40] + "..." if len(title) > 40 else title
    cv2.putText(
        frame,
        f"Title: {title_short}",
        (YOUTUBE_HUD_POS_X, YOUTUBE_HUD_POS_Y_LINE2),
        cv2.FONT_HERSHEY_SIMPLEX,
        YOUTUBE_HUD_FONT_SCALE_SMALL,
        TEXT_COLOR_WHITE,
        YOUTUBE_HUD_FONT_THICKNESS_THIN
    )
    
    # Frame/FPS line
    cv2.putText(
        frame,
        f"Frame: {frame_count} | FPS: {fps:.1f}",
        (YOUTUBE_HUD_POS_X, YOUTUBE_HUD_POS_Y_LINE3),
        cv2.FONT_HERSHEY_SIMPLEX,
        YOUTUBE_HUD_FONT_SCALE_SMALL,
        TEXT_COLOR_GRAY,
        YOUTUBE_HUD_FONT_THICKNESS_THIN
    )
    
    return frame

