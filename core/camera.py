"""
The Eye: Camera Capture Module.
Captures camera frames and extracts R, G, B genomic vectors.

This is PERCEPTION (like mic.py for audio).
For INTROSPECTION, see vision_cortex.py.

OPU v3.1 - Multi-Modal Integration
"""

try:
    import cv2
except ImportError:
    cv2 = None
    print("[VISION] Warning: opencv-python not installed. Vision disabled.")

import numpy as np


class VisualPerception:
    """
    The Eye: Visual Perception (Camera Capture).
    
    This module handles PERCEPTION - capturing visual input from the camera
    and extracting genomic vectors. It's the visual equivalent of perception.py.
    
    For INTROSPECTION (calculating surprise), see vision_cortex.py.
    
    Theory:
    - Each color channel (R, G, B) is treated as an independent entropy stream
    - Visual Genomic Bit = Standard Deviation of pixel intensities
    - Frame = <σ_R, σ_G, σ_B> (3-channel structure vector)
    """
    
    def __init__(self, camera_index=0):
        """
        Initialize Visual Cortex with webcam.
        
        Args:
            camera_index: Camera device index (default: 0)
        """
        if cv2 is None:
            self.active = False
            self.cap = None
            print("[VISION] Warning: opencv-python not installed. Vision disabled.")
            return
        
        # Initialize Webcam
        self.cap = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("[VISION] Warning: Camera not found or inaccessible.")
            self.active = False
        else:
            self.active = True
            # Set resolution for object detection (640x480 is good for detection)
            # Still efficient enough for real-time processing
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Reduce frame rate for performance (we don't need 30fps for entropy)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            print("[VISION] Visual Cortex Initialized (3-Channel Mode + Object Detection).")
    
    def get_visual_input(self):
        """
        Captures frame and extracts the 3-channel Genomic Vector.
        
        Math:
        Visual_Bit = < std(Red), std(Green), std(Blue) >
        
        The Standard Deviation measures "Texture/Entropy":
        - Low StdDev = Uniform scene (blank wall, static background)
        - High StdDev = Complex scene (movement, detail, chaos)
        
        Returns:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            frame: Raw BGR frame for display (or None if no camera)
        """
        if not self.active:
            # Return zero-entropy vector if no camera
            return np.array([0.0, 0.0, 0.0]), None

        ret, frame = self.cap.read()
        if not ret:
            return np.array([0.0, 0.0, 0.0]), None

        # 1. SPLIT CHANNELS (OpenCV uses BGR format)
        # We treat each color as a separate orthogonal axis
        if cv2 is None:
            return np.array([0.0, 0.0, 0.0]), None
        
        b, g, r = cv2.split(frame)

        # 2. CALCULATE VISUAL GENOMIC BITS (Scale Invariant Structure)
        # We use Standard Deviation because it measures "Texture/Entropy".
        # A blank wall = Low StdDev. A waving hand = High StdDev.
        sigma_r = np.std(r)
        sigma_g = np.std(g)
        sigma_b = np.std(b)

        visual_vector = np.array([sigma_r, sigma_g, sigma_b], dtype=np.float32)
        
        return visual_vector, frame
    
    def is_active(self):
        """Check if visual cortex is active."""
        return self.active
    
    def cleanup(self):
        """Release camera resources."""
        if self.active:
            self.cap.release()
            self.active = False
            print("[VISION] Visual Cortex cleaned up.")

