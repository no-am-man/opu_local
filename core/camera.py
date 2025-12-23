"""
The Eye: Camera Capture Module.
Captures camera frames and extracts R, G, B genomic vectors.

This is PERCEPTION (like mic.py for audio).
For INTROSPECTION, see vision_cortex.py.

OPU v3.2 - Recursive Perceptual Loop
The OPU can now analyze frames that have been annotated with OpenCV graphics
(bounding boxes, text, overlays). This creates a cybernetic feedback loop where
the OPU "sees its own thoughts" as visual entropy.
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
    and extracting genomic vectors. It's the visual equivalent of mic.py.
    
    For INTROSPECTION (calculating surprise), see vision_cortex.py.
    
    Theory:
    - Each color channel (R, G, B) is treated as an independent entropy stream
    - Visual Genomic Bit = Standard Deviation of pixel intensities
    - Frame = <σ_R, σ_G, σ_B> (3-channel structure vector)
    
    Recursive Perception (v3.2):
    - The OPU can analyze frames that have been annotated with OpenCV graphics
    - Yellow bounding boxes become high-contrast visual entropy
    - Red HUD overlays (panic state) become part of visual reality
    - This creates cybernetic feedback: the OPU sees its own thoughts
    - Enables complex behaviors like "adrenaline loops" and visual confirmation
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
            # This is the capture resolution - display will be resized for preview
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Reduce frame rate for performance (we don't need 30fps for entropy)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            print("[VISION] Visual Cortex Initialized (3-Channel Mode + Object Detection).")
            print("[VISION] WebCam preview window will show object detection overlays.")
    
    def capture_frame(self):
        """
        Captures a raw frame from the webcam.
        This is the unprocessed camera feed.
        
        Returns:
            frame: Raw BGR frame (or None if no camera/error)
        """
        if not self.active:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None
    
    def analyze_frame(self, frame):
        """
        Calculates the Genomic Vector from ANY frame (Raw or Annotated).
        
        This is the key to Recursive Perception: The OPU can analyze
        frames that have been annotated with OpenCV graphics (bounding boxes,
        text, HUD overlays). The OPU will "see its own thoughts" as visual entropy.
        
        Math:
        Visual_Bit = < std(Red), std(Green), std(Blue) >
        
        The Standard Deviation measures "Texture/Entropy":
        - Low StdDev = Uniform scene (blank wall, static background)
        - High StdDev = Complex scene (movement, detail, chaos, ANNOTATIONS)
        
        Args:
            frame: BGR image frame (can be raw or processed/annotated)
            
        Returns:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
        """
        if frame is None:
            return np.array([0.0, 0.0, 0.0])
        
        if cv2 is None:
            return np.array([0.0, 0.0, 0.0])
        
        # 1. SPLIT CHANNELS (OpenCV uses BGR format)
        # We treat each color as a separate orthogonal axis
        b, g, r = cv2.split(frame)

        # 2. CALCULATE VISUAL GENOMIC BITS (Scale Invariant Structure)
        # We use Standard Deviation because it measures "Texture/Entropy".
        # A blank wall = Low StdDev. A waving hand = High StdDev.
        # A yellow bounding box = High StdDev in the Yellow channel (R+G).
        # This is the Recursive Perception: Graphics become part of reality.
        sigma_r = np.std(r)
        sigma_g = np.std(g)
        sigma_b = np.std(b)

        visual_vector = np.array([sigma_r, sigma_g, sigma_b], dtype=np.float32)
        
        return visual_vector
    
    def get_visual_input(self):
        """
        Legacy method for backward compatibility.
        Captures frame and extracts the 3-channel Genomic Vector.
        
        Returns:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            frame: Raw BGR frame for display (or None if no camera)
        """
        frame = self.capture_frame()
        visual_vector = self.analyze_frame(frame)
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

