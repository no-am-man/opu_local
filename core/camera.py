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
    
    Color Constancy (v3.3):
    - If enabled: Normalizes RGB by luminance (R+G+B) to achieve shadow-invariance
    - Chromaticity (r=R/Σ, g=G/Σ, b=B/Σ) remains constant under lighting changes
    - OPU only responds to actual color/structure changes, not shadows/flickers
    - This implements biological color constancy (lateral inhibition)
    
    Recursive Perception (v3.2):
    - The OPU can analyze frames that have been annotated with OpenCV graphics
    - Yellow bounding boxes become high-contrast visual entropy
    - Red HUD overlays (panic state) become part of visual reality
    - This creates cybernetic feedback: the OPU sees its own thoughts
    - Enables complex behaviors like "adrenaline loops" and visual confirmation
    """
    
    def __init__(self, camera_index=0, use_color_constancy=True):
        """
        Initialize Visual Cortex with webcam.
        
        Args:
            camera_index: Camera device index (default: 0)
            use_color_constancy: If True, uses normalized chromaticity (shadow-invariant).
                                If False, uses raw RGB channels (legacy mode).
        """
        if cv2 is None:
            self.active = False
            self.cap = None
            print("[VISION] Warning: opencv-python not installed. Vision disabled.")
            return
        
        self.use_color_constancy = use_color_constancy
        
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
            mode = "Color Constancy (Shadow-Invariant)" if use_color_constancy else "Raw RGB (Legacy)"
            print(f"[VISION] Visual Cortex Initialized ({mode} + Object Detection).")
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
        
        OPU v3.3 - Color Constancy (Shadow Invariance):
        - If use_color_constancy=True: Normalizes RGB by luminance (R+G+B)
          This makes the OPU invariant to lighting changes (shadows, flickers)
          and only respond to actual color/structure changes.
        - If use_color_constancy=False: Uses raw RGB channels (legacy mode)
        
        This is the key to Recursive Perception: The OPU can analyze
        frames that have been annotated with OpenCV graphics (bounding boxes,
        text, HUD overlays). The OPU will "see its own thoughts" as visual entropy.
        
        Math (Color Constancy Mode):
        - Luminance: Σ = R + G + B (Energy/Intensity)
        - Chromaticity: r = R/Σ, g = G/Σ, b = B/Σ (Color Identity, shadow-invariant)
        - Visual_Bit = < std(r_norm), std(g_norm), std(b_norm) >
        
        Math (Legacy Mode):
        - Visual_Bit = < std(Red), std(Green), std(Blue) >
        
        The Standard Deviation measures "Texture/Entropy":
        - Low StdDev = Uniform scene (blank wall, static background)
        - High StdDev = Complex scene (movement, detail, chaos, ANNOTATIONS)
        
        Args:
            frame: BGR image frame (can be raw or processed/annotated)
            
        Returns:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b]) or
                         np.array([sigma_r_norm, sigma_g_norm, sigma_b_norm])
        """
        if frame is None:
            return np.array([0.0, 0.0, 0.0])
        
        if cv2 is None:
            return np.array([0.0, 0.0, 0.0])
        
        # 1. SPLIT CHANNELS (OpenCV uses BGR format)
        # Convert to float32 for precise calculations
        b, g, r = cv2.split(frame.astype(np.float32))
        
        if self.use_color_constancy:
            # 2. COLOR CONSTANCY MODE: Normalized Chromaticity (Shadow-Invariant)
            # Calculate Total Luminance (Energy/Intensity)
            # Add epsilon to prevent division by zero
            epsilon = 0.001
            luminance = r + g + b + epsilon
            
            # 3. Calculate Normalized Chromaticity (Color Identity)
            # This removes shadows/lighting changes from the equation
            # If a cloud passes: R, G, B all drop proportionally → r_norm, g_norm, b_norm stay constant
            # If a red ball moves: r_norm changes → OPU detects actual structure change
            r_norm = r / luminance
            g_norm = g / luminance
            b_norm = b / luminance
            
            # 4. Calculate Entropy of NORMALIZED channels
            # This detects "Movement of Color" independent of "Movement of Light"
            # Scale up small ratios (0-1 range) to match raw RGB scale (0-255 range)
            scale_factor = 100.0  # Amplify normalized ratios for comparable magnitude
            sigma_r = np.std(r_norm) * scale_factor
            sigma_g = np.std(g_norm) * scale_factor
            sigma_b = np.std(b_norm) * scale_factor
            
        else:
            # 2. LEGACY MODE: Raw RGB Channels
            # Calculate Visual Genomic Bits (Scale Invariant Structure)
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

