"""
The Face: Matplotlib HUD for Cognitive Map visualization.
Shows pulse size (s_score), color (Tension), and shape integrity (Coherence).

Now implements Observer Pattern to react to OPU state changes.
"""

# CRITICAL: Set matplotlib backend BEFORE importing pyplot to avoid tkinter conflicts
# This must be done before any matplotlib imports
import os
import platform

# FIX: FORCE NON-INTERACTIVE BACKEND (Agg)
# This prevents Matplotlib from trying to open a Tkinter window,
# which crashes macOS when OpenCV is also running.
# Matplotlib will render to an image buffer instead, which we'll display via OpenCV.
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - renders to image buffer only

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Optional cv2 import for rendering to OpenCV image
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from config import VISUALIZATION_UPDATE_RATE, WINDOW_SIZE
from core.patterns.observer import OPUObserver


class CognitiveMapVisualizer(OPUObserver):
    """
    Real-time visualization of the OPU's cognitive state.
    """
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=WINDOW_SIZE, dpi=100)
        self.fig.tight_layout(pad=2.0)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('OPU Cognitive Map', fontsize=16, fontweight='bold')
        
        # History for smooth animation
        self.s_score_history = deque(maxlen=100)
        self.coherence_history = deque(maxlen=100)
        self.maturity_history = deque(maxlen=100)
        
        # Current state
        self.current_s_score = 0.0
        self.current_coherence = 0.0
        self.current_maturity = 0.0
        self.current_maturity_level = 0
        
        # Visualization elements
        self.pulse_circle = None
        self.coherence_shape = None
        self.text_info = None
        
        self.setup_visualization()
    
    def setup_visualization(self):
        """Initialize visualization elements."""
        # Initial pulse circle (will be updated)
        self.pulse_circle = plt.Circle((0, 0), 0.1, color='blue', alpha=0.5)
        self.ax.add_patch(self.pulse_circle)
        
        # Coherence shape (polygon representing integrity)
        self.coherence_shape = plt.Polygon(
            [[0, 0], [0.1, 0], [0.05, 0.1], [-0.05, 0.1]],
            color='green',
            alpha=0.3
        )
        self.ax.add_patch(self.coherence_shape)
        
        # Text info
        self.text_info = self.ax.text(
            0, -1.8,
            'Initializing...',
            ha='center',
            fontsize=10,
            family='monospace'
        )
    
    def update_state(self, s_score, coherence, maturity, maturity_level=None):
        """
        Update the visualization with new cognitive state.
        
        Args:
            s_score: surprise score (pulse size)
            coherence: coherence value (shape integrity)
            maturity: maturity index (0.0 to 1.0)
            maturity_level: maturity level (0-5) representing time scale
        """
        self.current_s_score = s_score
        self.current_coherence = coherence
        self.current_maturity = maturity
        if maturity_level is not None:
            self.current_maturity_level = maturity_level
        
        # Add to history
        self.s_score_history.append(s_score)
        self.coherence_history.append(coherence)
        self.maturity_history.append(maturity)
    
    def draw_cognitive_map(self):
        """
        Draws the cognitive map with:
        - Pulse size = s_score
        - Color = Tension (red = high, blue = low)
        - Shape integrity = Coherence
        """
        try:
            # Clear previous frame
            self.ax.clear()
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            self.ax.set_title('OPU Cognitive Map', fontsize=16, fontweight='bold')
            
            # Calculate pulse size from s_score
            # Normalize s_score to reasonable radius (0.1 to 1.5)
            # Ensure minimum visible size even when s_score is 0
            pulse_radius = 0.15 + min(self.current_s_score / 5.0, 1.35)
            
            # Calculate color based on tension (s_score)
            # High tension = red, low tension = blue
            # When s_score is 0, use a light blue-gray to make it visible
            tension = min(self.current_s_score / 10.0, 1.0)
            if self.current_s_score <= 0.01:
                # Very low or zero s_score: use light blue-gray for visibility
                color_r = 0.4
                color_g = 0.5
                color_b = 0.7
            else:
                color_r = tension
                color_b = 1.0 - tension
                color_g = 0.3
            pulse_color = (color_r, color_g, color_b)
            
            # Draw pulse circle
            self.pulse_circle = plt.Circle(
                (0, 0),
                pulse_radius,
                facecolor=pulse_color,
                alpha=0.6,
                edgecolor='black',
                linewidth=2
            )
            self.ax.add_patch(self.pulse_circle)
            
            # Draw coherence shape (polygon with integrity based on coherence)
            # Higher coherence = more regular shape
            coherence_points = self._generate_coherence_shape(self.current_coherence)
            self.coherence_shape = plt.Polygon(
                coherence_points,
                facecolor='green',
                alpha=0.4 * self.current_coherence,
                edgecolor='darkgreen',
                linewidth=2
            )
            self.ax.add_patch(self.coherence_shape)
            
            # Draw maturity indicator (outer ring)
            maturity_radius = 1.5 + (self.current_maturity * 0.3)
            maturity_circle = plt.Circle(
                (0, 0),
                maturity_radius,
                fill=False,
                edgecolor='gold',
                linewidth=2 + (self.current_maturity * 3),
                alpha=0.5
            )
            self.ax.add_patch(maturity_circle)
            
            # Get time scale name for current maturity level
            time_scales = {
                0: "1 minute",
                1: "1 hour",
                2: "1 day",
                3: "1 week",
                4: "1 month",
                5: "1 year",
                6: "10 years"
            }
            time_scale = time_scales.get(self.current_maturity_level, "unknown")
            
            # Draw text info
            info_text = (
                f"s_score: {self.current_s_score:.4f}\n"
                f"Coherence: {self.current_coherence:.2f}\n"
                f"Maturity Level: {self.current_maturity_level} ({time_scale})\n"
                f"Maturity Index: {self.current_maturity:.2f}\n"
                f"Pitch: {440.0 - (self.current_maturity * 330.0):.0f}Hz"
            )
            self.text_info = self.ax.text(
                0, -1.8,
                info_text,
                ha='center',
                fontsize=10,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            # Draw history trace
            if len(self.s_score_history) > 1:
                history_array = np.array(list(self.s_score_history))
                normalized_history = (history_array - history_array.min()) / (history_array.max() - history_array.min() + 1e-6)
                x_trace = np.linspace(-1.5, 1.5, len(normalized_history))
                y_trace = normalized_history * 0.5 - 1.0
                self.ax.plot(x_trace, y_trace, 'b-', alpha=0.3, linewidth=1)
        except (KeyboardInterrupt, SystemExit):
            # Re-raise to allow proper shutdown
            raise
        except Exception:
            # Silently ignore matplotlib errors (e.g., window closed, threading issues)
            # This prevents crashes when matplotlib operations are interrupted
            pass
    
    def _generate_coherence_shape(self, coherence):
        """
        Generates a polygon shape based on coherence.
        Higher coherence = more regular (hexagon-like)
        Lower coherence = irregular (triangle-like)
        
        Args:
            coherence: coherence value (0.0 to 1.0)
            
        Returns:
            list of (x, y) tuples for polygon vertices
        """
        # Number of sides increases with coherence
        num_sides = int(3 + coherence * 3)  # 3 to 6 sides
        
        # Radius varies with coherence
        radius = 0.2 + (coherence * 0.3)
        
        # Generate regular polygon
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
        points = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append((x, y))
        
        return points
    
    def render_to_image(self):
        """
        Renders the current Matplotlib figure to a Numpy/OpenCV image.
        This allows us to display the graph inside the OpenCV loop
        without crashing the thread.
        
        Returns:
            numpy.ndarray: BGR image array (OpenCV format) or None if cv2 unavailable
        """
        if not CV2_AVAILABLE:
            return None
        
        try:
            # 1. Draw the canvas (this renders the figure to the buffer)
            self.fig.canvas.draw()
            
            # 2. Convert to Numpy Array
            # Use buffer_rgba() for better compatibility, then convert to RGB
            buf = self.fig.canvas.buffer_rgba()
            ncols, nrows = self.fig.canvas.get_width_height()
            data = np.frombuffer(buf, dtype=np.uint8)
            image = data.reshape((nrows, ncols, 4))  # RGBA
            
            # 3. Convert RGBA -> RGB (drop alpha channel)
            image_rgb = image[:, :, :3]
            
            # 4. Convert RGB (Matplotlib) to BGR (OpenCV)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            return image_bgr
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            # Log error for debugging (but don't crash)
            print(f"[VISUALIZATION] render_to_image error: {e}")
            return None
    
    def show(self):
        """Display method - now handled by OpenCV in main.py."""
        pass  # Do nothing, main.py handles display now
    
    def refresh(self):
        """Refresh method - now handled by OpenCV in main.py."""
        pass  # Do nothing, main.py handles display now
    
    def on_state_changed(self, state):
        """
        Observer callback: React to OPU state changes.
        
        Args:
            state: Dictionary containing OPU state
        """
        s_score = state.get('s_score', 0.0)
        coherence = state.get('coherence', 0.0)
        maturity = state.get('maturity', 0.0)
        
        # Update visualization state
        self.update_state(s_score, coherence, maturity)
        
        # Redraw (if visualization is active)
        try:
            self.draw_cognitive_map()
            self.refresh()
        except Exception:
            # Silently fail if visualization window is closed
            pass

