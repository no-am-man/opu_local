"""
The Face: Matplotlib HUD for Cognitive Map visualization.
Shows pulse size (s_score), color (Tension), and shape integrity (Coherence).
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from config import VISUALIZATION_UPDATE_RATE, WINDOW_SIZE


class CognitiveMapVisualizer:
    """
    Real-time visualization of the OPU's cognitive state.
    """
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=WINDOW_SIZE)
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
    
    def update_state(self, s_score, coherence, maturity):
        """
        Update the visualization with new cognitive state.
        
        Args:
            s_score: surprise score (pulse size)
            coherence: coherence value (shape integrity)
            maturity: maturity index (0.0 to 1.0)
        """
        self.current_s_score = s_score
        self.current_coherence = coherence
        self.current_maturity = maturity
        
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
        # Clear previous frame
        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('OPU Cognitive Map', fontsize=16, fontweight='bold')
        
        # Calculate pulse size from s_score
        # Normalize s_score to reasonable radius (0.1 to 1.5)
        pulse_radius = 0.1 + min(self.current_s_score / 5.0, 1.4)
        
        # Calculate color based on tension (s_score)
        # High tension = red, low tension = blue
        tension = min(self.current_s_score / 10.0, 1.0)
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
        
        # Draw text info
        info_text = (
            f"s_score: {self.current_s_score:.2f}\n"
            f"Coherence: {self.current_coherence:.2f}\n"
            f"Maturity: {self.current_maturity:.2f}\n"
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
    
    def show(self):
        """Display the visualization window."""
        plt.ion()  # Turn on interactive mode
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)  # Small pause to ensure window is ready
    
    def refresh(self):
        """Refresh the display (call after draw_cognitive_map)."""
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update

