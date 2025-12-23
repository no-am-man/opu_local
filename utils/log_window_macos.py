"""
OPU Log Window: macOS-specific implementation using threading workaround.

This module provides a macOS-compatible log window that initializes tkinter
in a separate thread to avoid the NSApplication crash issue.
"""

import os
import platform
import threading
import queue
import sys
from datetime import datetime

# Set environment variable BEFORE any tkinter imports
if platform.system() == 'Darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Import tkinter only when needed (lazy import)
_tkinter_available = None
_tk = None
_scrolledtext = None


def _check_tkinter():
    """Check if tkinter is available and import it."""
    global _tkinter_available, _tk, _scrolledtext
    if _tkinter_available is None:
        try:
            import tkinter as tk
            from tkinter import scrolledtext
            _tk = tk
            _scrolledtext = scrolledtext
            _tkinter_available = True
        except Exception:
            _tkinter_available = False
    return _tkinter_available


class OPULogWindow:
    """
    A dedicated window for displaying OPU log messages in real-time.
    macOS-compatible version that initializes tkinter in a separate thread.
    """
    
    def __init__(self, title="OPU Log", width=800, height=600):
        """
        Initialize the log window.
        
        Args:
            title: Window title
            width: Window width in pixels
            height: Window height in pixels
        """
        if not _check_tkinter():
            raise ImportError("tkinter not available")
        
        self.title = title
        self.width = width
        self.height = height
        
        # Queue for thread-safe message passing
        self.message_queue = queue.Queue()
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Flag to track if window is ready
        self.window_ready = threading.Event()
        self.init_error = None
        
        # Start tkinter in a separate thread (macOS workaround)
        self.tk_thread = threading.Thread(target=self._init_tkinter, daemon=True)
        self.tk_thread.start()
        
        # Wait for window to be ready (with timeout)
        if not self.window_ready.wait(timeout=2.0):
            raise RuntimeError("Log window initialization timeout")
        
        if self.init_error:
            raise self.init_error
        
        # Redirect stdout/stderr after window is ready
        sys.stdout = self
        sys.stderr = self
        
        # Start message processing
        self.running = True
        
        # Add initial message
        self.write("=" * 80 + "\n", 'normal')
        self.write("OPU Log Window - Real-time Log Viewer\n", 'info')
        self.write("=" * 80 + "\n", 'normal')
        self.write(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n", 'timestamp')
    
    def _init_tkinter(self):
        """Initialize tkinter in this thread (macOS workaround)."""
        try:
            # Create Tk root in this thread
            self.root = _tk.Tk()
            self.root.title(self.title)
            self.root.geometry(f"{self.width}x{self.height}")
            
            # Create scrolled text widget
            self.text_widget = _scrolledtext.ScrolledText(
                self.root,
                wrap=_tk.WORD,
                font=('Courier', 10),
                bg='#1e1e1e',
                fg='#d4d4d4',
                insertbackground='#d4d4d4',
                selectbackground='#264f78',
                state=_tk.DISABLED
            )
            self.text_widget.pack(fill=_tk.BOTH, expand=True, padx=5, pady=5)
            
            # Configure text tags
            self.text_widget.tag_config('timestamp', foreground='#808080')
            self.text_widget.tag_config('error', foreground='#f48771')
            self.text_widget.tag_config('warning', foreground='#dcdcaa')
            self.text_widget.tag_config('info', foreground='#4ec9b0')
            self.text_widget.tag_config('normal', foreground='#d4d4d4')
            
            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Signal that window is ready
            self.window_ready.set()
            
            # Start tkinter main loop in this thread
            self.root.mainloop()
        except Exception as e:
            self.init_error = e
            self.window_ready.set()
    
    def write(self, message, tag='normal'):
        """
        Write a message to the log window (thread-safe).
        
        Args:
            message: Message to write
            tag: Text tag for styling
        """
        if not self.running:
            return
        
        # Put message in queue for thread-safe processing
        self.message_queue.put((message, tag))
    
    def flush(self):
        """Flush method required for stdout/stderr redirection."""
        pass
    
    def process_queue(self):
        """Process messages from the queue (called from main thread)."""
        if not hasattr(self, 'text_widget') or not self.text_widget:
            return
        
        try:
            while True:
                try:
                    message, tag = self.message_queue.get_nowait()
                    self._append_text(message, tag)
                except queue.Empty:
                    break
        except Exception as e:
            # Fallback to original stdout if something goes wrong
            if hasattr(self, 'original_stdout'):
                self.original_stdout.write(f"Log window error: {e}\n")
    
    def _append_text(self, message, tag='normal'):
        """Append text to the widget (must be called from tkinter thread)."""
        if not hasattr(self, 'text_widget') or not self.text_widget:
            return
        
        try:
            # Schedule update in tkinter thread
            self.root.after(0, lambda: self._do_append(message, tag))
        except Exception:
            pass
    
    def _do_append(self, message, tag):
        """Actually append text (called in tkinter thread)."""
        try:
            self.text_widget.config(state=_tk.NORMAL)
            self.text_widget.insert(_tk.END, message, tag)
            self.text_widget.see(_tk.END)
            self.text_widget.config(state=_tk.DISABLED)
            
            # Limit text to prevent memory issues
            if int(self.text_widget.index('end-1c').split('.')[0]) > 10000:
                self.text_widget.delete('1.0', '1000.0')
        except Exception:
            pass
    
    def on_closing(self):
        """Handle window close event."""
        self.running = False
        # Restore stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Destroy window
        if hasattr(self, 'root'):
            self.root.destroy()
    
    def start(self):
        """Start the log window (already started in __init__)."""
        pass
    
    def update(self):
        """Update the log window (process message queue)."""
        self.process_queue()
    
    def stop(self):
        """Stop the log window and restore stdout/stderr."""
        self.running = False
        if hasattr(self, 'root'):
            try:
                self.root.after(0, self.root.destroy)
            except Exception:
                pass
        # Restore stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

