"""
OPU Log Window: Real-time log viewer for OPU output.
Displays all log messages in a dedicated GUI window.
"""

# CRITICAL: Set environment variable BEFORE importing tkinter on macOS
# This prevents the NSApplication macOSVersion crash on Python 3.13+
import os
import platform
if platform.system() == 'Darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import queue
import sys
from datetime import datetime


class OPULogWindow:
    """
    A dedicated window for displaying OPU log messages in real-time.
    Captures stdout/stderr and displays messages with timestamps.
    """
    
    def __init__(self, title="OPU Log", width=800, height=600):
        """
        Initialize the log window.
        
        Args:
            title: Window title
            width: Window width in pixels
            height: Window height in pixels
        """
        # Environment variable is already set at module level (before tkinter import)
        # Now create the Tk root
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        
        # Create button frame
        self.button_frame = tk.Frame(self.root, bg='#1e1e1e')
        self.button_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Copy All button
        self.copy_all_btn = tk.Button(
            self.button_frame,
            text="Copy All",
            command=self.copy_all,
            bg='#264f78',
            fg='#d4d4d4',
            activebackground='#3a6ba0',
            activeforeground='#ffffff',
            relief=tk.FLAT,
            padx=10,
            pady=2
        )
        self.copy_all_btn.pack(side=tk.LEFT, padx=2)
        
        # Copy Selected button
        self.copy_selected_btn = tk.Button(
            self.button_frame,
            text="Copy Selected",
            command=self.copy_selected,
            bg='#264f78',
            fg='#d4d4d4',
            activebackground='#3a6ba0',
            activeforeground='#ffffff',
            relief=tk.FLAT,
            padx=10,
            pady=2
        )
        self.copy_selected_btn.pack(side=tk.LEFT, padx=2)
        
        # Save to File button
        self.save_btn = tk.Button(
            self.button_frame,
            text="Save to File",
            command=self.save_to_file,
            bg='#264f78',
            fg='#d4d4d4',
            activebackground='#3a6ba0',
            activeforeground='#ffffff',
            relief=tk.FLAT,
            padx=10,
            pady=2
        )
        self.save_btn.pack(side=tk.LEFT, padx=2)
        
        # Create scrolled text widget (ENABLED for selection/copying)
        self.text_widget = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=('Courier', 10),
            bg='#1e1e1e',  # Dark background
            fg='#d4d4d4',  # Light text
            insertbackground='#d4d4d4',
            selectbackground='#264f78',
            state=tk.NORMAL  # ENABLED for text selection
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind Ctrl+C for copying
        self.text_widget.bind('<Control-c>', lambda e: self.copy_selected())
        self.text_widget.bind('<Command-c>', lambda e: self.copy_selected())  # macOS
        
        # Configure text tags for different log levels
        self.text_widget.tag_config('timestamp', foreground='#808080')
        self.text_widget.tag_config('error', foreground='#f48771')
        self.text_widget.tag_config('warning', foreground='#dcdcaa')
        self.text_widget.tag_config('info', foreground='#4ec9b0')
        self.text_widget.tag_config('normal', foreground='#d4d4d4')
        
        # Queue for thread-safe message passing
        self.message_queue = queue.Queue()
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Redirect stdout/stderr
        sys.stdout = self
        sys.stderr = self
        
        # Start message processing
        self.running = True
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Add initial message
        self.write("=" * 80 + "\n", 'normal')
        self.write("OPU Log Window - Real-time Log Viewer\n", 'info')
        self.write("=" * 80 + "\n", 'normal')
        self.write(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n", 'timestamp')
    
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
        try:
            while True:
                try:
                    message, tag = self.message_queue.get_nowait()
                    self._append_text(message, tag)
                except queue.Empty:
                    break
        except Exception as e:
            # Fallback to original stdout if something goes wrong
            self.original_stdout.write(f"Log window error: {e}\n")
    
    def _append_text(self, message, tag='normal'):
        """Append text to the widget (must be called from main thread)."""
        try:
            # Widget is always in NORMAL state now for selection
            # Auto-detect log level from message
            if not tag or tag == 'normal':
                if '[ERROR]' in message or 'Error' in message:
                    tag = 'error'
                elif '[WARNING]' in message or 'Warning' in message:
                    tag = 'warning'
                elif '[INFO]' in message or '[OPU]' in message or '[AFL]' in message or '[PERSISTENCE]' in message or '[GENESIS]' in message or '[PHONEME]' in message or '[EVOLUTION]' in message or '[VISION]' in message or '[DETECTION]' in message or '[CYCLE]' in message:
                    tag = 'info'
                else:
                    tag = 'normal'
            
            # Insert message
            self.text_widget.insert(tk.END, message, tag)
            
            # Auto-scroll to bottom
            self.text_widget.see(tk.END)
            
            # Limit text to prevent memory issues (keep last 10000 lines)
            lines = int(self.text_widget.index('end-1c').split('.')[0])
            if lines > 10000:
                self.text_widget.delete('1.0', f'{lines - 10000}.0')
        except Exception as e:
            # Fallback to original stdout
            self.original_stdout.write(f"Log append error: {e}\n")
    
    def copy_all(self):
        """Copy all log content to clipboard."""
        try:
            content = self.text_widget.get('1.0', tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.root.update()  # Update clipboard
            # Show brief feedback
            self.text_widget.insert(tk.END, "\n[LOG] All content copied to clipboard\n", 'info')
            self.text_widget.see(tk.END)
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy: {e}")
    
    def copy_selected(self):
        """Copy selected text to clipboard."""
        try:
            if self.text_widget.tag_ranges(tk.SEL):
                content = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                self.root.update()  # Update clipboard
            else:
                messagebox.showinfo("No Selection", "Please select text to copy.")
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy: {e}")
    
    def save_to_file(self):
        """Save all log content to a file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")],
                title="Save OPU Log"
            )
            if filename:
                content = self.text_widget.get('1.0', tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Saved", f"Log saved to:\n{filename}")
                self.text_widget.insert(tk.END, f"\n[LOG] Saved to: {filename}\n", 'info')
                self.text_widget.see(tk.END)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {e}")
    
    def on_closing(self):
        """Handle window close event."""
        self.running = False
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.root.destroy()
    
    def start(self):
        """Start the log window (make it visible)."""
        try:
            # Make window visible
            self.root.update_idletasks()
            self.root.deiconify()
        except Exception as e:
            self.original_stdout.write(f"Log window start error: {e}\n")
    
    def update(self):
        """Update the log window (call this periodically from main loop)."""
        """Process pending messages and update the window."""
        if not self.running:
            return
        
        try:
            # Process message queue
            self.process_queue()
            
            # Update tkinter window (non-blocking)
            self.root.update_idletasks()
        except Exception as e:
            # Silently fail if window was closed
            if 'application has been destroyed' not in str(e).lower():
                self.original_stdout.write(f"Log window update error: {e}\n")
    
    def stop(self):
        """Stop the log window and restore stdout/stderr."""
        self.running = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, self.on_closing)
        except Exception:
            pass
