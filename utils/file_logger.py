"""
File Logger: Writes all OPU output to a log file for debugging.
"""

import sys
import os
from datetime import datetime


class FileLogger:
    """
    A file logger that writes all stdout/stderr to a log file.
    Can be used alongside the log window or independently.
    """
    
    def __init__(self, log_file_path='opu_debug.log', chain_to=None):
        """
        Initialize the file logger.
        
        Args:
            log_file_path: Path to the log file
            chain_to: Another stdout-like object to chain to (for log window compatibility)
        """
        self.log_file_path = log_file_path
        self.chain_to = chain_to  # For chaining with log window
        self.original_stdout = sys.stdout if chain_to is None else chain_to
        self.original_stderr = sys.stderr if chain_to is None else chain_to
        self.log_file = None
        self.enabled = False
        
        # Try to open the log file
        try:
            self.log_file = open(log_file_path, 'a', encoding='utf-8')
            self.enabled = True
            
            # Write header
            self.log_file.write("\n" + "=" * 80 + "\n")
            self.log_file.write(f"OPU Debug Log - Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 80 + "\n\n")
            self.log_file.flush()
            
        except Exception as e:
            if self.original_stdout:
                self.original_stdout.write(f"[OPU] Warning: Could not open log file '{log_file_path}': {e}\n")
            self.enabled = False
    
    def write(self, message):
        """
        Write a message to both the log file and chain to the next stdout (log window or original).
        
        Args:
            message: Message to write
        """
        # Write to log file FIRST
        if self.enabled and self.log_file:
            try:
                self.log_file.write(message)
                self.log_file.flush()  # Ensure immediate write for debugging
            except Exception as e:
                # If file write fails, try to notify via original stdout
                if self.original_stdout:
                    self.original_stdout.write(f"[OPU] Log file write error: {e}\n")
                    self.original_stdout.flush()
                self.enabled = False
        
        # Chain to next stdout (log window or original)
        if self.original_stdout:
            self.original_stdout.write(message)
            self.original_stdout.flush()
    
    def flush(self):
        """Flush both stdout/stderr and log file."""
        if self.original_stdout:
            self.original_stdout.flush()
        if self.enabled and self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass
    
    def close(self):
        """Close the log file and restore original stdout/stderr."""
        if self.enabled and self.log_file:
            try:
                self.log_file.write("\n" + "=" * 80 + "\n")
                self.log_file.write(f"OPU Debug Log - Session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write("=" * 80 + "\n\n")
                self.log_file.close()
            except Exception:
                pass
        
        self.enabled = False
        self.log_file = None
    
    def get_log_path(self):
        """Get the path to the log file."""
        return self.log_file_path if self.enabled else None

