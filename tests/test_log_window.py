"""
Tests for Log Window (utils/log_window.py)
"""

import pytest
import sys
import queue
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime

# Mock tkinter before importing log_window
mock_tk = MagicMock()
mock_scrolledtext = MagicMock()
mock_filedialog = MagicMock()
mock_messagebox = MagicMock()

sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.scrolledtext'] = mock_scrolledtext
sys.modules['tkinter.filedialog'] = mock_filedialog
sys.modules['tkinter.messagebox'] = mock_messagebox

from utils.log_window import OPULogWindow


class TestOPULogWindow:
    """Test suite for OPULogWindow class."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        
        window = OPULogWindow(title="Test Log", width=600, height=400)
        
        assert window.title == "Test Log" or window.root.title() == "Test Log"
        assert window.running is True
        assert window.message_queue is not None
        assert window.original_stdout == sys.__stdout__
        mock_tk.Tk.assert_called()
    
    def test_init_defaults(self):
        """Test initialization with default parameters."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        
        window = OPULogWindow()
        
        assert window.running is True
        mock_root.geometry.assert_called()
    
    def test_write_message(self):
        """Test writing a message."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        test_message = "Test log message\n"
        window.write(test_message)
        
        # Message should be in queue
        assert not window.message_queue.empty()
        message, tag = window.message_queue.get_nowait()
        assert message == test_message
        assert tag == 'normal'
    
    def test_write_with_tag(self):
        """Test writing with a specific tag."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.write("Error message\n", tag='error')
        
        message, tag = window.message_queue.get_nowait()
        assert tag == 'error'
    
    def test_write_when_stopped(self):
        """Test write when window is stopped."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        window.running = False
        
        window.write("Message\n")
        
        # Queue should be empty (message not added)
        assert window.message_queue.empty()
    
    def test_flush(self):
        """Test flush method."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        # Should not raise exception
        window.flush()
    
    def test_process_queue(self):
        """Test processing message queue."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.index.return_value = '100.0'
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        # Add messages to queue
        window.message_queue.put(("Message 1\n", 'normal'))
        window.message_queue.put(("Message 2\n", 'info'))
        
        window.process_queue()
        
        # Messages should be processed
        assert window.message_queue.empty()
        mock_text.insert.assert_called()
    
    def test_process_queue_empty(self):
        """Test processing empty queue."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        # Should not raise exception
        window.process_queue()
    
    def test_process_queue_exception(self):
        """Test queue processing with exception."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.insert.side_effect = Exception("Widget error")
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.message_queue.put(("Message\n", 'normal'))
        
        # Should handle exception gracefully
        window.process_queue()
        assert window.original_stdout.write.called
    
    def test_append_text_auto_tag_detection(self):
        """Test automatic tag detection from message content."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.index.return_value = '100.0'
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        # Test error detection
        window._append_text("[ERROR] Something went wrong\n", 'normal')
        mock_text.insert.assert_called()
        
        # Test warning detection
        window._append_text("[WARNING] This is a warning\n", 'normal')
        
        # Test info detection
        window._append_text("[INFO] Information message\n", 'normal')
    
    def test_copy_all(self):
        """Test copying all content."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.get.return_value = "All log content\n"
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.copy_all()
        
        mock_text.get.assert_called_with('1.0', mock_tk.END)
        mock_root.clipboard_clear.assert_called()
        mock_root.clipboard_append.assert_called()
    
    def test_copy_all_exception(self):
        """Test copy_all with exception."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.get.side_effect = Exception("Copy error")
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.copy_all()
        
        mock_messagebox.showerror.assert_called()
    
    def test_copy_selected(self):
        """Test copying selected text."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.tag_ranges.return_value = [1, 2]  # Has selection
        mock_text.get.return_value = "Selected text\n"
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.copy_selected()
        
        mock_text.get.assert_called_with(mock_tk.SEL_FIRST, mock_tk.SEL_LAST)
        mock_root.clipboard_clear.assert_called()
    
    def test_copy_selected_no_selection(self):
        """Test copy_selected when nothing is selected."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.tag_ranges.return_value = []  # No selection
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.copy_selected()
        
        mock_messagebox.showinfo.assert_called()
    
    def test_save_to_file(self):
        """Test saving to file."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.get.return_value = "Log content\n"
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        mock_filedialog.asksaveasfilename.return_value = "/tmp/test.log"
        
        with patch('builtins.open', mock_open()) as mock_file:
            window.save_to_file()
            
            mock_file.assert_called()
            mock_messagebox.showinfo.assert_called()
    
    def test_save_to_file_cancelled(self):
        """Test save when user cancels file dialog."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        mock_filedialog.asksaveasfilename.return_value = ""  # User cancelled
        
        window.save_to_file()
        
        # Should not try to save
        mock_text.get.assert_not_called()
    
    def test_save_to_file_exception(self):
        """Test save_to_file with exception."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.get.side_effect = Exception("Save error")
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        mock_filedialog.asksaveasfilename.return_value = "/tmp/test.log"
        
        window.save_to_file()
        
        mock_messagebox.showerror.assert_called()
    
    def test_on_closing(self):
        """Test window close handler."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        original_stdout = sys.stdout
        window.on_closing()
        
        assert window.running is False
        assert sys.stdout == original_stdout
        mock_root.destroy.assert_called()
    
    def test_start(self):
        """Test start method."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.start()
        
        mock_root.update_idletasks.assert_called()
        mock_root.deiconify.assert_called()
    
    def test_start_exception(self):
        """Test start with exception."""
        mock_root = MagicMock()
        mock_root.update_idletasks.side_effect = Exception("Start error")
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.start()
        
        # Should handle exception gracefully
        assert window.original_stdout.write.called
    
    def test_update(self):
        """Test update method."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.index.return_value = '100.0'
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window.message_queue.put(("Message\n", 'normal'))
        window.update()
        
        mock_root.update_idletasks.assert_called()
    
    def test_update_when_stopped(self):
        """Test update when window is stopped."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        window.running = False
        
        window.update()
        
        # Should return early
        mock_root.update_idletasks.assert_not_called()
    
    def test_stop(self):
        """Test stop method."""
        mock_root = MagicMock()
        mock_root.winfo_exists.return_value = True
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        original_stdout = sys.stdout
        window.stop()
        
        assert window.running is False
        assert sys.stdout == original_stdout
        mock_root.after.assert_called()
    
    def test_text_limit(self):
        """Test text limit (10000 lines)."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_text.index.return_value = '15000.0'  # Over limit
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        window._append_text("Message\n", 'normal')
        
        # Should delete old lines
        mock_text.delete.assert_called()
    
    def test_stdout_redirection(self):
        """Test that stdout is redirected."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        window = OPULogWindow()
        
        # stdout should be redirected to window
        assert sys.stdout == window
        assert sys.stderr == window

