"""
Tests for macOS Log Window (utils/log_window_macos.py)
"""

import pytest
import sys
import threading
import queue
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime

# Mock tkinter before importing
mock_tk = MagicMock()
mock_scrolledtext = MagicMock()

sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.scrolledtext'] = mock_scrolledtext

# Mock platform
with patch('platform.system', return_value='Darwin'):
    from utils.log_window_macos import OPULogWindow, _check_tkinter


class TestOPULogWindowMacOS:
    """Test suite for OPULogWindow macOS class."""
    
    def test_check_tkinter_available(self):
        """Test tkinter availability check."""
        with patch('utils.log_window_macos._tkinter_available', None):
            with patch('builtins.__import__', return_value=mock_tk):
                result = _check_tkinter()
                assert result is True
    
    def test_check_tkinter_unavailable(self):
        """Test tkinter unavailability."""
        with patch('utils.log_window_macos._tkinter_available', None):
            with patch('builtins.__import__', side_effect=ImportError()):
                result = _check_tkinter()
                assert result is False
    
    def test_init_basic(self):
        """Test basic initialization."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        
        # Mock threading
        with patch('threading.Thread') as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            # Mock the _init_tkinter to complete quickly
            with patch.object(OPULogWindow, '_init_tkinter') as mock_init:
                def set_ready():
                    window = sys.modules['utils.log_window_macos'].OPULogWindow.__new__(OPULogWindow)
                    window.window_ready = threading.Event()
                    window.window_ready.set()
                    window.init_error = None
                    window.root = mock_root
                    window.text_widget = mock_text
                    return window
                
                with patch.object(OPULogWindow, '__new__', side_effect=set_ready):
                    window = OPULogWindow(title="Test Log", width=600, height=400)
                    
                    assert window.title == "Test Log"
                    assert window.width == 600
                    assert window.height == 400
                    assert window.running is True
                    assert window.message_queue is not None
    
    def test_init_tkinter_success(self):
        """Test successful tkinter initialization."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.title = "Test"
        window.width = 800
        window.height = 600
        window.window_ready = threading.Event()
        window.init_error = None
        
        with patch('utils.log_window_macos._tk', mock_tk):
            with patch('utils.log_window_macos._scrolledtext', mock_scrolledtext):
                window._init_tkinter()
                
                assert window.window_ready.is_set()
                assert window.init_error is None
                mock_tk.Tk.assert_called()
    
    def test_init_tkinter_exception(self):
        """Test tkinter initialization with exception."""
        window = OPULogWindow.__new__(OPULogWindow)
        window.title = "Test"
        window.width = 800
        window.height = 600
        window.window_ready = threading.Event()
        window.init_error = None
        
        with patch('utils.log_window_macos._tk') as mock_tk_module:
            mock_tk_module.Tk.side_effect = Exception("Tk error")
            
            window._init_tkinter()
            
            assert window.window_ready.is_set()
            assert window.init_error is not None
    
    def test_write_message(self):
        """Test writing a message."""
        mock_root = MagicMock()
        mock_tk.Tk.return_value = mock_root
        
        mock_text = MagicMock()
        mock_scrolledtext.ScrolledText.return_value = mock_text
        
        with patch('threading.Thread'):
            with patch.object(OPULogWindow, '_init_tkinter'):
                window = OPULogWindow.__new__(OPULogWindow)
                window.running = True
                window.message_queue = queue.Queue()
                window.window_ready = threading.Event()
                window.window_ready.set()
                window.init_error = None
                window.root = mock_root
                window.text_widget = mock_text
                
                test_message = "Test message\n"
                window.write(test_message)
                
                assert not window.message_queue.empty()
                message, tag = window.message_queue.get_nowait()
                assert message == test_message
    
    def test_write_when_stopped(self):
        """Test write when window is stopped."""
        window = OPULogWindow.__new__(OPULogWindow)
        window.running = False
        window.message_queue = queue.Queue()
        
        window.write("Message\n")
        
        assert window.message_queue.empty()
    
    def test_flush(self):
        """Test flush method."""
        window = OPULogWindow.__new__(OPULogWindow)
        
        # Should not raise exception
        window.flush()
    
    def test_process_queue(self):
        """Test processing message queue."""
        mock_root = MagicMock()
        mock_text = MagicMock()
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.text_widget = mock_text
        window.message_queue = queue.Queue()
        window.message_queue.put(("Message\n", 'normal'))
        window.root = mock_root
        
        window.process_queue()
        
        assert window.message_queue.empty()
    
    def test_process_queue_no_widget(self):
        """Test queue processing when widget not available."""
        window = OPULogWindow.__new__(OPULogWindow)
        window.message_queue = queue.Queue()
        window.message_queue.put(("Message\n", 'normal'))
        
        # Should handle gracefully
        window.process_queue()
    
    def test_append_text(self):
        """Test appending text."""
        mock_root = MagicMock()
        mock_text = MagicMock()
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.root = mock_root
        window.text_widget = mock_text
        
        window._append_text("Message\n", 'normal')
        
        mock_root.after.assert_called()
    
    def test_do_append(self):
        """Test actual text appending."""
        mock_root = MagicMock()
        mock_text = MagicMock()
        mock_text.index.return_value = '100.0'
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.root = mock_root
        window.text_widget = mock_text
        
        with patch('utils.log_window_macos._tk', mock_tk):
            window._do_append("Message\n", 'normal')
            
            mock_text.config.assert_called()
            mock_text.insert.assert_called()
    
    def test_do_append_text_limit(self):
        """Test text limit in do_append."""
        mock_root = MagicMock()
        mock_text = MagicMock()
        mock_text.index.return_value = '15000.0'  # Over limit
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.root = mock_root
        window.text_widget = mock_text
        
        with patch('utils.log_window_macos._tk', mock_tk):
            window._do_append("Message\n", 'normal')
            
            mock_text.delete.assert_called()
    
    def test_on_closing(self):
        """Test window close handler."""
        mock_root = MagicMock()
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.running = True
        window.root = mock_root
        window.original_stdout = sys.stdout
        window.original_stderr = sys.stderr
        
        window.on_closing()
        
        assert window.running is False
        assert sys.stdout == window.original_stdout
        mock_root.destroy.assert_called()
    
    def test_start(self):
        """Test start method (no-op for macOS version)."""
        window = OPULogWindow.__new__(OPULogWindow)
        
        # Should not raise exception
        window.start()
    
    def test_update(self):
        """Test update method."""
        mock_text = MagicMock()
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.text_widget = mock_text
        window.message_queue = queue.Queue()
        
        window.update()
        
        # Should process queue
        window.process_queue()
    
    def test_stop(self):
        """Test stop method."""
        mock_root = MagicMock()
        
        window = OPULogWindow.__new__(OPULogWindow)
        window.running = True
        window.root = mock_root
        window.original_stdout = sys.stdout
        window.original_stderr = sys.stderr
        
        window.stop()
        
        assert window.running is False
        assert sys.stdout == window.original_stdout
        mock_root.after.assert_called()
    
    def test_stop_no_root(self):
        """Test stop when root doesn't exist."""
        window = OPULogWindow.__new__(OPULogWindow)
        window.running = True
        window.original_stdout = sys.stdout
        window.original_stderr = sys.stderr
        
        # Should not raise exception
        window.stop()
    
    def test_init_timeout(self):
        """Test initialization timeout."""
        with patch('threading.Thread') as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            window = OPULogWindow.__new__(OPULogWindow)
            window.window_ready = threading.Event()
            window.init_error = None
            
            # Simulate timeout
            with patch.object(window.window_ready, 'wait', return_value=False):
                with pytest.raises(RuntimeError, match="timeout"):
                    OPULogWindow.__init__(window, title="Test")
    
    def test_init_with_error(self):
        """Test initialization with error."""
        with patch('threading.Thread') as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            window = OPULogWindow.__new__(OPULogWindow)
            window.window_ready = threading.Event()
            window.window_ready.set()
            window.init_error = Exception("Init error")
            
            with pytest.raises(Exception, match="Init error"):
                OPULogWindow.__init__(window, title="Test")

