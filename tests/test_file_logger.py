"""
Tests for File Logger (utils/file_logger.py)
"""

import pytest
import sys
import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open, call
from datetime import datetime

from utils.file_logger import FileLogger


class TestFileLogger:
    """Test suite for FileLogger class."""
    
    def test_init_success(self):
        """Test successful initialization."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            
            assert logger.log_file_path == log_path
            assert logger.enabled is True
            assert logger.log_file is not None
            assert logger.chain_to is None
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_init_failure(self):
        """Test initialization with invalid path."""
        invalid_path = '/nonexistent/directory/file.log'
        
        logger = FileLogger(log_file_path=invalid_path)
        
        assert logger.log_file_path == invalid_path
        assert logger.enabled is False
        assert logger.log_file is None
    
    def test_init_with_chain(self):
        """Test initialization with chaining."""
        mock_chain = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path, chain_to=mock_chain)
            
            assert logger.chain_to == mock_chain
            assert logger.original_stdout == mock_chain
            assert logger.original_stderr == mock_chain
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_write_enabled(self):
        """Test write when logger is enabled."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            test_message = "Test log message\n"
            
            logger.write(test_message)
            
            # Check file was written
            with open(log_path, 'r') as f:
                content = f.read()
                assert test_message in content
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_write_disabled(self):
        """Test write when logger is disabled."""
        mock_stdout = MagicMock()
        
        logger = FileLogger(log_file_path='/invalid/path.log')
        logger.original_stdout = mock_stdout
        
        test_message = "Test message\n"
        logger.write(test_message)
        
        # Should still write to original stdout
        mock_stdout.write.assert_called_with(test_message)
    
    def test_write_with_chain(self):
        """Test write with chaining."""
        mock_chain = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path, chain_to=mock_chain)
            test_message = "Chained message\n"
            
            logger.write(test_message)
            
            # Should write to both file and chain
            mock_chain.write.assert_called_with(test_message)
            mock_chain.flush.assert_called()
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_write_file_error(self):
        """Test write when file write fails."""
        mock_stdout = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            logger.original_stdout = mock_stdout
            
            # Simulate file write error
            with patch.object(logger.log_file, 'write', side_effect=IOError("Disk full")):
                test_message = "Test message\n"
                logger.write(test_message)
                
                # Should disable logger and notify stdout
                assert logger.enabled is False
                mock_stdout.write.assert_called()
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_flush(self):
        """Test flush method."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            mock_stdout = MagicMock()
            logger.original_stdout = mock_stdout
            
            logger.flush()
            
            mock_stdout.flush.assert_called()
            assert logger.log_file is not None
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_flush_disabled(self):
        """Test flush when logger is disabled."""
        mock_stdout = MagicMock()
        
        logger = FileLogger(log_file_path='/invalid/path.log')
        logger.original_stdout = mock_stdout
        
        logger.flush()
        
        mock_stdout.flush.assert_called()
    
    def test_flush_file_error(self):
        """Test flush when file flush fails."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            mock_stdout = MagicMock()
            logger.original_stdout = mock_stdout
            
            # Simulate file flush error
            with patch.object(logger.log_file, 'flush', side_effect=IOError()):
                logger.flush()
                
                # Should not raise exception
                mock_stdout.flush.assert_called()
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_close_enabled(self):
        """Test close when logger is enabled."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            logger.write("Test message\n")
            
            logger.close()
            
            assert logger.enabled is False
            assert logger.log_file is None
            
            # Check footer was written
            with open(log_path, 'r') as f:
                content = f.read()
                assert "Session ended" in content
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_close_disabled(self):
        """Test close when logger is already disabled."""
        logger = FileLogger(log_file_path='/invalid/path.log')
        
        # Should not raise exception
        logger.close()
        
        assert logger.enabled is False
        assert logger.log_file is None
    
    def test_close_file_error(self):
        """Test close when file operations fail."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            
            # Simulate file write/close error
            with patch.object(logger.log_file, 'write', side_effect=IOError()):
                logger.close()
                
                # Should not raise exception
                assert logger.enabled is False
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_get_log_path_enabled(self):
        """Test get_log_path when enabled."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            
            assert logger.get_log_path() == log_path
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_get_log_path_disabled(self):
        """Test get_log_path when disabled."""
        logger = FileLogger(log_file_path='/invalid/path.log')
        
        assert logger.get_log_path() is None
    
    def test_header_written(self):
        """Test that header is written on initialization."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            
            with open(log_path, 'r') as f:
                content = f.read()
                assert "OPU Debug Log" in content
                assert "Session started" in content
                assert "=" * 80 in content
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_multiple_writes(self):
        """Test multiple write operations."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            
            messages = ["Message 1\n", "Message 2\n", "Message 3\n"]
            for msg in messages:
                logger.write(msg)
            
            with open(log_path, 'r') as f:
                content = f.read()
                for msg in messages:
                    assert msg in content
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)
    
    def test_unicode_support(self):
        """Test that logger handles unicode characters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log', encoding='utf-8') as tmp:
            log_path = tmp.name
        
        try:
            logger = FileLogger(log_file_path=log_path)
            
            unicode_message = "Test with unicode: 你好 🌟\n"
            logger.write(unicode_message)
            
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "你好" in content
                assert "🌟" in content
            
            logger.close()
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

