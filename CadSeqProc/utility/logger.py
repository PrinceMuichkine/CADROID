"""
Logging utility for enhanced geometry system.
"""

from loguru import logger
import sys
import logging
from typing import Any, Union, Optional

class CLGLogger:
    """Logger class with enhanced configuration options."""
    
    def __init__(self) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)

    def configure_logger(self, verbose: bool = True) -> 'CLGLogger':
        """Configure logger with specified verbosity level."""
        logger.remove()
        
        if verbose:
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file.name}:{line}</cyan> | <level><level>{message}</level></level>",
                level="TRACE",
                colorize=True,
                backtrace=True,
                diagnose=True,
                enqueue=True,
                catch=True)
        else:
            # Disable logging by adding a null handler
            logger.add(logging.NullHandler())

        return self
    
    def add_log_file(self, file_path: str) -> 'CLGLogger':
        """Add a file handler to the logger."""
        # Add a log file handler
        logger.add(
            file_path,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file.relative}:{line}</cyan> | <level><level>{message}</level></level>",
            level="DEBUG",
            rotation="100 MB"
        )

        return self
    
    def add_log_rotation(self, rotation_size: Union[str, int], backup_count: int) -> 'CLGLogger':
        """Configure log rotation settings."""
        # Set log rotation size and backup count
        logger.configure(
            handlers=[{"sink": sys.stderr, "level": "TRACE", "diagnose": True},
                      {"sink": logging.FileHandler, "level": "DEBUG", "rotation": rotation_size, "backtrace": True, "enqueue": True, "catch": True, "diagnose": True}],
            backtrace=True,
            diagnose=True,
            rotation=rotation_size,
            compression="zip",
            retention=backup_count
        )

        return self

    def add_log_level(self, level: Union[str, int]) -> 'CLGLogger':
        """Set the logging level."""
        # Set the log level
        logger.level(level)

        return self
