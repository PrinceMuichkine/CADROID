"""
Logging utility for enhanced geometry system.
"""

from loguru import logger
import sys
import logging
from typing import Any, Union, Optional, Dict
from pathlib import Path
import os
from loguru._logger import Logger

def setup_logger(name: str) -> Logger:
    """Set up a logger instance for the given module."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger with default settings
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{line}</cyan> | <level>{message}</level>",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    logger.add(
        f"logs/{name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="10 days",
        compression="zip"
    )
    
    return logger

class CLGLogger:
    """Logger class for CAD sequence generation."""
    
    def __init__(self, name: str) -> None:
        """Initialize logger with module name."""
        self._logger = setup_logger(name)
    
    def debug(self, msg: str) -> None:
        self._logger.debug(msg)
    
    def info(self, msg: str) -> None:
        self._logger.info(msg)
    
    def warning(self, msg: str) -> None:
        self._logger.warning(msg)
    
    def error(self, msg: str) -> None:
        self._logger.error(msg)
    
    def critical(self, msg: str) -> None:
        self._logger.critical(msg)
    
    def level(self, level: str) -> None:
        self._logger.level(level)

    def configure_logger(self, verbose: bool = True) -> 'CLGLogger':
        """Configure logger with specified verbosity level."""
        logger.remove()
        
        if verbose:
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file.name}:{line}</cyan> | <level>{message}</level>",
                level="TRACE",
                colorize=True,
                enqueue=True,
                catch=True
            )
        else:
            logger.add(logging.NullHandler())

        return self
    
    def add_log_file(self, file_path: str) -> 'CLGLogger':
        """Add a file handler to the logger."""
        logger.add(
            file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file.relative}:{line} | {message}",
            level="DEBUG",
            rotation="100 MB"
        )
        return self

    def add_log_rotation(self, rotation_size: Union[str, int], backup_count: int) -> 'CLGLogger':
        """Configure log rotation settings."""
        # Set log rotation size and backup count
        logger.configure(
            handlers=[
                {
                    "sink": sys.stderr,
                    "level": "TRACE",
                    "diagnose": True,
                    "serialize": True
                },
                {
                    "sink": logging.FileHandler,
                    "level": "DEBUG",
                    "rotation": rotation_size,
                    "backtrace": True,
                    "enqueue": True,
                    "catch": True,
                    "diagnose": True,
                    "serialize": True
                }
            ],
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

    def _format_message(self, message: str, *args: Any, **kwargs: Any) -> str:
        """Format message with args and kwargs."""
        try:
            if args:
                message = message % args
            if kwargs:
                message = message % kwargs
            return str(message)
        except Exception as e:
            return f"Error formatting message: {message} with args={args} kwargs={kwargs}. Error: {str(e)}"
