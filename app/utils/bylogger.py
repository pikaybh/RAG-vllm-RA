import os
import logging
from typing import Optional
from abc import ABC, abstractmethod

from .oses import extfomatter


LOG_DIR = "logs"
EXT = ".log"



class LoggerConfig(ABC):
    """
    Abstract Base Class for logger configuration.
    Defines the required properties for configuring a logger.
    """

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Specifies a prefix (directory) for the log file path."""
        pass

    @property
    @abstractmethod
    def ext(self) -> str:
        """Specifies the extension for the log file (e.g., .log)."""
        pass

    @property
    @abstractmethod
    def file_format(self) -> str:
        """Specifies the format for log messages in the log file."""
        pass

    @property
    @abstractmethod
    def file_encoding(self) -> str:
        """Specifies the encoding to use for the log file."""
        pass

    @property
    @abstractmethod
    def file_handler_level(self) -> int:
        """Defines the logging level for the file handler."""
        pass

    @property
    @abstractmethod
    def stream_format(self) -> str:
        """Specifies the format for log messages in the stream handler."""
        pass

    @property
    @abstractmethod
    def stream_handler_level(self) -> int:
        """Defines the logging level for the stream handler."""
        pass



class DefaultLoggerConfig(LoggerConfig):
    """
    Default configuration for logger.
    Provides default values for file and stream handlers.
    """

    @property
    def prefix(self) -> str:
        return LOG_DIR

    @property
    def ext(self) -> str:
        return EXT

    @property
    def file_format(self) -> str:
        return r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'

    @property
    def file_encoding(self) -> str:
        return 'utf-8-sig'

    @property
    def file_handler_level(self) -> int:
        return logging.DEBUG

    @property
    def stream_format(self) -> str:
        return r'%(message)s'

    @property
    def stream_handler_level(self) -> int:
        return logging.INFO



class CustomLogger(logging.Logger):
    """
    Custom Logger class to expose internal variables.
    Extends the logging.Logger class to include attributes for configuration details.
    """
    def __init__(self, name: str, config: LoggerConfig, file_path: str):
        super().__init__(name)
        self.prefix = config.prefix  # Prefix for path to the log file
        self.ext = config.ext  # Extension of log file
        self.file_path = file_path  # Full path to the log file
        self.file_format = config.file_format  # Format for log file messages
        self.file_encoding = config.file_encoding  # Encoding used for the log file
        self.file_handler_level = config.file_handler_level  # Logging level for file handler
        self.stream_format = config.stream_format  # Format for stream handler messages
        self.stream_handler_level = config.stream_handler_level  # Logging level for stream handler


def get_logger(name: str, root: str, config: Optional[LoggerConfig] = DefaultLoggerConfig()) -> CustomLogger:
    """
    Creates and configures a logger with both file and stream handlers.

    Args:
        name (str): The name of the logger, typically used to identify the logging source.
        root (str): The root directory where the log file will be stored.
        config (LoggerConfig): Configuration for the logger. Defaults to DefaultLoggerConfig().

    Returns:
        CustomLogger: Configured logger instance with exposed attributes for customization.

    Example:
        >>> config = DefaultLoggerConfig()
        >>> logger = get_logger("my_logger", "./", config)
        >>> logger.info("This is an info message")
        >>> logger.debug("This is a debug message")
        >>> print(logger.file_path)  # Access internal variable
    """

    file_name = name + extfomatter(config.ext)
    file_path = os.path.join(root, config.prefix, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create CustomLogger instance
    logger = CustomLogger(name, config, file_path)
    logger.setLevel(logging.DEBUG)

    # File Handler
    file_handler = logging.FileHandler(logger.file_path, encoding=config.file_encoding)
    file_handler.setLevel(config.file_handler_level)
    file_handler.setFormatter(logging.Formatter(config.file_format))
    logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(config.stream_handler_level)
    stream_handler.setFormatter(logging.Formatter(config.stream_format))
    logger.addHandler(stream_handler)

    return logger


__all__ = ["get_logger", "LoggerConfig", "DefaultLoggerConfig"]
