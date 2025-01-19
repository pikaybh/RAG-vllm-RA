import os
import logging
from functools import wraps
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from .extensions import extfomatter


SELF_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SELF_DIR)  # 한 단계 상위 폴더로 이동
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

    def _create_log_decorator(self, log_func, level: str, message_format: Optional[str] = None) -> Callable:
        """Creates a logging decorator for the specified log level.
        
        Args:
            log_func: Logging function to use (e.g., self.info, self.debug)
            level: Name of the log level for documentation
            message_format: Optional format string for the log message
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                fmt = message_format or "{func_name}: {result}"
                log_func(fmt.format(
                    func_name=func.__name__,
                    result=result,
                    args=args,
                    kwargs=kwargs
                ))
                return result
            return wrapper
            
        # 소괄호 없이 사용할 경우를 위한 처리
        return decorator if message_format else decorator

    def pinfo(self, message_format: Optional[str] = None) -> Callable:
        """Decorator that logs the output at INFO level."""
        if isinstance(message_format, Callable):  # 소괄호 없이 사용된 경우
            return self._create_log_decorator(self.info, "INFO")(message_format)
        return self._create_log_decorator(self.info, "INFO", message_format)

    def pdebug(self, message_format: Optional[str] = None) -> Callable:
        """Decorator that logs the output at DEBUG level."""
        if isinstance(message_format, Callable):
            return self._create_log_decorator(self.debug, "DEBUG")(message_format)
        return self._create_log_decorator(self.debug, "DEBUG", message_format)

    def pwarn(self, message_format: Optional[str] = None) -> Callable:
        """Decorator that logs the output at WARNING level."""
        if isinstance(message_format, Callable):
            return self._create_log_decorator(self.warning, "WARNING")(message_format)
        return self._create_log_decorator(self.warning, "WARNING", message_format)

    def perror(self, message_format: Optional[str] = None) -> Callable:
        """Decorator that logs the output at ERROR level."""
        if isinstance(message_format, Callable):
            return self._create_log_decorator(self.error, "ERROR")(message_format)
        return self._create_log_decorator(self.error, "ERROR", message_format)


def get_logger(name: str, root: Optional[str] = ROOT_DIR, config: Optional[LoggerConfig] = DefaultLoggerConfig()) -> CustomLogger:
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


__all__ = ["get_logger", "LoggerConfig", "DefaultLoggerConfig", "CustomLogger"]
