import time
from functools import wraps
from typing import Callable, Any
from utils import get_logger

logger = get_logger("utils.decorators")

def timer(func: Callable) -> Callable:
    """Measures execution time of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Get class name if method belongs to a class
        if args and hasattr(args[0], '__class__'):
            qualifier = f"{args[0].__class__.__name__}."
        else:
            qualifier = ""
            
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{qualifier}{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper

__all__ = ["timer"] 