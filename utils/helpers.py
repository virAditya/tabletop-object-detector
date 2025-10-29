"""
Utility Helper Functions
Common utilities used across modules
"""

import time
from functools import wraps


def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱ {func.__name__} took {end - start:.3f} seconds")
        return result
    return wrapper


def validate_image_shape(image, expected_channels=3):
    """
    Validate image shape
    
    Args:
        image: Input image
        expected_channels: Expected number of channels
        
    Returns:
        bool: True if valid
    """
    if image is None:
        return False
    if len(image.shape) != 3:
        return False
    if image.shape[2] != expected_channels:
        return False
    return True


def print_separator(char="=", length=70):
    """Print a separator line"""
    print(char * length)


def print_header(title):
    """Print formatted header"""
    print_separator()
    print(title.center(70))
    print_separator()
