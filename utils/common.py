# utils/common.py
# General utility functions

# Debug control
DEBUG = True

def setup_debug(enable=True):
    """Set up debugging mode"""
    global DEBUG
    DEBUG = enable

def debug(*args, **kwargs):
    """Print debug messages when DEBUG is enabled"""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def generate_timestamp():
    """Generate a timestamp for file naming"""
    import time
    return time.strftime("%Y%m%d_%H%M%S")
