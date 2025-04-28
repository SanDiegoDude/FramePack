# utils/common.py
import datetime

# Default to debug off
DEBUG = False

def setup_debug(enabled=False):
    """Set the debug output state"""
    global DEBUG
    DEBUG = enabled
    if enabled:
        print("[DEBUG] Debug logging enabled")

def debug(*args, **kwargs):
    """Print debug output if debug mode is enabled"""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def generate_timestamp():
    """Generate a timestamp string for filenames (platform independent with microseconds)"""
    # Returns e.g. '240610_131537_234'
    now = datetime.datetime.now()
    return now.strftime("%y%m%d_%H%M%S_%f")[:-3]
# year, month, day, hour, min, sec, first 3 digits of microseconds (= milliseconds)
