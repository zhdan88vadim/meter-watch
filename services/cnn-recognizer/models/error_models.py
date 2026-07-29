class MeterMonitorError(Exception):
    """Base exception for the monitor"""
    pass

class ImageFetchError(MeterMonitorError):
    """Image fetching error"""
    pass

class RecognitionError(MeterMonitorError):
    """Recognition error"""
    pass