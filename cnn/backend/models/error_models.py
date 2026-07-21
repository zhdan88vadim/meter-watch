class MeterMonitorError(Exception):
    """Базовое исключение для монитора"""
    pass

class ImageFetchError(MeterMonitorError):
    """Ошибка получения изображения"""
    pass

class RecognitionError(MeterMonitorError):
    """Ошибка распознавания"""
    pass
