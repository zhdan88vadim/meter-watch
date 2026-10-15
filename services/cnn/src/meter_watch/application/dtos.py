"""Application-level DTOs for serialization."""
from __future__ import annotations

from meter_watch.domain.models import MeterReading, RecognitionResult


def serialize_reading(reading: MeterReading) -> dict:
    """Serialize a MeterReading to a JSON-friendly dict."""
    return {
        "digits": list(reading.number.digits),
        "number": reading.number.value,
        "min_conf": reading.min_confidence,
        "time_str": reading.time_str,
        "timestamp": reading.timestamp,
    }


def serialize_recognition(result: RecognitionResult) -> dict:
    """Serialize a RecognitionResult to a JSON-friendly dict."""
    return {
        "full_number": result.number.text,
        "min_conf": result.min_confidence,
        "digits": [
            {
                "position": d.position,
                "prediction": d.prediction.value,
                "confidence": d.prediction.confidence,
                "heatmap_gradcam": d.heatmap_gradcam,
                "heatmap_saliency": d.heatmap_saliency,
                "digit_image": d.digit_image_b64,
            }
            for d in result.digits
        ],
    }
