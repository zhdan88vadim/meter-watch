"""
Integration tests for YoloDetector using a real image with a person.

Run:
    pytest tests/integration/test_yolo_detector.py -v
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.domain.models import Detection
from app.infra.yolo_detector import YoloDetector


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        pytest.fail(f"Could not read image: {path}")
    return img


@pytest.mark.integration
def test_detects_person_in_image(person_image_path: Path):
    detector = YoloDetector("yolov8n.pt")
    image = _read_image(person_image_path)

    detections = list(detector.detect(image))

    print(f"\nFound {len(detections)} detection(s):")

    assert detections, f"No person detected in {person_image_path.name}"
    assert all(isinstance(d, Detection) for d in detections)


@pytest.mark.integration
def test_detects_person_in_image_next_image():
    path = "tests/assets/person2.png"
    detector = YoloDetector("yolov8n.pt")
    image = _read_image(path)

    detections = list(detector.detect(image))

    print(f"\nFound {len(detections)} detection(s):")

    assert detections, f"No person detected in {path}"
    assert all(isinstance(d, Detection) for d in detections)


@pytest.mark.integration
def test_detection_fields_are_valid(person_image_path: Path):
    detector = YoloDetector("yolov8n.pt")
    image = _read_image(person_image_path)
    h, w = image.shape[:2]

    detections = list(detector.detect(image))

    print(f"\nFound {len(detections)} detection(s):")

    assert detections, "Expected at least one detection"

    for det in detections:
        assert det.track_id >= 0
        assert 0 <= det.x1 < det.x2 <= w, f"Bad bbox x: {det}"
        assert 0 <= det.y1 < det.y2 <= h, f"Bad bbox y: {det}"


@pytest.mark.integration
def test_returns_empty_on_image_without_people(empty_image_path: Path):
    detector = YoloDetector("yolov8n.pt")
    image = _read_image(empty_image_path)

    detections = list(detector.detect(image))

    print(f"\nFound {len(detections)} detection(s):")

    assert detections == [], (
        f"Expected no detections in {empty_image_path.name}, got {detections}"
    )


@pytest.mark.integration
def test_detects_same_person_on_repeated_calls(person_image_path: Path):
    """
    Two consecutive calls on the same image should return the same bbox.
    This catches accidental tracker-state bugs (persist=True keeps state).
    """
    detector = YoloDetector("yolov8n.pt")
    image = _read_image(person_image_path)

    first = list(detector.detect(image))
    second = list(detector.detect(image))

    assert first and second
    f = first[0]
    s = second[0]
    assert (f.x1, f.y1, f.x2, f.y2) == (s.x1, s.y1, s.x2, s.y2)


@pytest.mark.integration
def test_returns_empty_on_blank_image():
    """
    Pure black image: no person, no crash, empty list.
    """
    detector = YoloDetector("yolov8n.pt")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = list(detector.detect(blank))
    assert detections == []


@pytest.mark.integration
def test_detect_does_not_raise_on_garbage_image():
    """
    Random noise: no person, no crash, empty list.
    """
    rng = np.random.default_rng(seed=0)
    noise = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    detector = YoloDetector("yolov8n.pt")
    detections = list(detector.detect(noise))
    assert detections == []