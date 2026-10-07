from __future__ import annotations

from typing import Sequence

from ultralytics import YOLO

from app.domain.models import Detection, Frame


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt") -> None:
        self._model = YOLO(weights)

    def detect(self, frame: Frame) -> Sequence[Detection]:
        try:
            results = self._model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                verbose=False,
            )
        except Exception:
            return []

        if not results or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy().astype(int)

        detections = []
        for idx, track_id in enumerate(ids):
            x1, y1, x2, y2 = xyxy[idx]
            detections.append(
                Detection(int(track_id), int(x1), int(y1), int(x2), int(y2))
            )
        return detections
