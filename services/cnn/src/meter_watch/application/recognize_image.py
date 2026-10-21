"""Use case: recognize the digits in an image."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from meter_watch.domain.models import (
    DigitRecognition,
    MeterNumber,
    RecognitionResult,
)
from meter_watch.domain.protocols import (
    DigitClassifier,
    ImageAnnotator,
    Logger,
)
from meter_watch.image_processing.centering import center_digit
from meter_watch.image_processing.preprocessing import preprocess_image
from meter_watch.image_processing.splitter import split_digits


@dataclass
class RecognizeImageUseCase:
    """Recognize digits in a meter image.

    Collaborators are injected so the use case can be tested with fakes.
    """

    classifier: DigitClassifier
    annotator: ImageAnnotator
    logger: Logger

    def execute(self, image: np.ndarray) -> RecognitionResult:
        thresh = preprocess_image(image)
        digit_rois = split_digits(thresh, image)

        if not digit_rois:
            self.logger.warning("no digits found in image")
            return RecognitionResult(
                digits=[],
                number=MeterNumber(digits=[]),
                min_confidence=0.0,
            )

        recognitions: List[DigitRecognition] = []
        min_conf = 1.0

        for i, roi in enumerate(digit_rois):
            centered = center_digit(roi)
            pred = self.classifier.predict(centered)
            min_conf = min(min_conf, pred.confidence)

            annotation = self.annotator.annotate(centered)
            recognitions.append(
                DigitRecognition(
                    position=i + 1,
                    prediction=pred,
                    heatmap_gradcam=annotation.get("heatmap_gradcam", ""),
                    heatmap_saliency=annotation.get("heatmap_saliency", ""),
                    digit_image_b64=annotation.get("digit_image", ""),
                )
            )

        number = MeterNumber(digits=[r.prediction.value for r in recognitions])

        return RecognitionResult(
            digits=recognitions,
            number=number,
            min_confidence=min_conf,
        )
