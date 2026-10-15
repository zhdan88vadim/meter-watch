"""Integration test: real torch model, no HTTP, no DB."""
import numpy as np
import pytest
import torch

from meter_watch.infrastructure.classifier.pytorch_classifier import (
    PyTorchDigitClassifier,
)
from meter_watch.ml.model import DigitRecognizer
from tests.fixtures.fakes import NullLogger


@pytest.mark.integration
def test_classifier_predicts_on_random_input() -> None:
    model = DigitRecognizer()
    classifier = PyTorchDigitClassifier(model, torch.device("cpu"), NullLogger())
    roi = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    pred = classifier.predict(roi)
    assert 0 <= pred.value <= 9
    assert 0.0 <= pred.confidence <= 1.0
