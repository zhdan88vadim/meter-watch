"""Integration test: RecognizeImageUseCase on real images with the real model."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest
import torch
import yaml

from meter_watch.infrastructure.logging.null_logger import NullLogger
from meter_watch.application.recognize_image import RecognizeImageUseCase
from meter_watch.infrastructure.classifier.heatmap import SimpleHeatmap
from meter_watch.infrastructure.classifier.heatmap_annotator import GradCamAnnotator
from meter_watch.infrastructure.classifier.model_loader import ModelLoader
from meter_watch.infrastructure.classifier.pytorch_classifier import (
    PyTorchDigitClassifier,
)
from meter_watch.ml.model import DigitRecognizer

FIXTURES = Path(__file__).parent.parent / "fixtures" / "images"
MODEL_PATH = Path("trained_models/digit_recognizer.pth")


def _load_expected() -> dict[str, str]:
    print("FIXTURES", FIXTURES)

    with open(FIXTURES / "expected.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"model not found at {MODEL_PATH}")
    device = torch.device("cpu")
    m = ModelLoader(
        model_class=DigitRecognizer,
        path=str(MODEL_PATH),
        device=device,
        logger=NullLogger(),
    ).load()
    assert m is not None
    return m


@pytest.fixture(scope="module")
def use_case(model):
    device = torch.device("cpu")
    classifier = PyTorchDigitClassifier(model, device, NullLogger())
    annotator = GradCamAnnotator(model, SimpleHeatmap(model), device)
    return RecognizeImageUseCase(
        classifier=classifier,
        annotator=annotator,
        logger=NullLogger(),
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename,expected_digits",
    list(_load_expected().items()),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_recognizes_known_digits(use_case, filename: str, expected_digits: str) -> None:
    image = cv2.imread(str(FIXTURES / filename))
    assert image is not None, f"could not load {filename}"

    result = use_case.execute(image)

    # Exact string match (leading zeros preserved)
    assert result.number.text == expected_digits, (
        f"{filename}: expected {expected_digits!r}, got {result.number.text!r}"
    )
    # One DigitRecognition per digit
    assert len(result.digits) == len(expected_digits)
    # Positions are 1-indexed and sequential
    assert [d.position for d in result.digits] == list(
        range(1, len(expected_digits) + 1)
    )
    # min_confidence is the minimum across digits
    assert result.min_confidence == min(
        d.prediction.confidence for d in result.digits
    )