import numpy as np
import pytest

from meter_watch.application.recognize_image import RecognizeImageUseCase
from tests.fixtures.fakes import FakeClassifier, NullAnnotator, NullLogger


def test_no_digits_returns_empty_result() -> None:
    uc = RecognizeImageUseCase(
        classifier=FakeClassifier(),
        annotator=NullAnnotator(),
        logger=NullLogger(),
    )
    # Blank image — splitter should find nothing
    result = uc.execute(np.zeros((100, 100, 3), np.uint8))
    assert result.digits == []
    assert result.min_confidence == 0.0
