"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from meter_watch.application.monitor_meter import MonitorMeterUseCase
from meter_watch.application.recognize_image import RecognizeImageUseCase
from meter_watch.domain.services.anomaly_detector import AnomalyDetector

from tests.fixtures.fakes import (
    FakeClassifier,
    FakeClock,
    FakeFetcher,
    FakeRepo,
    NullAnnotator,
    NullLogger,
)


@pytest.fixture
def logger() -> NullLogger:
    return NullLogger()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def repo() -> FakeRepo:
    return FakeRepo()


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier(value=5)


@pytest.fixture
def recognizer(classifier, logger) -> RecognizeImageUseCase:
    return RecognizeImageUseCase(
        classifier=classifier,
        annotator=NullAnnotator(),
        logger=logger,
    )


@pytest.fixture
def monitor(recognizer, repo, clock, logger) -> MonitorMeterUseCase:
    return MonitorMeterUseCase(
        fetcher=FakeFetcher(),
        recognizer=recognizer,
        anomaly_detector=AnomalyDetector(min_confidence=0.5),
        repository=repo,
        clock=clock,
        logger=logger,
        poll_interval=0.01,
    )
