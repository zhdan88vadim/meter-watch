"""Composition root: the only place concrete classes are wired together."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from meter_watch.domain.errors import ModelNotLoadedError
from meter_watch.application.monitor_meter import MonitorMeterUseCase
from meter_watch.application.recognize_image import RecognizeImageUseCase
from meter_watch.domain.protocols import (
    Clock,
    DigitClassifier,
    ImageAnnotator,
    ImageFetcher,
    Logger,
    ReadingRepository,
)
from meter_watch.domain.services.anomaly_detector import AnomalyDetector

from meter_watch.infrastructure.classifier.heatmap import SimpleHeatmap
from meter_watch.infrastructure.classifier.heatmap_annotator import (
    GradCamAnnotator,
    NoopAnnotator,
)
from meter_watch.infrastructure.classifier.model_loader import ModelLoader
from meter_watch.infrastructure.classifier.pytorch_classifier import (
    PyTorchDigitClassifier,
)
from meter_watch.infrastructure.config.file_provider import FileConfigProvider
from meter_watch.infrastructure.fetcher.requests_fetcher import RequestsImageFetcher
from meter_watch.infrastructure.logging.stdlib_logger import StdlibLogger
from meter_watch.infrastructure.repository.sqlalchemy_repository import (
    SqlAlchemyReadingRepository,
)
from meter_watch.infrastructure.time.system_clock import SystemClock
from meter_watch.ml.model import DigitRecognizer


@dataclass
class Container:
    """Holds every constructed collaborator."""

    config: Any
    logger: Logger
    clock: Clock
    repository: ReadingRepository
    fetcher: ImageFetcher
    classifier: DigitClassifier
    recognizer: RecognizeImageUseCase
    monitor: MonitorMeterUseCase


def build_container(
    config_path: str = "config.json",
    model_path: str = "trained_models/digit_recognizer.pth",
    device: torch.device | None = None,
    session_factory: Any = None,
    start_monitoring: bool = True,
    poll_interval: float = 20.0,
    max_history: int = 1000,
) -> Container:
    """Build the container. All side effects happen here."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = StdlibLogger()
    clock = SystemClock()
    config = FileConfigProvider(config_path)

    model = ModelLoader(
        model_class=DigitRecognizer,
        path=model_path,
        device=device,
        logger=logger,
    ).load()

    if model is None:
        raise ModelNotLoadedError("Model is None")
        
    classifier = PyTorchDigitClassifier(model, device, logger)
    annotator: ImageAnnotator = GradCamAnnotator(model, SimpleHeatmap(model), device)

    recognizer = RecognizeImageUseCase(
        classifier=classifier,
        annotator=annotator,
        logger=logger,
    )

    # --- Fetcher ---
    fetcher = RequestsImageFetcher(
        url_provider=lambda: config.get("camera_url"),
        logger=logger,
    )

    # --- Repository ---
    if session_factory is None:
        try:
            from meter_watch_shared.db import SessionLocal  # noqa: PLC0415
            session_factory = SessionLocal
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"no session_factory; DB persistence disabled: {exc}")
            session_factory = lambda: _NoopSession()

    repository = SqlAlchemyReadingRepository(session_factory, logger)

    # --- Anomaly detector ---
    anomaly = AnomalyDetector(min_confidence=config.get("save_threshold"))

    # --- Monitor ---
    monitor = MonitorMeterUseCase(
        fetcher=fetcher,
        recognizer=recognizer,
        anomaly_detector=anomaly,
        repository=repository,
        clock=clock,
        logger=logger,
        poll_interval=poll_interval,
        max_history=max_history,
    )

    if start_monitoring and recognizer is not None and config.get("monitoring_enabled"):
        monitor.start()

    return Container(
        config=config,
        logger=logger,
        clock=clock,
        repository=repository,
        fetcher=fetcher,
        classifier=classifier,
        recognizer=recognizer,
        monitor=monitor,
    )


class _NoopSession:
    """Minimal stand-in for a DB session when persistence is disabled."""

    def add(self, *args: object, **kwargs: object) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None
