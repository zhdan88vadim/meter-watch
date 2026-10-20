"""HTTP image fetcher using requests + OpenCV."""
from __future__ import annotations

import time
from typing import Callable, Optional

import cv2
import numpy as np
import requests

from meter_watch.domain.protocols import Logger


class RequestsImageFetcher:
    """Fetches images from an HTTP camera endpoint."""

    def __init__(
        self,
        url_provider: Callable[[], str],
        logger: Logger,
        timeout: float = 5.0,
    ) -> None:
        self._url_provider = url_provider
        self._logger = logger
        self._timeout = timeout

    def fetch(self) -> Optional[np.ndarray]:
        base_url = self._url_provider()
        url = f"{base_url}{int(time.time() * 1000)}"
        try:
            resp = requests.get(url, stream=True, timeout=self._timeout)
            if resp.status_code != 200:
                self._logger.warning(
                    "fetch failed", status=resp.status_code, url=url
                )
                return None
            buf = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if image is None:
                self._logger.warning("decode failed", url=url)
            return image
        except Exception as exc:  # noqa: BLE001
            self._logger.error("fetch exception", error=str(exc), url=url)
            return None
