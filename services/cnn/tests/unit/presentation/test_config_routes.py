from flask import Flask

from meter_watch.infrastructure.config.in_memory_provider import (
    InMemoryConfigProvider,
)
from meter_watch.presentation.routes.config_routes import make_config_bp


def _client(provider):  # noqa: ANN001
    app = Flask(__name__)
    app.register_blueprint(make_config_bp(provider))
    return app.test_client()


def test_get_config() -> None:
    provider = InMemoryConfigProvider(camera_url="rtsp://test")
    client = _client(provider)
    r = client.get("/config")
    assert r.status_code == 200
    assert r.get_json()["config"]["camera_url"] == "rtsp://test"


def test_get_unknown_key() -> None:
    provider = InMemoryConfigProvider()
    client = _client(provider)
    r = client.get("/config/nope")
    assert r.status_code == 404


def test_update_many() -> None:
    provider = InMemoryConfigProvider()
    client = _client(provider)
    r = client.post("/config", json={"camera_request_pause": 3})
    assert r.status_code == 200
    assert r.get_json()["changes"]["camera_request_pause"]["new"] == 3


def test_update_one_put() -> None:
    provider = InMemoryConfigProvider()
    client = _client(provider)
    r = client.put("/config/camera_request_pause", json={"value": 7})
    assert r.status_code == 200
    assert r.get_json()["new"] == 7


def test_reset() -> None:
    provider = InMemoryConfigProvider(camera_url="rtsp://changed")
    client = _client(provider)
    r = client.post("/config/reset")
    assert r.status_code == 200
    assert provider.get("camera_url") == "rtsp://username:password@ip:port/stream"
