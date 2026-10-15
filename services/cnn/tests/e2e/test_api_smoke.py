"""End-to-end smoke test: build the app with no model and hit /status."""
from meter_watch.composition_root import build_container
from meter_watch.presentation.app_factory import create_app


def test_app_starts_without_model(tmp_path) -> None:
    container = build_container(
        config_path=str(tmp_path / "config.json"),
        model_path=str(tmp_path / "missing.pth"),
        start_monitoring=False,
        session_factory=lambda: _NoopSession(),
    )
    app = create_app(container)
    client = app.test_client()

    r = client.get("/status")
    assert r.status_code == 200
    assert r.get_json()["total_recognized"] == 0

    r = client.get("/config")
    assert r.status_code == 200
    assert "camera_url" in r.get_json()["config"]


class _NoopSession:
    def add(self, *a, **k) -> None:
        pass

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass
