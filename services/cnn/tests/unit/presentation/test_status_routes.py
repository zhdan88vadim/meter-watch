from flask import Flask

from meter_watch.presentation.routes.status_routes import bp


def _client(container):  # noqa: ANN001
    app = Flask(__name__)
    app.container = container
    app.register_blueprint(bp)
    return app.test_client()


class _FakeMonitor:
    def status(self):  # noqa: ANN201
        from meter_watch.application.monitor_meter import MonitorStatus
        return MonitorStatus(total_readings=0, last_reading=None, is_running=False)

    def history(self):  # noqa: ANN201
        return []


class _FakeContainer:
    monitor = _FakeMonitor()


def test_status_empty() -> None:
    client = _client(_FakeContainer())
    r = client.get("/status")
    assert r.status_code == 200
    assert r.get_json()["total_recognized"] == 0


def test_history_empty() -> None:
    client = _client(_FakeContainer())
    r = client.get("/history")
    assert r.status_code == 200
    assert r.get_json()["history"] == []


def test_metrics_zero() -> None:
    client = _client(_FakeContainer())
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"last_recognition_value 0" in r.data
