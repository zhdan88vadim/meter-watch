"""Status / history / metrics routes."""
from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify

from meter_watch.application.dtos import serialize_reading


bp = Blueprint("status", __name__)


@bp.route("/status", methods=["GET"])
def status() -> Response:
    monitor = current_app.container.monitor  # type: ignore[attr-defined]
    s = monitor.status()
    return jsonify({
        "total_recognized": s.total_readings,
        "is_running": s.is_running,
        "last": serialize_reading(s.last_reading) if s.last_reading else None,
    })


@bp.route("/history", methods=["GET"])
def history() -> Response:
    monitor = current_app.container.monitor  # type: ignore[attr-defined]
    return jsonify({
        "history": [serialize_reading(r) for r in monitor.history()],
    })


@bp.route("/last_activity", methods=["GET"])
def last_activity() -> Response:
    monitor = current_app.container.monitor  # type: ignore[attr-defined]
    s = monitor.status()
    recent = monitor.history()[-4:]
    return jsonify({
        "latest_update": serialize_reading(s.last_reading) if s.last_reading else None,
        "recent_history": [serialize_reading(r) for r in recent],
    })


@bp.route("/metrics", methods=["GET"])
def metrics() -> Response:
    monitor = current_app.container.monitor  # type: ignore[attr-defined]
    last = monitor.status().last_reading
    value = last.number.value if last else 0
    body = (
        "# HELP last_recognition_value Last recognized value\n"
        "# TYPE last_recognition_value gauge\n"
        f"last_recognition_value {value}\n"
    )
    return Response(body, mimetype="text/plain")
