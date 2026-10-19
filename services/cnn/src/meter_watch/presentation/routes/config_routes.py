"""Config routes. Built as a factory taking a config provider."""
from __future__ import annotations

from flask import Blueprint, jsonify, request
from meter_watch.presentation.protocols import ConfigProvider


def make_config_bp(config_provider: ConfigProvider) -> Blueprint:  # noqa: ANN001
    bp = Blueprint("config", __name__)

    @bp.route("/config", methods=["GET"])
    def get_all():
        return jsonify({
            "status": "success",
            "config": config_provider.to_dict(),
        })

    @bp.route("/config/<string:key>", methods=["GET"])
    def get_one(key: str):
        if not config_provider.has(key):
            return jsonify({
                "status": "error",
                "message": f"Parameter '{key}' not found",
            }), 404
        return jsonify({
            "status": "success",
            "param": key,
            "value": config_provider.get(key),
        })

    @bp.route("/config", methods=["POST"])
    def update_many():
        data = request.get_json() or {}
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        return jsonify({
            "status": "success",
            "changes": config_provider.update(data),
        })

    @bp.route("/config/<string:key>", methods=["PUT"])
    def update_one(key: str):
        data = request.get_json() or {}
        if "value" not in data:
            return jsonify({"status": "error", "message": "Missing 'value'"}), 400
        if not config_provider.has(key):
            return jsonify({
                "status": "error",
                "message": f"Parameter '{key}' not found",
            }), 404
        old = config_provider.get(key)
        config_provider.set(key, data["value"])
        return jsonify({
            "status": "success",
            "param": key,
            "old": old,
            "new": data["value"],
        })

    @bp.route("/config/reset", methods=["POST"])
    def reset():
        # Reset to defaults by writing empty overrides
        defaults = {
            "camera_url": "rtsp://username:password@ip:port/stream",
            "camera_request_pause": 5,
            "monitoring_enabled": True,
            "save_threshold": 0.6,
            "save_bad_photos": True,
        }
        config_provider.update(defaults)
        return jsonify({"status": "success"})

    return bp
