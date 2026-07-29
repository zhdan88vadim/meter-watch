from flask import Blueprint, jsonify, request
from services.config import AppConfig, ConfigKeys, config
from typing import Dict, Any

config_bp = Blueprint('config', __name__)

@config_bp.route("/config", methods=["GET"])
def get_config():
    """Get all configuration settings"""
    return jsonify({
        "status": "success",
        "config": config.to_dict()
    })

@config_bp.route("/config/<string:key>", methods=["GET"])
def get_config_param(key: str):
    """Get a specific configuration parameter"""
    if not hasattr(config, key):
        return jsonify({"status": "error", "message": f"Parameter '{key}' not found"}), 404
    
    return jsonify({
        "status": "success",
        "param": key,
        "value": getattr(config, key)
    })

@config_bp.route("/config", methods=["POST"])
def update_config():
    """Update configuration settings"""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data"}), 400
    
    changes = config.update(data)
    
    return jsonify({
        "status": "success",
        "changes": changes
    })

@config_bp.route("/config/<string:key>", methods=["PUT"])
def update_config_param(key: str):
    """Update a specific configuration parameter"""
    if not hasattr(config, key):
        return jsonify({"status": "error", "message": f"Parameter '{key}' not found"}), 404
    
    data = request.get_json()
    if not data or "value" not in data:
        return jsonify({"status": "error", "message": "Missing 'value'"}), 400
    
    old_value = getattr(config, key)
    changes = config.update({key: data["value"]})
    
    return jsonify({
        "status": "success",
        "param": key,
        "old": old_value,
        "new": data["value"]
    })