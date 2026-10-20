"""Manual recognition routes. Built as a factory taking a use case."""
from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, request
from PIL import Image

from meter_watch.application.dtos import serialize_recognition
from meter_watch.presentation.protocols import ConfigProvider, Recognizer

FlaskResponse = Response | tuple[Response, int]


def make_recognition_bp(
    recognizer: Recognizer,
    config_provider: ConfigProvider,
) -> Blueprint:
    bp = Blueprint("recognition", __name__)

    @bp.route("/recognize", methods=["POST"])
    def recognize() -> FlaskResponse:
        data = request.get_json() or {}
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        try:
            raw = base64.b64decode(data["image"])
            img = cv2.cvtColor(
                np.array(Image.open(io.BytesIO(raw))),
                cv2.COLOR_RGB2BGR,
            )
            result = recognizer.execute(img)
            return jsonify(serialize_recognition(result))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

    @bp.route("/test", methods=["GET"])
    def test_file() -> FlaskResponse:
        filename = request.args.get("filename")
        dataset = request.args.get("dataset")
        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        # base_path = config_provider.get("output_dir") if config_provider.has("output_dir") else "output"
        
        base_path = "/media/vadim/1TB_SSD/my_github/meter-watch/output"
        file_path = os.path.join(base_path, dataset or "", filename)
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404

        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        result = recognizer.execute(img)
        payload = serialize_recognition(result)
        payload["filename"] = filename
        return jsonify(payload)

    @bp.route("/test/list", methods=["GET"])
    def test_list() -> FlaskResponse:
        dataset = request.args.get("dataset", "")
        # base_path = "output"
        base_path = "/media/vadim/1TB_SSD/my_github/meter-watch/output"
        dataset_path = os.path.join(base_path, dataset)
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        if not os.path.isdir(dataset_path):
            return jsonify({"files": [], "count": 0})
        files = [
            f for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f))
            and f.lower().endswith(valid_ext)
        ]
        return jsonify({"files": files, "count": len(files)})

    @bp.route("/delete-file", methods=["POST"])
    def delete_file() -> FlaskResponse:
        data = request.get_json() or {}
        filename = data.get("filename")
        dataset = data.get("dataset", "")
        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        path = os.path.join("output", dataset, filename)
        if not os.path.exists(path):
            return jsonify({"error": f"File not found: {filename}"}), 404
        os.remove(path)
        return jsonify({"success": True, "filename": filename})

    @bp.route("/save-digit", methods=["POST"])
    def save_digit() -> FlaskResponse:
        data = request.get_json() or {}
        digit = data.get("digit")
        image_b64 = data.get("image_base64")
        filename = data.get("filename")

        if digit is None:
            return jsonify({"error": "Digit is required"}), 400
        try:
            digit = int(digit)
        except (TypeError, ValueError):
            return jsonify({"error": "Digit must be a number"}), 400
        if not 0 <= digit <= 9:
            return jsonify({"error": "Digit must be 0-9"}), 400
        if not image_b64:
            return jsonify({"error": "Image data is required"}), 400

        folder = Path("output") / "manual_recognized_data" / str(digit)
        folder.mkdir(parents=True, exist_ok=True)

        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{digit}_{ts}_{uuid4().hex[:8]}_org.png"
        elif not filename.endswith(".png"):
            filename += ".png"

        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_b64 = "".join(image_b64.split())

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Invalid base64: {exc}"}), 400

        filepath = folder / filename
        filepath.write_bytes(image_bytes)
        return jsonify({
            "success": True,
            "digit": digit,
            "filename": filename,
            "filepath": str(filepath),
        })

    return bp
