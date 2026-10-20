import json

from meter_watch.infrastructure.config.file_provider import FileConfigProvider


def test_get_defaults(tmp_path) -> None:
    p = FileConfigProvider(str(tmp_path / "config.json"))
    assert p.get("camera_request_pause") == 5


def test_set_persists(tmp_path) -> None:
    path = tmp_path / "config.json"
    p1 = FileConfigProvider(str(path))
    p1.set("camera_request_pause", 3)

    p2 = FileConfigProvider(str(path))
    assert p2.get("camera_request_pause") == 3


def test_update_returns_changes(tmp_path) -> None:
    p = FileConfigProvider(str(tmp_path / "config.json"))
    changes = p.update({"save_threshold": 0.9, "unknown": 1})
    assert changes["save_threshold"]["old"] == 0.6
    assert changes["save_threshold"]["new"] == 0.9
    assert "unknown" not in changes


def test_load_from_existing_file(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_url": "rtsp://loaded"}))
    p = FileConfigProvider(str(path))
    assert p.get("camera_url") == "rtsp://loaded"
