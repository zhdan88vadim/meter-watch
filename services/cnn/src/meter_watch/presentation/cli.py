"""Presentation CLI."""
from __future__ import annotations

import argparse

from meter_watch.presentation.app_factory import create_app


def serve() -> None:
    parser = argparse.ArgumentParser(description="Run the meter-watch server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--config", default="/media/vadim/1TB_SSD/my_github/meter-watch/services/cnn/config/default.json")
    parser.add_argument("--model", default="trained_models/digit_recognizer.pth")
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from meter_watch.composition_root import build_container

    container = build_container(
        config_path=args.config,
        model_path=args.model,
        start_monitoring=not args.no_monitor,
    )
    app = create_app(container)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    serve()
