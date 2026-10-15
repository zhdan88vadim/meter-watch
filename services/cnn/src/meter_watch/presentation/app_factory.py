"""Flask app factory. Wires blueprints to a Container."""
from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from meter_watch.composition_root import Container, build_container


def create_app(container: Container | None = None) -> Flask:
    """Build a Flask app wired to the given (or freshly built) container."""
    container = container or build_container()

    app = Flask(__name__)
    CORS(app)

    from meter_watch.presentation.routes.config_routes import make_config_bp
    from meter_watch.presentation.routes.recognition_routes import (
        make_recognition_bp,
    )
    from meter_watch.presentation.routes.status_routes import bp as status_bp

    app.register_blueprint(status_bp)
    app.register_blueprint(make_config_bp(container.config))
    if container.recognizer is not None:
        app.register_blueprint(
            make_recognition_bp(container.recognizer, container.config)
        )

    app.container = container  # type: ignore[attr-defined]
    return app
