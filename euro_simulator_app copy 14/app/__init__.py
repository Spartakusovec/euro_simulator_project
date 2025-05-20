from flask import Flask
import os
import logging


def create_app():
    """Creates and configures the Flask application instance."""
    app = Flask(__name__)

    app.config["BASE_DIR"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"BASE_DIR set in app.config: {app.config['BASE_DIR']}")

    with app.app_context():
        from . import routes

    logging.info("Flask app created successfully.")
    return app
