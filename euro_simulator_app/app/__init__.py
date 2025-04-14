# app/__init__.py
from flask import Flask
import os
import logging # Make sure logging is configured if used here

def create_app():
    """Creates and configures the Flask application instance."""
    app = Flask(__name__)

    # --- Configuration ---
    # Set configuration variables here if needed
    # Example: app.config['SECRET_KEY'] = 'your_secret_key' 
    
    # Store base directory for database path resolution in routes
    app.config['BASE_DIR'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"BASE_DIR set in app.config: {app.config['BASE_DIR']}")

    # --- Import and Register Routes (and other blueprints/extensions) ---
    with app.app_context():
        # Import routes *after* the app is created and context is available
        from . import routes 
        
        # If you were using Blueprints, you would register them here:
        # from .routes import bp 
        # app.register_blueprint(bp)

    # You could register extensions like SQLAlchemy, LoginManager etc. here
    # db.init_app(app)
    # login_manager.init_app(app)

    logging.info("Flask app created successfully.")
    return app

