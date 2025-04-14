# run.py
from app import create_app # Import the factory function
import os # Import os if setting environment variables

# Optional: Set environment for development/production
# os.environ['FLASK_ENV'] = 'development' 

app = create_app() # Call the factory to create the app instance

if __name__ == '__main__':
    # Use debug=True only for development
    # For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(debug=True) 
