# backend/app.py
from flask import Flask
from flask_cors import CORS
from app.api.routes import api_bp
from app.data.database import init_db

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Initialize database
init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)