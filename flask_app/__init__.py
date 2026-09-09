from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager

from database import db


def create_app():
    app = Flask(__name__)

    # CORS
    CORS(app)

    # JWT configuration
    app.config["JWT_SECRET_KEY"] = "change-this-to-a-random-secret-key"

    # SQLite database configuration
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///c_spark.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Initialize extensions
    db.init_app(app)
    JWTManager(app)

    # Create database tables
    with app.app_context():
        db.create_all()

    # Register routes
    from routes import register_routes
    register_routes(app)

    return app