from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager

from .database import db


def create_app():

    # -----------------------------------------
    # CREATE FLASK APP
    # -----------------------------------------

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    # -----------------------------------------
    # CONFIGURATION
    # -----------------------------------------

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///c_spark.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # JWT secret key
    app.config["JWT_SECRET_KEY"] = "change-this-to-a-random-secret-key"

    # -----------------------------------------
    # INITIALIZE EXTENSIONS
    # -----------------------------------------

    db.init_app(app)

    CORS(app)

    JWTManager(app)

    # -----------------------------------------
    # DATABASE
    # -----------------------------------------

    with app.app_context():
        db.create_all()

    # -----------------------------------------
    # REGISTER ROUTES
    # -----------------------------------------

    from .routes import register_routes

    register_routes(app)

    # -----------------------------------------
    # RETURN APP
    # -----------------------------------------

    return app