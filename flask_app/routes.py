from flask import request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from database import db, User


def register_routes(app):

    # ---------------- HOME ----------------
    @app.route("/")
    def home():
        return jsonify({
            "message": "C-Spark Employee Prediction API is running"
        })

    # ---------------- REGISTER ----------------
    @app.route("/api/register", methods=["POST"])
    def register():

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Request body is required"
            }), 400

        username = data.get("username")
        password = data.get("password")

        # Validate fields
        if not username or not password:
            return jsonify({
                "success": False,
                "message": "Username and password are required"
            }), 400

        # Check username already exists
        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            return jsonify({
                "success": False,
                "message": "Username already exists"
            }), 409

        # Hash password
        password_hash = generate_password_hash(password)

        # Create new user
        new_user = User(
            username=username,
            password_hash=password_hash,
            role="manager"
        )

        db.session.add(new_user)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Registration successful",
            "username": username,
            "role": "manager"
        }), 201

    # ---------------- LOGIN ----------------
    @app.route("/api/login", methods=["POST"])
    def login():

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Request body is required"
            }), 400

        username = data.get("username")
        password = data.get("password")

        # Validate fields
        if not username or not password:
            return jsonify({
                "success": False,
                "message": "Username and password are required"
            }), 400

        # Find user
        user = User.query.filter_by(username=username).first()

        if not user:
            return jsonify({
                "success": False,
                "message": "Invalid username or password"
            }), 401

        # Check password
        if not check_password_hash(user.password_hash, password):
            return jsonify({
                "success": False,
                "message": "Invalid username or password"
            }), 401

        # Create JWT token
        access_token = create_access_token(
            identity=str(user.id)
        )

        return jsonify({
            "success": True,
            "message": "Login successful",
            "username": user.username,
            "role": user.role,
            "access_token": access_token
        }), 200

    # ---------------- PROTECTED ----------------
    @app.route("/api/protected", methods=["GET"])
    @jwt_required()
    def protected():

        user_id = get_jwt_identity()

        user = db.session.get(User, int(user_id))

        if not user:
            return jsonify({
                "success": False,
                "message": "User not found"
            }), 404

        return jsonify({
            "success": True,
            "message": "Access granted",
            "username": user.username,
            "role": user.role
        }), 200