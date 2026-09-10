from flask import render_template, request, jsonify
import os
import csv

from flask_jwt_extended import (
    create_access_token,
    jwt_required
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from .database import db, User

from .data_service import (
    predict_employee,
    get_task_options,
    DATA_FILE
)


# =========================================================
# REGISTER ALL ROUTES
# =========================================================

def register_routes(app):

    # =====================================================
    # INSIGHTS PAGE
    # =====================================================

    @app.route("/insights")
    def insights_page():
        return render_template("insights.html")


    # =====================================================
    # LOGIN PAGE
    # =====================================================

    @app.route("/")
    def login_page():
        return render_template("login.html")


    # =====================================================
    # REGISTER PAGE
    # =====================================================

    @app.route("/register")
    def register():
        return render_template("register.html")


    # =====================================================
    # DASHBOARD PAGE
    # =====================================================

    @app.route("/dashboard")
    def dashboard():

        import pandas as pd

        try:
            df = pd.read_csv(DATA_FILE)

            total_employees = (
                df["employee_id"].nunique()
            )

            total_tasks = len(df)

            avg_rating = pd.to_numeric(
                df["rating"],
                errors="coerce"
            ).mean()

            avg_error_risk = pd.to_numeric(
                df["error_risk"],
                errors="coerce"
            ).mean()

            stats = {
                "total_employees": int(total_employees),
                "total_tasks": int(total_tasks),
                "avg_rating": (
                    round(float(avg_rating), 2)
                    if pd.notna(avg_rating)
                    else 0
                ),
                "avg_error_risk": (
                    round(float(avg_error_risk), 2)
                    if pd.notna(avg_error_risk)
                    else 0
                )
            }

            return render_template(
                "dashboard.html",
                stats=stats
            )

        except Exception as e:
            return (
                f"Dashboard error: {str(e)}",
                500
            )


    # =====================================================
    # PREDICT PAGE
    # =====================================================

    @app.route("/predict")
    def predict_page():
        return render_template("predict.html")


    # =====================================================
    # ANALYTICS PAGE
    # =====================================================

    @app.route("/analytics")
    def analytics_page():
        return render_template("analytics.html")


    # =====================================================
    # ANALYTICS API
    # =====================================================

    @app.route("/api/analytics")
    def analytics_api():

        import pandas as pd

        try:
            # -------------------------------------------------
            # LOAD ORIGINAL DATASET
            # -------------------------------------------------

            df = pd.read_csv(DATA_FILE)

            if df.empty:
                return jsonify({
                    "success": True,
                    "stats": {
                        "total_employees": 0,
                        "total_tasks": 0,
                        "avg_rating": 0,
                        "avg_error_risk": 0,
                        "completion_rate": 0
                    },
                    "performance_trend": {
                        "labels": [],
                        "values": []
                    },
                    "performance_distribution": {
                        "labels": [
                            "High Performer",
                            "Good Performer",
                            "Average Performer",
                            "Needs Improvement"
                        ],
                        "values": [0, 0, 0, 0]
                    },
                    "task_type_distribution": {
                        "labels": [],
                        "values": []
                    },
                    "priority_distribution": {
                        "labels": [],
                        "values": []
                    },
                    "risk_distribution": {
                        "labels": [
                            "Minimal",
                            "Very Low",
                            "Low",
                            "Moderate",
                            "High",
                            "Critical"
                        ],
                        "values": [0, 0, 0, 0, 0, 0]
                    },
                    "employees": []
                })

            # -------------------------------------------------
            # CLEAN NUMERIC COLUMNS
            # -------------------------------------------------

            numeric_columns = [
                "rating",
                "error_risk",
                "days_to_deadline",
                "volume_metric",
                "dependency_score"
            ]

            for column in numeric_columns:
                if column in df.columns:
                    df[column] = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    )

            # -------------------------------------------------
            # BASIC STATISTICS
            # -------------------------------------------------

            total_employees = (
                int(df["employee_id"].nunique())
                if "employee_id" in df.columns
                else 0
            )

            total_tasks = int(len(df))

            avg_rating = (
                float(df["rating"].mean())
                if "rating" in df.columns
                and df["rating"].notna().any()
                else 0
            )

            avg_error_risk = (
                float(df["error_risk"].mean())
                if "error_risk" in df.columns
                and df["error_risk"].notna().any()
                else 0
            )

            # -------------------------------------------------
            # COMPLETION RATE
            # -------------------------------------------------

            completion_rate = 0

            if "is_completed" in df.columns:

                completed = (
                    df["is_completed"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin([
                        "1",
                        "true",
                        "yes",
                        "y",
                        "completed"
                    ])
                    .sum()
                )

                completion_rate = (
                    (completed / total_tasks) * 100
                    if total_tasks > 0
                    else 0
                )

            # -------------------------------------------------
            # PERFORMANCE DISTRIBUTION
            # -------------------------------------------------

            performance_counts = {
                "High Performer": 0,
                "Good Performer": 0,
                "Average Performer": 0,
                "Needs Improvement": 0
            }

            if "rating" in df.columns:

                valid_ratings = df["rating"].dropna()

                performance_counts["High Performer"] = int(
                    (valid_ratings >= 8).sum()
                )

                performance_counts["Good Performer"] = int(
                    (
                        (valid_ratings >= 6)
                        & (valid_ratings < 8)
                    ).sum()
                )

                performance_counts["Average Performer"] = int(
                    (
                        (valid_ratings >= 4)
                        & (valid_ratings < 6)
                    ).sum()
                )

                performance_counts["Needs Improvement"] = int(
                    (valid_ratings < 4).sum()
                )

            # -------------------------------------------------
            # TASK TYPE DISTRIBUTION
            # -------------------------------------------------

            task_type_distribution = {
                "labels": [],
                "values": []
            }

            if "task_type" in df.columns:

                task_types = (
                    df["task_type"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unknown")
                    .value_counts()
                )

                task_type_distribution = {
                    "labels": [
                        str(value)
                        for value in task_types.index.tolist()
                    ],
                    "values": [
                        int(value)
                        for value in task_types.values.tolist()
                    ]
                }

            # -------------------------------------------------
            # PRIORITY DISTRIBUTION
            # -------------------------------------------------

            priority_distribution = {
                "labels": [],
                "values": []
            }

            if "priority" in df.columns:

                priority_series = (
                    df["priority"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unknown")
                )

                priority_counts = (
                    priority_series.value_counts()
                )

                preferred_order = [
                    "Low",
                    "Medium",
                    "High",
                    "Critical"
                ]

                ordered_labels = []

                for priority in preferred_order:
                    if priority in priority_counts.index:
                        ordered_labels.append(priority)

                for priority in priority_counts.index:
                    if priority not in ordered_labels:
                        ordered_labels.append(priority)

                priority_distribution = {
                    "labels": ordered_labels,
                    "values": [
                        int(priority_counts[label])
                        for label in ordered_labels
                    ]
                }

            # -------------------------------------------------
            # ERROR RISK DISTRIBUTION
            # -------------------------------------------------

            risk_labels = [
                "Minimal",
                "Very Low",
                "Low",
                "Moderate",
                "High",
                "Critical"
            ]

            risk_counts = [0, 0, 0, 0, 0, 0]

            if "error_risk" in df.columns:

                valid_risk = df["error_risk"].dropna()

                for risk in valid_risk:

                    try:
                        risk_value = float(risk)

                        if risk_value < 1:
                            index = 0
                        elif risk_value < 2:
                            index = 1
                        elif risk_value < 3:
                            index = 2
                        elif risk_value < 4:
                            index = 3
                        elif risk_value < 5:
                            index = 4
                        else:
                            index = 5

                        risk_counts[index] += 1

                    except (ValueError, TypeError):
                        continue

            # -------------------------------------------------
            # MONTHLY PERFORMANCE TREND
            # -------------------------------------------------

            trend_labels = []
            trend_values = []

            if (
                "task_given_date" in df.columns
                and "rating" in df.columns
            ):

                dates = pd.to_datetime(
                    df["task_given_date"],
                    errors="coerce"
                )

                trend_df = pd.DataFrame({
                    "date": dates,
                    "rating": df["rating"]
                }).dropna()

                if not trend_df.empty:

                    trend_df["month"] = (
                        trend_df["date"]
                        .dt.to_period("M")
                    )

                    monthly = (
                        trend_df
                        .groupby("month")["rating"]
                        .mean()
                        .sort_index()
                    )

                    monthly = monthly.tail(12)

                    trend_labels = [
                        str(period)
                        for period in monthly.index
                    ]

                    trend_values = [
                        round(float(value), 2)
                        for value in monthly.values
                    ]

            if not trend_labels and "rating" in df.columns:

                overall_rating = (
                    float(df["rating"].mean())
                    if df["rating"].notna().any()
                    else 0
                )

                trend_labels = ["Overall"]
                trend_values = [
                    round(overall_rating, 2)
                ]

            # -------------------------------------------------
            # EMPLOYEE PERFORMANCE
            # -------------------------------------------------

            employees = []

            if "employee_id" in df.columns:

                grouped = df.groupby(
                    "employee_id",
                    dropna=True
                )

                for employee_id, group in grouped:

                    employee_tasks = len(group)

                    employee_rating = (
                        group["rating"].mean()
                        if "rating" in group.columns
                        else 0
                    )

                    employee_risk = (
                        group["error_risk"].mean()
                        if "error_risk" in group.columns
                        else 0
                    )

                    employee_completion = 0

                    if "is_completed" in group.columns:

                        completed = (
                            group["is_completed"]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .isin([
                                "1",
                                "true",
                                "yes",
                                "y",
                                "completed"
                            ])
                            .sum()
                        )

                        employee_completion = (
                            completed / employee_tasks * 100
                            if employee_tasks > 0
                            else 0
                        )

                    if pd.isna(employee_rating):
                        employee_rating = 0

                    if pd.isna(employee_risk):
                        employee_risk = 0

                    if employee_rating >= 8:
                        category = "High Performer"
                    elif employee_rating >= 6:
                        category = "Good Performer"
                    elif employee_rating >= 4:
                        category = "Average Performer"
                    else:
                        category = "Needs Improvement"

                    if employee_risk < 1:
                        risk_level = "Minimal"
                    elif employee_risk < 2:
                        risk_level = "Very Low"
                    elif employee_risk < 3:
                        risk_level = "Low"
                    elif employee_risk < 4:
                        risk_level = "Moderate"
                    elif employee_risk < 5:
                        risk_level = "High"
                    else:
                        risk_level = "Critical"

                    employees.append({
                        "employee_id": str(employee_id),
                        "tasks": int(employee_tasks),
                        "avg_rating": round(
                            float(employee_rating),
                            2
                        ),
                        "completion": round(
                            float(employee_completion),
                            1
                        ),
                        "risk": round(
                            float(employee_risk),
                            2
                        ),
                        "risk_level": risk_level,
                        "category": category
                    })

                employees.sort(
                    key=lambda item: item["avg_rating"],
                    reverse=True
                )

            # -------------------------------------------------
            # RESPONSE
            # -------------------------------------------------

            return jsonify({
                "success": True,

                "stats": {
                    "total_employees": total_employees,
                    "total_tasks": total_tasks,
                    "avg_rating": round(avg_rating, 2),
                    "avg_error_risk": round(avg_error_risk, 2),
                    "completion_rate": round(
                        completion_rate,
                        1
                    )
                },

                "performance_trend": {
                    "labels": trend_labels,
                    "values": trend_values
                },

                "performance_distribution": {
                    "labels": [
                        "High Performer",
                        "Good Performer",
                        "Average Performer",
                        "Needs Improvement"
                    ],
                    "values": [
                        performance_counts["High Performer"],
                        performance_counts["Good Performer"],
                        performance_counts["Average Performer"],
                        performance_counts["Needs Improvement"]
                    ]
                },

                "task_type_distribution":
                    task_type_distribution,

                "priority_distribution":
                    priority_distribution,

                "risk_distribution": {
                    "labels": risk_labels,
                    "values": risk_counts
                },

                "employees": employees[:10]
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # CREATE TASK
    # =====================================================

    @app.route(
        "/create-task",
        methods=["GET", "POST"]
    )
    def create_task():

        if request.method == "GET":
            return render_template(
                "create_task.html"
            )

        try:

            data = request.get_json()

            if not data:
                return jsonify({
                    "success": False,
                    "error": "No task data received"
                }), 400

            # =================================================
            # READ VALUES
            # =================================================

            task_type = str(
                data.get("task_type", "")
            ).strip()

            priority = str(
                data.get("priority", "")
            ).strip()

            days_to_deadline = data.get(
                "days_to_deadline"
            )

            volume_metric = data.get(
                "volume_metric"
            )

            perceived_difficulty = str(
                data.get(
                    "perceived_difficulty",
                    ""
                )
            ).strip()

            error_risk = data.get(
                "error_risk"
            )

            dependency_score = data.get(
                "dependency_score"
            )

            task_date = str(
                data.get("task_date", "")
            ).strip()

            # =================================================
            # REQUIRED VALIDATION
            # =================================================

            if not task_type:
                return jsonify({
                    "success": False,
                    "error": "Task type is required"
                }), 400

            if not priority:
                return jsonify({
                    "success": False,
                    "error": "Priority is required"
                }), 400

            if days_to_deadline is None:
                return jsonify({
                    "success": False,
                    "error": "Days to deadline is required"
                }), 400

            if volume_metric is None:
                return jsonify({
                    "success": False,
                    "error": "Volume metric is required"
                }), 400

            if not perceived_difficulty:
                return jsonify({
                    "success": False,
                    "error": "Perceived difficulty is required"
                }), 400

            if error_risk is None:
                return jsonify({
                    "success": False,
                    "error": "Error risk is required"
                }), 400

            if dependency_score is None:
                return jsonify({
                    "success": False,
                    "error": "Dependency score is required"
                }), 400

            if not task_date:
                return jsonify({
                    "success": False,
                    "error": "Task date is required"
                }), 400

            # =================================================
            # NUMERIC CONVERSION
            # =================================================

            try:

                days_to_deadline = float(
                    days_to_deadline
                )

                volume_metric = float(
                    volume_metric
                )

                error_risk = float(
                    error_risk
                )

                dependency_score = float(
                    dependency_score
                )

            except (ValueError, TypeError):

                return jsonify({
                    "success": False,
                    "error":
                        "Numeric fields contain invalid values"
                }), 400

            # =================================================
            # DIFFICULTY CONVERSION
            # =================================================

            difficulty_text = str(
                perceived_difficulty
            ).strip()

            difficulty_lower = (
                difficulty_text.lower()
            )

            difficulty_map = {
                "low": 1.0,
                "easy": 1.0,
                "medium": 3.0,
                "moderate": 3.0,
                "high": 5.0,
                "hard": 5.0
            }

            if difficulty_lower in difficulty_map:

                difficulty_numeric = (
                    difficulty_map[
                        difficulty_lower
                    ]
                )

            else:

                try:
                    difficulty_numeric = float(
                        difficulty_text
                    )

                except (ValueError, TypeError):

                    return jsonify({
                        "success": False,
                        "error":
                            "Invalid perceived difficulty"
                    }), 400

            # =================================================
            # RANGE VALIDATION
            # =================================================

            if days_to_deadline < 0:
                return jsonify({
                    "success": False,
                    "error":
                        "Days to deadline cannot be negative"
                }), 400

            if volume_metric < 0:
                return jsonify({
                    "success": False,
                    "error":
                        "Volume metric cannot be negative"
                }), 400

            if error_risk < 0 or error_risk > 5:
                return jsonify({
                    "success": False,
                    "error":
                        "Error risk must be between 0 and 5"
                }), 400

            if (
                dependency_score < 0
                or dependency_score > 1
            ):
                return jsonify({
                    "success": False,
                    "error":
                        "Dependency score must be between 0 and 1"
                }), 400

            if (
                difficulty_numeric < 0
                or difficulty_numeric > 10
            ):
                return jsonify({
                    "success": False,
                    "error":
                        "Perceived difficulty must be between 0 and 10"
                }), 400

            # =================================================
            # CREATED TASKS FILE
            # =================================================

            tasks_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    "processed",
                    "created_tasks.csv"
                )
            )

            os.makedirs(
                os.path.dirname(tasks_file),
                exist_ok=True
            )

            # =================================================
            # GENERATE TASK ID
            # =================================================

            next_number = 1

            if os.path.exists(tasks_file):

                try:

                    with open(
                        tasks_file,
                        "r",
                        encoding="utf-8",
                        newline=""
                    ) as file:

                        reader = csv.DictReader(file)

                        for row in reader:

                            existing_id = str(
                                row.get(
                                    "task_id",
                                    ""
                                )
                            ).strip()

                            if existing_id.startswith("TSK-"):

                                try:

                                    number = int(
                                        existing_id.replace(
                                            "TSK-",
                                            ""
                                        )
                                    )

                                    next_number = max(
                                        next_number,
                                        number + 1
                                    )

                                except ValueError:
                                    pass

                except Exception:
                    next_number = 1

            task_id = f"TSK-{next_number:04d}"

            # =================================================
            # TASK OBJECT
            # =================================================

            task = {
                "task_id": task_id,
                "task_type": task_type,
                "priority": priority,
                "days_to_deadline": days_to_deadline,
                "volume_metric": volume_metric,
                "perceived_difficulty": difficulty_text,
                "perceived_difficulty_numeric":
                    difficulty_numeric,
                "error_risk": error_risk,
                "dependency_score": dependency_score,
                "task_date": task_date
            }

            # =================================================
            # CSV COLUMNS
            # =================================================

            fieldnames = [
                "task_id",
                "task_type",
                "priority",
                "days_to_deadline",
                "volume_metric",
                "perceived_difficulty",
                "perceived_difficulty_numeric",
                "error_risk",
                "dependency_score",
                "task_date"
            ]

            file_exists = os.path.exists(
                tasks_file
            )

            # =================================================
            # SAVE TASK
            # =================================================

            with open(
                tasks_file,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(task)

            return jsonify({
                "success": True,
                "message":
                    f"Task {task_id} created successfully",
                "task": task
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # TASK HISTORY PAGE
    # =====================================================

    @app.route("/tasks")
    def tasks_page():
        return render_template(
            "tasks.html"
        )


    # =====================================================
    # TASK HISTORY API
    # =====================================================

    @app.route("/api/tasks")
    def tasks_api():

        try:

            tasks_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    "processed",
                    "created_tasks.csv"
                )
            )

            tasks = []

            if os.path.exists(tasks_file):

                with open(
                    tasks_file,
                    "r",
                    encoding="utf-8",
                    newline=""
                ) as file:

                    reader = csv.DictReader(file)

                    for row in reader:

                        numeric_fields = [
                            "days_to_deadline",
                            "volume_metric",
                            "perceived_difficulty_numeric",
                            "error_risk",
                            "dependency_score"
                        ]

                        for field in numeric_fields:

                            if row.get(field) not in [
                                None,
                                ""
                            ]:

                                try:
                                    row[field] = float(
                                        row[field]
                                    )

                                except (
                                    ValueError,
                                    TypeError
                                ):
                                    pass

                        tasks.append(row)

            return jsonify({
                "success": True,
                "tasks": tasks,
                "total": len(tasks)
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # EDIT TASK API
    # =====================================================

    @app.route(
        "/api/tasks/<task_id>",
        methods=["PUT"]
    )
    def edit_task(task_id):

        try:

            task_id = str(task_id).strip()

            data = request.get_json(
                silent=True
            )

            if not data:
                return jsonify({
                    "success": False,
                    "error":
                        "No task data received"
                }), 400

            tasks_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    "processed",
                    "created_tasks.csv"
                )
            )

            if not os.path.exists(tasks_file):
                return jsonify({
                    "success": False,
                    "error": "Task file not found"
                }), 404

            with open(
                tasks_file,
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as file:

                reader = csv.DictReader(file)

                fieldnames = reader.fieldnames
                tasks = list(reader)

            if not fieldnames:
                return jsonify({
                    "success": False,
                    "error": "Task CSV is empty"
                }), 404

            required_fields = [
                "task_id",
                "task_type",
                "priority",
                "days_to_deadline",
                "volume_metric",
                "perceived_difficulty",
                "perceived_difficulty_numeric",
                "error_risk",
                "dependency_score",
                "task_date"
            ]

            for field in required_fields:
                if field not in fieldnames:
                    fieldnames.append(field)

            found = False
            updated_task = None

            for task in tasks:

                csv_task_id = str(
                    task.get("task_id", "")
                ).strip()

                if csv_task_id != task_id:
                    continue

                found = True

                # -----------------------------
                # Task Type
                # -----------------------------

                if "task_type" in data:

                    value = str(
                        data.get(
                            "task_type",
                            ""
                        )
                    ).strip()

                    if not value:
                        return jsonify({
                            "success": False,
                            "error":
                                "Task type is required"
                        }), 400

                    task["task_type"] = value

                # -----------------------------
                # Priority
                # -----------------------------

                if "priority" in data:

                    value = str(
                        data.get(
                            "priority",
                            ""
                        )
                    ).strip()

                    if not value:
                        return jsonify({
                            "success": False,
                            "error":
                                "Priority is required"
                        }), 400

                    task["priority"] = value

                # -----------------------------
                # Days to Deadline
                # -----------------------------

                if "days_to_deadline" in data:

                    try:
                        value = float(
                            data["days_to_deadline"]
                        )

                    except (ValueError, TypeError):

                        return jsonify({
                            "success": False,
                            "error":
                                "Invalid days to deadline"
                        }), 400

                    if value < 0:
                        return jsonify({
                            "success": False,
                            "error":
                                "Days to deadline cannot be negative"
                        }), 400

                    task["days_to_deadline"] = value

                # -----------------------------
                # Volume Metric
                # -----------------------------

                if "volume_metric" in data:

                    try:
                        value = float(
                            data["volume_metric"]
                        )

                    except (ValueError, TypeError):

                        return jsonify({
                            "success": False,
                            "error":
                                "Invalid volume metric"
                        }), 400

                    if value < 0:
                        return jsonify({
                            "success": False,
                            "error":
                                "Volume metric cannot be negative"
                        }), 400

                    task["volume_metric"] = value

                # -----------------------------
                # Perceived Difficulty
                # -----------------------------

                if "perceived_difficulty" in data:

                    difficulty_text = str(
                        data.get(
                            "perceived_difficulty",
                            ""
                        )
                    ).strip()

                    if not difficulty_text:
                        return jsonify({
                            "success": False,
                            "error":
                                "Perceived difficulty is required"
                        }), 400

                    difficulty_map = {
                        "low": 1.0,
                        "easy": 1.0,
                        "medium": 3.0,
                        "moderate": 3.0,
                        "high": 5.0,
                        "hard": 5.0
                    }

                    difficulty_lower = (
                        difficulty_text.lower()
                    )

                    if (
                        difficulty_lower
                        in difficulty_map
                    ):

                        difficulty_numeric = (
                            difficulty_map[
                                difficulty_lower
                            ]
                        )

                    else:

                        try:
                            difficulty_numeric = float(
                                difficulty_text
                            )

                        except (
                            ValueError,
                            TypeError
                        ):

                            return jsonify({
                                "success": False,
                                "error":
                                    "Invalid perceived difficulty"
                            }), 400

                    if (
                        difficulty_numeric < 0
                        or difficulty_numeric > 10
                    ):

                        return jsonify({
                            "success": False,
                            "error":
                                "Perceived difficulty must be between 0 and 10"
                        }), 400

                    task["perceived_difficulty"] = (
                        difficulty_text
                    )

                    task[
                        "perceived_difficulty_numeric"
                    ] = difficulty_numeric

                # -----------------------------
                # Error Risk
                # -----------------------------

                if "error_risk" in data:

                    try:
                        value = float(
                            data["error_risk"]
                        )

                    except (
                        ValueError,
                        TypeError
                    ):

                        return jsonify({
                            "success": False,
                            "error":
                                "Invalid error risk"
                        }), 400

                    if value < 0 or value > 5:

                        return jsonify({
                            "success": False,
                            "error":
                                "Error risk must be between 0 and 5"
                        }), 400

                    task["error_risk"] = value

                # -----------------------------
                # Dependency Score
                # -----------------------------

                if "dependency_score" in data:

                    try:
                        value = float(
                            data["dependency_score"]
                        )

                    except (
                        ValueError,
                        TypeError
                    ):

                        return jsonify({
                            "success": False,
                            "error":
                                "Invalid dependency score"
                        }), 400

                    if value < 0 or value > 1:

                        return jsonify({
                            "success": False,
                            "error":
                                "Dependency score must be between 0 and 1"
                        }), 400

                    task["dependency_score"] = value

                # -----------------------------
                # Task Date
                # -----------------------------

                if "task_date" in data:

                    value = str(
                        data.get(
                            "task_date",
                            ""
                        )
                    ).strip()

                    if not value:

                        return jsonify({
                            "success": False,
                            "error":
                                "Task date is required"
                        }), 400

                    task["task_date"] = value

                updated_task = task
                break

            if not found:

                return jsonify({
                    "success": False,
                    "error":
                        f"Task {task_id} not found"
                }), 404

            with open(
                tasks_file,
                "w",
                encoding="utf-8",
                newline=""
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                    extrasaction="ignore"
                )

                writer.writeheader()
                writer.writerows(tasks)

            return jsonify({
                "success": True,
                "message":
                    f"Task {task_id} updated successfully",
                "task": updated_task
            })

        except Exception as e:

            print(
                "EDIT TASK ERROR:",
                str(e)
            )

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # DELETE TASK API
    # =====================================================

    @app.route(
        "/api/tasks/<task_id>",
        methods=["DELETE"]
    )
    def delete_task(task_id):

        try:

            task_id = str(task_id).strip()

            tasks_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    "processed",
                    "created_tasks.csv"
                )
            )

            if not os.path.exists(tasks_file):

                return jsonify({
                    "success": False,
                    "error":
                        "Task file not found"
                }), 404

            with open(
                tasks_file,
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as file:

                reader = csv.DictReader(file)

                fieldnames = reader.fieldnames
                tasks = list(reader)

            if not fieldnames:

                return jsonify({
                    "success": False,
                    "error":
                        "Task CSV is empty"
                }), 404

            remaining_tasks = []
            deleted = False

            for task in tasks:

                csv_task_id = str(
                    task.get(
                        "task_id",
                        ""
                    )
                ).strip()

                if csv_task_id == task_id:

                    deleted = True
                    continue

                remaining_tasks.append(task)

            if not deleted:

                return jsonify({
                    "success": False,
                    "error":
                        f"Task {task_id} not found"
                }), 404

            with open(
                tasks_file,
                "w",
                encoding="utf-8",
                newline=""
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                    extrasaction="ignore"
                )

                writer.writeheader()
                writer.writerows(
                    remaining_tasks
                )

            print(
                f"TASK DELETED SUCCESSFULLY: {task_id}"
            )

            return jsonify({
                "success": True,
                "message":
                    f"Task {task_id} deleted successfully"
            })

        except Exception as e:

            print(
                "DELETE TASK ERROR:",
                str(e)
            )

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # EMPLOYEE DATA PAGE
    # =====================================================

    @app.route("/employee-data")
    def employee_data():
        return render_template(
            "employee_data.html"
        )


    # =====================================================
    # EMPLOYEE DATA API
    # =====================================================

    @app.route("/api/employee-data")
    def employee_data_api():

        import pandas as pd

        try:

            df = pd.read_csv(
                DATA_FILE
            )

            employee_data = []

            # =================================================
            # GROUP BY EMPLOYEE
            # =================================================

            for employee_id, group in df.groupby(
                "employee_id"
            ):

                total_tasks = len(
                    group
                )

                # =============================================
                # COMPLETED TASKS
                # =============================================

                completed_tasks = (
                    group["is_completed"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin([
                        "1",
                        "true",
                        "yes",
                        "y",
                        "completed"
                    ])
                    .sum()
                )

                # =============================================
                # COMPLETION RATE
                # =============================================

                completion_rate = (
                    (
                        completed_tasks
                        /
                        total_tasks
                    ) * 100
                    if total_tasks > 0
                    else 0
                )

                # =============================================
                # AVG RATING
                # =============================================

                avg_rating = pd.to_numeric(
                    group["rating"],
                    errors="coerce"
                ).mean()

                # =============================================
                # AVG ERROR RISK
                # =============================================

                avg_risk = pd.to_numeric(
                    group["error_risk"],
                    errors="coerce"
                ).mean()

                employee_data.append({
                    "employee_id": str(
                        employee_id
                    ),
                    "tasks": int(
                        total_tasks
                    ),
                    "avg_rating": (
                        round(
                            float(avg_rating),
                            2
                        )
                        if pd.notna(avg_rating)
                        else 0
                    ),
                    "completion": round(
                        float(completion_rate),
                        1
                    ),
                    "avg_risk": (
                        round(
                            float(avg_risk),
                            2
                        )
                        if pd.notna(avg_risk)
                        else 0
                    )
                })

            # =================================================
            # SORT
            # =================================================

            employee_data.sort(
                key=lambda x:
                    x["employee_id"]
            )

            # =================================================
            # RESPONSE
            # =================================================

            return jsonify({
                "success": True,
                "employees":
                    employee_data
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # REGISTER API
    # =====================================================

    @app.route(
        "/api/register",
        methods=["POST"]
    )
    def register_api():

        try:

            data = request.get_json()

            if not data:

                return jsonify({
                    "success": False,
                    "error":
                        "No registration data received"
                }), 400

            username = data.get(
                "username"
            )

            password = data.get(
                "password"
            )

            # =================================================
            # VALIDATION
            # =================================================

            if not username or not password:

                return jsonify({
                    "success": False,
                    "error":
                        "Username and password are required"
                }), 400

            # =================================================
            # EXISTING USER
            # =================================================

            existing_user = User.query.filter_by(
                username=username
            ).first()

            if existing_user:

                return jsonify({
                    "success": False,
                    "error":
                        "Username already exists"
                }), 400

            # =================================================
            # PASSWORD HASH
            # =================================================

            password_hash = (
                generate_password_hash(
                    password
                )
            )

            # =================================================
            # CREATE USER
            # =================================================

            user = User(
                username=username,
                password_hash=password_hash,
                role="manager"
            )

            db.session.add(
                user
            )

            db.session.commit()

            return jsonify({
                "success": True,
                "message":
                    "Registration successful"
            })

        except Exception as e:

            db.session.rollback()

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # LOGIN API
    # =====================================================

    @app.route(
        "/api/login",
        methods=["POST"]
    )
    def login_api():

        try:

            data = request.get_json()

            if not data:

                return jsonify({
                    "success": False,
                    "error":
                        "No login data received"
                }), 400

            username = data.get(
                "username"
            )

            password = data.get(
                "password"
            )

            # =================================================
            # VALIDATION
            # =================================================

            if not username or not password:

                return jsonify({
                    "success": False,
                    "error":
                        "Username and password are required"
                }), 400

            # =================================================
            # FIND USER
            # =================================================

            user = User.query.filter_by(
                username=username
            ).first()

            if not user:

                return jsonify({
                    "success": False,
                    "error":
                        "Invalid username or password"
                }), 401

            # =================================================
            # CHECK PASSWORD
            # =================================================

            if not check_password_hash(
                user.password_hash,
                password
            ):

                return jsonify({
                    "success": False,
                    "error":
                        "Invalid username or password"
                }), 401

            # =================================================
            # CREATE JWT
            # =================================================

            token = create_access_token(
                identity=str(
                    user.id
                )
            )

            return jsonify({
                "success": True,
                "token": token,
                "username": user.username,
                "role": user.role
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # TASK OPTIONS API
    # =====================================================

    @app.route("/api/task-options")
    def task_options():

        try:

            options = (
                get_task_options()
            )

            return jsonify({
                "success": True,
                "options": options
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # PREDICTION API
    # =====================================================

    @app.route(
        "/api/predict",
        methods=["POST"]
    )
    def predict_api():

        try:

            data = request.get_json()

            if not data:

                return jsonify({
                    "success": False,
                    "error":
                        "No prediction data received"
                }), 400

            result = predict_employee(
                data
            )

            return jsonify({
                "success": True,
                **result
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # DASHBOARD STATS API
    # =====================================================

    @app.route("/api/dashboard-stats")
    def dashboard_stats():

        import pandas as pd

        try:

            df = pd.read_csv(
                DATA_FILE
            )

            total_employees = (
                df["employee_id"]
                .nunique()
            )

            total_tasks = len(
                df
            )

            avg_rating = pd.to_numeric(
                df["rating"],
                errors="coerce"
            ).mean()

            avg_error_risk = pd.to_numeric(
                df["error_risk"],
                errors="coerce"
            ).mean()

            return jsonify({
                "success": True,
                "total_employees": int(
                    total_employees
                ),
                "total_tasks": int(
                    total_tasks
                ),
                "avg_rating": (
                    round(
                        float(avg_rating),
                        2
                    )
                    if pd.notna(avg_rating)
                    else 0
                ),
                "avg_error_risk": (
                    round(
                        float(avg_error_risk),
                        2
                    )
                    if pd.notna(avg_error_risk)
                    else 0
                )
            })

        except Exception as e:

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # PROTECTED API
    # =====================================================

    @app.route("/api/protected")
    @jwt_required()
    def protected():

        return jsonify({
            "success": True,
            "message":
                "Protected route accessed successfully"
        })
