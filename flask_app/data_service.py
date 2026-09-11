import os
import glob
import pickle
import numpy as np
import pandas as pd

from src.pipeline import run_pipeline

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "employee_tasks_dataset_3.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "regression"
)

# Exact feature order used when the Random Forest models were trained.
RANDOM_FOREST_FEATURES = [
    "task_type_encoded",
    "priority_encoded",
    "volume_metric",
    "dependency_score",
    "error_risk",
    "perceived_difficulty",
    "primary_skill_matching",
    "secondary_skill_matching",
    "given_day_of_week",
    "given_month",
    "given_year",
    "days_to_deadline",
]


# =========================================================
# DATASET
# =========================================================

def load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found:\n{DATA_FILE}")
    return pd.read_csv(DATA_FILE)


# =========================================================
# TASK OPTIONS
# =========================================================

def get_task_options():
    df = load_dataset()

    task_types = (
        df["task_type"].dropna().astype(str).unique().tolist()
        if "task_type" in df.columns else []
    )

    priorities = (
        df["priority"].dropna().astype(str).unique().tolist()
        if "priority" in df.columns else []
    )

    return {
        "task_types": sorted(task_types),
        "priorities": sorted(priorities),
    }


# =========================================================
# CONVERSION HELPERS
# =========================================================

def difficulty_to_number(value):
    if value is None:
        return 3.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()

    mapping = {
        "low": 1.0,
        "easy": 1.0,
        "medium": 3.0,
        "moderate": 3.0,
        "high": 5.0,
        "hard": 5.0,
    }

    if text in mapping:
        return mapping[text]

    try:
        return float(text)
    except Exception:
        return 3.0


def priority_to_number(value):
    if value is None:
        return 2.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()

    mapping = {
        "low": 1.0,
        "medium": 2.0,
        "normal": 2.0,
        "high": 3.0,
        "critical": 4.0,
        "urgent": 4.0,
    }

    if text in mapping:
        return mapping[text]

    try:
        return float(text)
    except Exception:
        return 2.0


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


# =========================================================
# DATE FEATURES
# =========================================================

def get_date_features(data, df=None):
    """
    Reproduce the date features used by the training pipeline:
    given_day_of_week, given_month, given_year.

    If task_given_date is supplied by the request, use it.
    Otherwise use today's date. This keeps the prediction endpoint
    usable when the frontend does not send a date.
    """
    date_value = data.get("task_given_date") or data.get("task_date")

    if date_value:
        dt = pd.to_datetime(date_value, errors="coerce")
    else:
        dt = pd.Timestamp.today()

    if pd.isna(dt):
        dt = pd.Timestamp.today()

    return {
        "given_day_of_week": int(dt.dayofweek),
        "given_month": int(dt.month),
        "given_year": int(dt.year),
    }


# =========================================================
# MODEL LOADING
# =========================================================

def get_employee_model_path(employee_id):
    text = str(employee_id).strip()

    # Dataset IDs are expected to be EMP_001 ... EMP_020.
    if text.upper().startswith("EMP_"):
        filename = text.upper()
    elif text.lower().startswith("emp"):
        try:
            number = int(text[3:].replace("_", ""))
            filename = f"EMP_{number:03d}"
        except Exception:
            filename = text.upper()
    else:
        try:
            filename = f"EMP_{int(float(text)):03d}"
        except Exception:
            filename = text

    return os.path.join(MODEL_DIR, f"{filename}.pkl")


def load_employee_model(employee_id):
    path = get_employee_model_path(employee_id)

    if not os.path.exists(path):
        print(f"SYNQ: Employee model not found: {path}")
        return None

    try:
        with open(path, "rb") as file:
            model_data = pickle.load(file)

        if not isinstance(model_data, dict):
            print(f"SYNQ: Invalid model format: {path}")
            return None

        required = {"model", "scaler", "task_encoder"}
        missing = required - set(model_data.keys())

        if missing:
            print(f"SYNQ: Missing model components {missing}: {path}")
            return None

        return model_data

    except Exception as e:
        print(f"SYNQ: Employee model loading failed: {e}")
        return None


def count_available_models():
    if not os.path.isdir(MODEL_DIR):
        return 0

    return len(
        glob.glob(os.path.join(MODEL_DIR, "EMP_*.pkl"))
    )


# Backward-compatible generic loader.
def find_model_file():
    files = glob.glob(os.path.join(MODEL_DIR, "EMP_*.pkl"))
    return files[0] if files else None


def load_model():
    """
    Backward-compatible helper.

    The real prediction flow uses load_employee_model() because
    each employee has a separate trained model.
    """
    model_path = find_model_file()

    if model_path is None:
        return None

    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        print(f"SYNQ: Model loading failed: {e}")
        return None


# =========================================================
# RATING / RISK
# =========================================================

def get_rating_category(rating):
    rating = float(rating)

    if rating >= 8:
        return "High Performer"
    if rating >= 6:
        return "Good Performer"
    if rating >= 4:
        return "Average Performer"
    return "Needs Improvement"


def get_risk_level(error_risk):
    value = float(error_risk)

    if value <= 1:
        return "Low Risk"
    if value <= 2:
        return "Medium Risk"
    if value <= 3:
        return "Moderate Risk"
    if value <= 4:
        return "High Risk"
    return "Critical Risk"


def normalize_rating(value):
    try:
        value = float(value)
    except Exception:
        return 5.0

    # The trained target is kept as-is when already on a 1-10 scale.
    # If a model ever returns a 1-5 value, convert it to the UI's 1-10 scale.
    if value <= 5:
        value = value * 2

    return float(np.clip(value, 1, 10))


# =========================================================
# BEST FIT EMPLOYEE
# =========================================================

def find_best_employee(df, data):
    if "employee_id" not in df.columns:
        return "EMP_001"

    requested_task = str(
        data.get("task_type", "")
    ).strip().lower()

    requested_priority = str(
        data.get("priority", "")
    ).strip().lower()

    employee_scores = []

    for employee_id, group in df.groupby("employee_id"):
        score = 0.0

        # Rating performance: 50%
        if "rating" in group.columns:
            ratings = pd.to_numeric(
                group["rating"], errors="coerce"
            ).dropna()

            avg_rating = float(ratings.mean()) if len(ratings) else 5.0
        else:
            avg_rating = 5.0

        score += (avg_rating / 10.0) * 0.50

        # Task type experience: 25%
        experience_score = 0.0

        if "task_type" in group.columns and requested_task:
            task_matches = (
                group["task_type"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(requested_task)
                .sum()
            )
            experience_score = min(task_matches / 20.0, 1.0)

        score += experience_score * 0.25

        # Priority experience: 5%
        priority_score = 0.0

        if "priority" in group.columns and requested_priority:
            priority_matches = (
                group["priority"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(requested_priority)
                .sum()
            )
            priority_score = min(priority_matches / 20.0, 1.0)

        score += priority_score * 0.05

        # Completion performance: 20%
        speed_score = 0.5

        if "is_completed" in group.columns:
            completed = pd.to_numeric(
                group["is_completed"], errors="coerce"
            ).dropna()

            if len(completed):
                speed_score = float(completed.mean())

        score += speed_score * 0.20

        employee_scores.append((str(employee_id), score))

    if not employee_scores:
        return "EMP_001"

    employee_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return employee_scores[0][0]


# =========================================================
# TRAINED MODEL PREDICTION
# =========================================================

def model_rating_prediction(model_data, data):
    """
    Predict with the saved employee-specific RandomForest model.

    Training architecture:
      - task_type -> LabelEncoder, then +1
      - priority -> numeric mapping
      - days_to_deadline -> saved MinMaxScaler
      - remaining numeric features unchanged
    """
    if not model_data:
        return None

    try:
        model = model_data["model"]
        scaler = model_data["scaler"]
        task_encoder = model_data["task_encoder"]

        task_type = str(
            data.get("task_type", "")
        ).strip()

        if not task_type:
            raise ValueError("Task type is required")

        # Use the exact encoder fitted during training.
        known_tasks = list(task_encoder.classes_)

        # Case-insensitive matching while preserving the trained label.
        matched_task = None
        for task in known_tasks:
            if str(task).strip().lower() == task_type.lower():
                matched_task = task
                break

        if matched_task is None:
            raise ValueError(
                f"Unknown task type '{task_type}'. "
                f"Expected one of: {known_tasks}"
            )

        task_type_encoded = int(
            task_encoder.transform([matched_task])[0]
        ) + 1

        priority_encoded = priority_to_number(
            data.get("priority", "Medium")
        )

        volume_metric = safe_float(
            data.get("volume_metric"), 0.5
        )

        dependency_score = safe_float(
            data.get("dependency_score"), 0.0
        )

        error_risk = safe_float(
            data.get("error_risk"), 1.0
        )

        perceived_difficulty = difficulty_to_number(
            data.get("perceived_difficulty", "Medium")
        )

        primary_skill_matching = safe_float(
            data.get("primary_skill_matching"), 0.0
        )

        secondary_skill_matching = safe_float(
            data.get("secondary_skill_matching"), 0.0
        )

        date_features = get_date_features(data)

        days_to_deadline = safe_float(
            data.get("days_to_deadline"), 5.0
        )

        # Build features in the EXACT order used by training.
        input_df = pd.DataFrame([{
            "task_type_encoded": task_type_encoded,
            "priority_encoded": priority_encoded,
            "volume_metric": volume_metric,
            "dependency_score": dependency_score,
            "error_risk": error_risk,
            "perceived_difficulty": perceived_difficulty,
            "primary_skill_matching": primary_skill_matching,
            "secondary_skill_matching": secondary_skill_matching,
            "given_day_of_week": date_features["given_day_of_week"],
            "given_month": date_features["given_month"],
            "given_year": date_features["given_year"],
            "days_to_deadline": days_to_deadline,
        }], columns=RANDOM_FOREST_FEATURES)

        # Training scaled only days_to_deadline.
        input_df[["days_to_deadline"]] = scaler.transform(
            input_df[["days_to_deadline"]]
        )

        prediction = model.predict(input_df)

        if len(prediction) == 0:
            return None

        return normalize_rating(prediction[0])

    except Exception as e:
        print(f"SYNQ: Trained model prediction failed: {e}")
        return None


# =========================================================
# HISTORICAL FALLBACK
# =========================================================

def historical_rating_prediction(df, data):
    working = df.copy()

    if "task_type" in working.columns:
        matched = working[
            working["task_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                str(data.get("task_type", "")).strip().lower()
            )
        ]

        if len(matched) > 20:
            working = matched

    if (
        "priority" in working.columns
        and data.get("priority")
    ):
        priority_matched = working[
            working["priority"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                str(data.get("priority")).strip().lower()
            )
        ]

        if len(priority_matched) > 20:
            working = priority_matched

    if "rating" not in working.columns:
        return 5.0

    ratings = pd.to_numeric(
        working["rating"], errors="coerce"
    ).dropna()

    if len(ratings) == 0:
        return 5.0

    base_rating = float(ratings.mean())
    adjustment = 0.0

    difficulty = difficulty_to_number(
        data.get("perceived_difficulty", "Medium")
    )

    error_risk = safe_float(
        data.get("error_risk"), 1.0
    )

    dependency = safe_float(
        data.get("dependency_score"), 0.0
    )

    days_to_deadline = safe_float(
        data.get("days_to_deadline"), 5.0
    )

    volume_metric = safe_float(
        data.get("volume_metric"), 0.5
    )

    priority = priority_to_number(
        data.get("priority", "Medium")
    )

    if difficulty >= 5:
        adjustment -= 0.25
    elif difficulty <= 1:
        adjustment += 0.10

    adjustment -= (error_risk - 1) * 0.20
    adjustment -= dependency * 0.10

    if days_to_deadline <= 2:
        adjustment -= 0.20
    elif days_to_deadline >= 7:
        adjustment += 0.05

    if volume_metric > 0.80:
        adjustment -= 0.15
    elif volume_metric < 0.20:
        adjustment += 0.05

    if priority >= 4:
        adjustment -= 0.10
    elif priority == 1:
        adjustment += 0.05

    return float(
        np.clip(base_rating + adjustment, 1, 10)
    )

_predictor = None

def get_predictor():
    global _predictor

    if _predictor is None:
        _predictor = run_pipeline()

    return _predictor


# =========================================================
# MAIN PREDICTION
# =========================================================

def predict_employee(data):
    """
    Run the complete employee prediction pipeline.

    Stage 1:
        Classification models identify employees predicted
        to complete the task.

    Stage 2:
        Regression models predict ratings for eligible employees.

    Final:
        Employees are ranked by predicted rating and the
        highest-rated employee is recommended.
    """

    # Get the saved-model prediction pipeline.
    predictor = get_predictor()

    # The predictor expects task_given_date and task_deadline.
    # The frontend currently sends days_to_deadline instead.
    given_date = pd.Timestamp.today().normalize()

    days_to_deadline = safe_float(
        data.get("days_to_deadline"),
        5.0
    )

    deadline_date = given_date + pd.Timedelta(
        days=days_to_deadline
    )

    # Create the input expected by EmployeePredictor.
    prediction_data = dict(data)

    prediction_data["task_given_date"] = given_date
    prediction_data["task_deadline"] = deadline_date

    # Run Stage 1 -> Stage 2 -> ranking.
    result = predictor.predict(prediction_data)

    # Convert the predictor result into the format
    # expected by the existing frontend.
    ranked_employees = result.get(
        "ranked_employees",
        []
    )

    if not ranked_employees:
        return {
            "best_fit_employee": None,
            "predicted_rating": None,
            "category": "No Eligible Employee",
            "error_risk": round(
                float(
                    np.clip(
                        safe_float(
                            data.get("error_risk"),
                            1.0
                        ),
                        0,
                        5
                    )
                ),
                2
            ),
            "risk_level": get_risk_level(
                np.clip(
                    safe_float(
                        data.get("error_risk"),
                        1.0
                    ),
                    0,
                    5
                )
            ),
            "model_available": True,
            "fallback_used": False,
            "model_type": "Classification + Random Forest Regression",
            "eligible_employees": [],
            "ranked_employees": [],
        }

    best_employee = ranked_employees[0]

    predicted_rating = normalize_rating(
        best_employee["predicted_rating"]
    )

    error_risk = np.clip(
        safe_float(
            data.get("error_risk"),
            1.0
        ),
        0,
        5
    )

    return {
        "best_fit_employee": best_employee["employee_id"],
        "predicted_rating": round(
            float(predicted_rating),
            2
        ),
        "category": get_rating_category(
            predicted_rating
        ),
        "error_risk": round(
            float(error_risk),
            2
        ),
        "risk_level": get_risk_level(
            error_risk
        ),
        "model_available": True,
        "fallback_used": False,
        "model_type": "Classification + Random Forest Regression",
        "eligible_employees": result.get(
            "eligible_employees",
            []
        ),
        "ranked_employees": ranked_employees,
    }
