import os
import glob
import pickle
import numpy as np
import pandas as pd


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
        if "task_type" in df.columns
        else []
    )

    priorities = (
        df["priority"].dropna().astype(str).unique().tolist()
        if "priority" in df.columns
        else []
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
    Otherwise use today's date.
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

    # Dataset/model target is on a 1-10 scale.
    # Only convert if a future model actually returns a 1-5 score.
    if value <= 5:
        value = value * 2

    return float(np.clip(value, 1, 10))


# =========================================================
# BUILD MODEL INPUT
# =========================================================

def build_model_input(model_data, data):
    """
    Build exactly the 12 features used by the trained
    employee-specific Random Forest models.
    """

    task_encoder = model_data["task_encoder"]
    scaler = model_data["scaler"]

    task_type = str(
        data.get("task_type", "")
    ).strip()

    if not task_type:
        raise ValueError("Task type is required")

    known_tasks = list(task_encoder.classes_)

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

    # Training used LabelEncoder followed by +1.
    task_type_encoded = int(
        task_encoder.transform([matched_task])[0]
    ) + 1

    priority_encoded = priority_to_number(
        data.get("priority", "Medium")
    )

    volume_metric = safe_float(
        data.get("volume_metric"),
        0.5
    )

    dependency_score = safe_float(
        data.get("dependency_score"),
        0.0
    )

    error_risk = safe_float(
        data.get("error_risk"),
        1.0
    )

    perceived_difficulty = difficulty_to_number(
        data.get("perceived_difficulty", "Medium")
    )

    primary_skill_matching = safe_float(
        data.get("primary_skill_matching"),
        0.0
    )

    secondary_skill_matching = safe_float(
        data.get("secondary_skill_matching"),
        0.0
    )

    date_features = get_date_features(data)

    days_to_deadline = safe_float(
        data.get("days_to_deadline"),
        5.0
    )

    input_df = pd.DataFrame(
        [{
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
        }],
        columns=RANDOM_FOREST_FEATURES
    )

    # Training scaled ONLY days_to_deadline.
    input_df[["days_to_deadline"]] = scaler.transform(
        input_df[["days_to_deadline"]]
    )

    return input_df


# =========================================================
# TRAINED MODEL PREDICTION
# =========================================================

def model_rating_prediction(model_data, data):
    """
    Predict rating using one employee's saved Random Forest model.
    """

    if not model_data:
        return None

    try:
        model = model_data["model"]

        input_df = build_model_input(
            model_data,
            data
        )

        prediction = model.predict(input_df)

        if len(prediction) == 0:
            return None

        return normalize_rating(prediction[0])

    except Exception as e:
        print(
            "SYNQ: Trained model prediction failed:",
            e
        )
        return None


# =========================================================
# BEST FIT EMPLOYEE
# =========================================================

def find_best_employee(df, data):
    """
    Select the best employee from the 20 available employee
    regression models.

    The employee with the highest trained-model predicted
    rating is selected.
    """

    employee_ids = []

    # Prefer employee IDs from the dataset.
    if "employee_id" in df.columns:
        employee_ids = (
            df["employee_id"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    # Also include every available trained model.
    model_files = glob.glob(
        os.path.join(MODEL_DIR, "EMP_*.pkl")
    )

    for model_file in model_files:
        filename = os.path.basename(model_file)
        employee_id = os.path.splitext(filename)[0]

        if employee_id not in employee_ids:
            employee_ids.append(employee_id)

    employee_ids = sorted(
        employee_ids,
        key=lambda x: (
            0,
            int(x.split("_")[-1])
        )
        if x.upper().startswith("EMP_")
        and x.split("_")[-1].isdigit()
        else (1, x)
    )

    return employee_ids


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
                str(
                    data.get("task_type", "")
                ).strip().lower()
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
                str(
                    data.get("priority")
                ).strip().lower()
            )
        ]

        if len(priority_matched) > 20:
            working = priority_matched

    if "rating" not in working.columns:
        return 5.0

    ratings = pd.to_numeric(
        working["rating"],
        errors="coerce"
    ).dropna()

    if len(ratings) == 0:
        return 5.0

    base_rating = float(ratings.mean())
    adjustment = 0.0

    difficulty = difficulty_to_number(
        data.get("perceived_difficulty", "Medium")
    )

    error_risk = safe_float(
        data.get("error_risk"),
        1.0
    )

    dependency = safe_float(
        data.get("dependency_score"),
        0.0
    )

    days_to_deadline = safe_float(
        data.get("days_to_deadline"),
        5.0
    )

    volume_metric = safe_float(
        data.get("volume_metric"),
        0.5
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
        np.clip(
            base_rating + adjustment,
            1,
            10
        )
    )


# =========================================================
# MAIN PREDICTION
# =========================================================

def predict_employee(data):
    """
    Predict the best employee using the 20 saved
    employee-specific Random Forest regression models.

    There is NO dependency on classification models here.

    Process:
        1. Load dataset.
        2. Find available employee IDs/models.
        3. Run the same task through each employee model.
        4. Rank employees by predicted rating.
        5. Return the highest-rated employee.
    """

    df = load_dataset()

    employee_ids = find_best_employee(
        df,
        data
    )

    ranked = []
    failed_models = []

    for employee_id in employee_ids:

        model_data = load_employee_model(
            employee_id
        )

        if model_data is None:
            failed_models.append(
                str(employee_id)
            )
            continue

        predicted_rating = model_rating_prediction(
            model_data,
            data
        )

        if predicted_rating is None:
            failed_models.append(
                str(employee_id)
            )
            continue

        ranked.append({
            "employee_id": str(employee_id),
            "predicted_rating": round(
                float(predicted_rating),
                2
            ),
            "category": get_rating_category(
                predicted_rating
            ),
        })

    # Highest predicted rating first.
    ranked.sort(
        key=lambda item: item["predicted_rating"],
        reverse=True
    )

    error_risk = float(
        np.clip(
            safe_float(
                data.get("error_risk"),
                1.0
            ),
            0,
            5
        )
    )

    # ---------------------------------------------------------
    # No model could predict
    # ---------------------------------------------------------

    if not ranked:

        fallback_rating = historical_rating_prediction(
            df,
            data
        )

        # If there are no trained models at all, still provide
        # a usable fallback employee.
        fallback_employee = (
            "EMP_001"
            if "EMP_001" in employee_ids
            else (
                employee_ids[0]
                if employee_ids
                else "EMP_001"
            )
        )

        return {
            "best_fit_employee": fallback_employee,
            "predicted_rating": round(
                float(fallback_rating),
                2
            ),
            "category": get_rating_category(
                fallback_rating
            ),
            "error_risk": round(
                error_risk,
                2
            ),
            "risk_level": get_risk_level(
                error_risk
            ),
            "model_available": False,
            "fallback_used": True,
            "model_type": "Historical Fallback",
            "eligible_employees": [
                str(x) for x in employee_ids
            ],
            "ranked_employees": [],
            "failed_models": failed_models,
        }

    # ---------------------------------------------------------
    # Best employee
    # ---------------------------------------------------------

    best_employee = ranked[0]

    return {
        "best_fit_employee": best_employee["employee_id"],
        "predicted_rating": round(
            float(best_employee["predicted_rating"]),
            2
        ),
        "category": best_employee["category"],
        "error_risk": round(
            error_risk,
            2
        ),
        "risk_level": get_risk_level(
            error_risk
        ),
        "model_available": True,
        "fallback_used": False,
        "model_type": "Random Forest Regression",
        "eligible_employees": [
            item["employee_id"]
            for item in ranked
        ],
        "ranked_employees": ranked,
        "failed_models": failed_models,
    }
