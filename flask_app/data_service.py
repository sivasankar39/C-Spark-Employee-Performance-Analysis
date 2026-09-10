import os
import glob
import pickle
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "employee_tasks_dataset_3.csv"
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


# =========================================================
# FIND MODEL
# =========================================================

def find_model_file():

    possible_paths = [

        os.path.join(
            BASE_DIR,
            "employee_performance_model.pkl"
        ),

        os.path.join(
            BASE_DIR,
            "models",
            "employee_performance_model.pkl"
        ),

        os.path.join(
            BASE_DIR,
            "models",
            "model.pkl"
        ),

        os.path.join(
            BASE_DIR,
            "models",
            "performance_model.pkl"
        )
    ]

    for path in possible_paths:

        if os.path.exists(path):

            return path

    # Search recursively
    patterns = [

        os.path.join(
            BASE_DIR,
            "**",
            "*.pkl"
        ),

        os.path.join(
            BASE_DIR,
            "**",
            "*.joblib"
        )
    ]

    for pattern in patterns:

        files = glob.glob(
            pattern,
            recursive=True
        )

        if files:

            for file in files:

                filename = os.path.basename(
                    file
                ).lower()

                if (
                    "model" in filename
                    or
                    "performance" in filename
                    or
                    "employee" in filename
                ):

                    return file

            return files[0]

    return None


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    model_path = find_model_file()

    if model_path is None:

        print(
            "SYNQ: No trained model found."
        )

        return None

    try:

        with open(
            model_path,
            "rb"
        ) as file:

            model = pickle.load(file)

        print(
            "SYNQ: Model loaded:",
            model_path
        )

        return model

    except Exception as e:

        print(
            "SYNQ: Model loading failed:",
            e
        )

        return None


# =========================================================
# TASK OPTIONS
# =========================================================

def get_task_options():

    df = load_dataset()

    # Task types
    if "task_type" in df.columns:

        task_types = (
            df["task_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        task_types = []

    # Priorities
    if "priority" in df.columns:

        priorities = (
            df["priority"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        priorities = []

    return {

        "task_types":
            sorted(task_types),

        "priorities":
            sorted(priorities)

    }


# =========================================================
# DIFFICULTY TO NUMBER
# =========================================================

def difficulty_to_number(value):

    if value is None:

        return 3.0

    if isinstance(
        value,
        (int, float)
    ):

        return float(value)

    text = str(
        value
    ).strip().lower()

    mapping = {

        "low": 1.0,

        "easy": 1.0,

        "medium": 3.0,

        "moderate": 3.0,

        "high": 5.0,

        "hard": 5.0

    }

    if text in mapping:

        return mapping[text]

    try:

        return float(text)

    except Exception:

        return 3.0


# =========================================================
# PRIORITY TO NUMBER
# =========================================================

def priority_to_number(value):

    if value is None:

        return 2.0

    if isinstance(
        value,
        (int, float)
    ):

        return float(value)

    text = str(
        value
    ).strip().lower()

    mapping = {

        "low": 1.0,

        "medium": 2.0,

        "normal": 2.0,

        "high": 3.0,

        "critical": 4.0,

        "urgent": 4.0

    }

    if text in mapping:

        return mapping[text]

    try:

        return float(text)

    except Exception:

        return 2.0


# =========================================================
# RATING CATEGORY
# =========================================================

def get_rating_category(rating):

    rating = float(
        rating
    )

    if rating >= 8:

        return "High Performer"

    if rating >= 6:

        return "Good Performer"

    if rating >= 4:

        return "Average Performer"

    return "Needs Improvement"


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(error_risk):

    value = float(
        error_risk
    )

    if value <= 1:

        return "Low Risk"

    if value <= 2:

        return "Medium Risk"

    if value <= 3:

        return "Moderate Risk"

    if value <= 4:

        return "High Risk"

    return "Critical Risk"


# =========================================================
# NORMALIZE RATING
# =========================================================

def normalize_rating(value):

    try:

        value = float(value)

    except Exception:

        return 5.0

    # Convert 1-5 model output to 1-10
    if value <= 5:

        value = value * 2

    return float(
        np.clip(
            value,
            1,
            10
        )
    )


# =========================================================
# MODEL RATING PREDICTION
# =========================================================

def model_rating_prediction(
    model,
    data
):

    if model is None:

        return None

    # Convert values safely
    difficulty = difficulty_to_number(
        data.get(
            "perceived_difficulty",
            "Medium"
        )
    )

    try:

        error_risk = float(
            data.get(
                "error_risk",
                1
            )
        )

    except Exception:

        error_risk = 1.0

    try:

        dependency = float(
            data.get(
                "dependency_score",
                0
            )
        )

    except Exception:

        dependency = 0.0

    try:

        days_to_deadline = float(
            data.get(
                "days_to_deadline",
                5
            )
        )

    except Exception:

        days_to_deadline = 5.0

    try:

        volume_metric = float(
            data.get(
                "volume_metric",
                0.5
            )
        )

    except Exception:

        volume_metric = 0.5

    # Skill matching values
    try:

        primary_skill = float(
            data.get(
                "primary_skill_matching",
                0
            )
        )

    except Exception:

        primary_skill = 0.0

    try:

        secondary_skill = float(
            data.get(
                "secondary_skill_matching",
                0
            )
        )

    except Exception:

        secondary_skill = 0.0

    try:

        ternary_skill = float(
            data.get(
                "ternary_skill_matching",
                0
            )
        )

    except Exception:

        ternary_skill = 0.0

    # =====================================================
    # MODEL INPUT
    # =====================================================

    input_data = {

        "task_type":
            data.get(
                "task_type",
                ""
            ),

        "priority":
            data.get(
                "priority",
                ""
            ),

        "days_to_deadline":
            days_to_deadline,

        "volume_metric":
            volume_metric,

        "perceived_difficulty":
            difficulty,

        "error_risk":
            error_risk,

        "primary_skill_matching":
            primary_skill,

        "secondary_skill_matching":
            secondary_skill,

        "ternary_skill_matching":
            ternary_skill,

        "dependency_score":
            dependency

    }

    input_df = pd.DataFrame(
        [input_data]
    )

    # =====================================================
    # TRY MODEL
    # =====================================================

    try:

        prediction = model.predict(
            input_df
        )

        return normalize_rating(
            prediction[0]
        )

    except Exception as e:

        print(
            "Model prediction failed:",
            e
        )

        return None


# =========================================================
# HISTORICAL PREDICTION
# =========================================================

def historical_rating_prediction(
    df,
    data
):

    working = df.copy()

    # =====================================================
    # MATCH TASK TYPE
    # =====================================================

    if "task_type" in working.columns:

        matched = working[
            working["task_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            str(
                data.get(
                    "task_type",
                    ""
                )
            )
            .strip()
            .lower()
        ]

        if len(matched) > 20:

            working = matched

    # =====================================================
    # MATCH PRIORITY
    # =====================================================

    if (
        "priority" in working.columns
        and
        data.get("priority")
    ):

        priority_matched = working[
            working["priority"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            str(
                data.get(
                    "priority"
                )
            )
            .strip()
            .lower()
        ]

        if len(priority_matched) > 20:

            working = priority_matched

    # =====================================================
    # RATING
    # =====================================================

    if "rating" not in working.columns:

        return 5.0

    ratings = pd.to_numeric(
        working["rating"],
        errors="coerce"
    ).dropna()

    if len(ratings) == 0:

        return 5.0

    base_rating = float(
        ratings.mean()
    )

    # =====================================================
    # TASK CHARACTERISTICS
    # =====================================================

    adjustment = 0.0

    difficulty = difficulty_to_number(
        data.get(
            "perceived_difficulty",
            "Medium"
        )
    )

    try:

        error_risk = float(
            data.get(
                "error_risk",
                1
            )
        )

    except Exception:

        error_risk = 1.0

    try:

        dependency = float(
            data.get(
                "dependency_score",
                0
            )
        )

    except Exception:

        dependency = 0.0

    try:

        days_to_deadline = float(
            data.get(
                "days_to_deadline",
                5
            )
        )

    except Exception:

        days_to_deadline = 5.0

    try:

        volume_metric = float(
            data.get(
                "volume_metric",
                0.5
            )
        )

    except Exception:

        volume_metric = 0.5

    priority = priority_to_number(
        data.get(
            "priority",
            "Medium"
        )
    )

    # =====================================================
    # DIFFICULTY EFFECT
    # =====================================================

    if difficulty >= 5:

        adjustment -= 0.25

    elif difficulty <= 1:

        adjustment += 0.10

    # =====================================================
    # ERROR RISK EFFECT
    # =====================================================

    adjustment -= (
        error_risk - 1
    ) * 0.20

    # =====================================================
    # DEPENDENCY EFFECT
    # =====================================================

    adjustment -= (
        dependency * 0.10
    )

    # =====================================================
    # DEADLINE EFFECT
    # =====================================================

    if days_to_deadline <= 2:

        adjustment -= 0.20

    elif days_to_deadline >= 7:

        adjustment += 0.05

    # =====================================================
    # VOLUME EFFECT
    # =====================================================

    if volume_metric > 0.80:

        adjustment -= 0.15

    elif volume_metric < 0.20:

        adjustment += 0.05

    # =====================================================
    # PRIORITY EFFECT
    # =====================================================

    if priority >= 4:

        adjustment -= 0.10

    elif priority == 1:

        adjustment += 0.05

    # =====================================================
    # FINAL RATING
    # =====================================================

    return float(
        np.clip(
            base_rating + adjustment,
            1,
            10
        )
    )


# =========================================================
# BEST FIT EMPLOYEE
# =========================================================

def find_best_employee(
    df,
    data
):

    if "employee_id" not in df.columns:

        return "Emp 001"

    employee_scores = []

    requested_task = str(
        data.get(
            "task_type",
            ""
        )
    ).strip().lower()

    requested_priority = str(
        data.get(
            "priority",
            ""
        )
    ).strip().lower()

    # =====================================================
    # GROUP EMPLOYEES
    # =====================================================

    for employee_id, group in df.groupby(
        "employee_id"
    ):

        score = 0.0

        # =================================================
        # RATING SCORE
        # =================================================

        if "rating" in group.columns:

            ratings = pd.to_numeric(
                group["rating"],
                errors="coerce"
            ).dropna()

            if len(ratings):

                avg_rating = float(
                    ratings.mean()
                )

            else:

                avg_rating = 5.0

        else:

            avg_rating = 5.0

        rating_score = (
            avg_rating / 10
        )

        score += (
            rating_score * 0.50
        )

        # =================================================
        # TASK EXPERIENCE
        # =================================================

        experience_score = 0.0

        if "task_type" in group.columns:

            task_matches = (
                group["task_type"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(
                    requested_task
                )
                .sum()
            )

            experience_score = min(
                task_matches / 20,
                1
            )

        score += (
            experience_score * 0.25
        )

        # =================================================
        # PRIORITY EXPERIENCE
        # =================================================

        priority_score = 0.0

        if (
            "priority" in group.columns
            and
            requested_priority
        ):

            priority_matches = (
                group["priority"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(
                    requested_priority
                )
                .sum()
            )

            priority_score = min(
                priority_matches / 20,
                1
            )

        score += (
            priority_score * 0.05
        )

        # =================================================
        # COMPLETION PERFORMANCE
        # =================================================

        speed_score = 0.5

        if "is_completed" in group.columns:

            completed = pd.to_numeric(
                group["is_completed"],
                errors="coerce"
            )

            if len(completed):

                speed_score = (
                    completed.mean()
                )

        score += (
            speed_score * 0.20
        )

        # =================================================
        # STORE SCORE
        # =================================================

        employee_scores.append(
            (
                employee_id,
                score
            )
        )

    # =====================================================
    # NO EMPLOYEES
    # =====================================================

    if not employee_scores:

        return "Emp 001"

    # =====================================================
    # SORT
    # =====================================================

    employee_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_employee = (
        employee_scores[0][0]
    )

    # =====================================================
    # FORMAT EMPLOYEE ID
    # =====================================================

    text = str(
        best_employee
    )

    if text.lower().startswith("emp"):

        return text

    try:

        number = int(
            float(text)
        )

        return f"Emp {number:03d}"

    except Exception:

        return text


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict_employee(
    data
):

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = load_dataset()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = load_model()

    # =====================================================
    # BEST EMPLOYEE
    # =====================================================

    best_employee = find_best_employee(
        df,
        data
    )

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    predicted_rating = (
        model_rating_prediction(
            model,
            data
        )
    )

    # =====================================================
    # HISTORICAL FALLBACK
    # =====================================================

    if predicted_rating is None:

        predicted_rating = (
            historical_rating_prediction(
                df,
                data
            )
        )

    # =====================================================
    # ERROR RISK
    # =====================================================

    try:

        error_risk = float(
            data.get(
                "error_risk",
                1
            )
        )

    except Exception:

        error_risk = 1.0

    error_risk = float(
        np.clip(
            error_risk,
            0,
            5
        )
    )

    # =====================================================
    # CATEGORY
    # =====================================================

    category = get_rating_category(
        predicted_rating
    )

    # =====================================================
    # RISK LEVEL
    # =====================================================

    risk_level = get_risk_level(
        error_risk
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "best_fit_employee":
            best_employee,

        "predicted_rating":
            round(
                float(
                    predicted_rating
                ),
                2
            ),

        "category":
            category,

        "error_risk":
            round(
                error_risk,
                2
            ),

        "risk_level":
            risk_level,

        "model_available":
            model is not None

    }