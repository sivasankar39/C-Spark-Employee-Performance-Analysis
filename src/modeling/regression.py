import warnings

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import os
import pickle

warnings.filterwarnings("ignore")


def train_regression_models(
    df: pd.DataFrame,
    employee_column: str = "employee_id",
    target_column: str = "rating",
    test_size: float = 0.2,
    random_state: int = 42,
    min_samples: int = 10,
):
    """
    Train Linear Regression and Random Forest Regression models
    separately for each employee.

    Regression is performed only on completed tasks that have a rating.

    Linear Regression uses 9 features.
    Random Forest Regression uses 12 features.

    Encoders and scaler are fitted only on training data.
    """

    linear_regression_features = [
        "task_type_encoded",
        "priority_encoded",
        "volume_metric",
        "dependency_score",
        "error_risk",
        "given_day_of_week",
        "given_month",
        "given_year",
        "days_to_deadline",
    ]

    random_forest_features = [
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

    scale_columns = ["days_to_deadline"]

    # Only completed tasks with a rating are used for regression
    completed_df = df[
        (df["is_completed"] == 1)
        & (df[target_column].notna())
    ].copy()

    regression_results = {}

    for employee_id in completed_df[employee_column].unique():

        employee_df = completed_df[
            completed_df[employee_column] == employee_id
        ].copy()

        # Skip employees with insufficient data
        if len(employee_df) < min_samples:
            continue

        # Raw features needed before encoding
        raw_features = [
            "task_type",
            "priority",
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

        X = employee_df[raw_features].copy()
        y = employee_df[target_column].copy()

        # Split before encoding/scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        # -------------------------
        # Encoding
        # -------------------------

        # Priority encoding
        priority_mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }

        X_train["priority_encoded"] = (
            X_train["priority"]
            .map(priority_mapping)
            .fillna(1)
        )

        X_test["priority_encoded"] = (
            X_test["priority"]
            .map(priority_mapping)
            .fillna(1)
        )

        # Task type encoding
        task_encoder = LabelEncoder()

        X_train["task_type_encoded"] = (
            np.asarray(
                task_encoder.fit_transform(X_train["task_type"]),
                dtype=int
            ) + 1
        )

        unseen_tasks = set(X_test["task_type"]) - set(
            task_encoder.classes_
        )

        if unseen_tasks:
            raise ValueError(
                f"Unseen task types for employee {employee_id}: "
                f"{unseen_tasks}"
            )

        X_test["task_type_encoded"] = (
            np.asarray(
                task_encoder.transform(X_test["task_type"]),
                dtype=int
            ) + 1
        )

        # Remove original categorical columns
        X_train = X_train.drop(
            columns=["task_type", "priority"]
        )

        X_test = X_test.drop(
            columns=["task_type", "priority"]
        )

        # -------------------------
        # Scaling
        # -------------------------

        scaler = MinMaxScaler()

        X_train[scale_columns] = scaler.fit_transform(
            X_train[scale_columns]
        )

        X_test[scale_columns] = scaler.transform(
            X_test[scale_columns]
        )

        # -------------------------
        # Linear Regression
        # -------------------------

        linear_model = LinearRegression()

        linear_model.fit(
            X_train[linear_regression_features],
            y_train,
        )

        linear_predictions = linear_model.predict(
            X_test[linear_regression_features]
        )

        # -------------------------
        # Random Forest Regression
        # -------------------------

        random_forest_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
        )

        random_forest_model.fit(
            X_train[random_forest_features],
            y_train,
        )

        random_forest_predictions = random_forest_model.predict(
            X_test[random_forest_features]
        )

        # Store everything needed later for prediction
        regression_results[employee_id] = {
            "linear_regression": {
                "model": linear_model,
                "predictions": linear_predictions,
            },
            "random_forest": {
                "model": random_forest_model,
                "predictions": random_forest_predictions,
            },
            "scaler": scaler,
            "task_encoder": task_encoder,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        # Save the Random Forest regression model
        os.makedirs("models/regression", exist_ok=True)

        regression_model_data = {
            "model": random_forest_model,
            "scaler": scaler,
            "task_encoder": task_encoder,
        }

        model_path = (
            f"models/regression/{employee_id}.pkl"
        )

        with open(model_path, "wb") as file:
            pickle.dump(regression_model_data, file)

    return regression_results

if __name__ == "__main__":
    from src.synthetic_dataset.data_loader import load_raw_data
    from src.synthetic_dataset.preprocessing import preprocess_employee_task

    df = load_raw_data()
    df = preprocess_employee_task(df)

    results = train_regression_models(df)

    print(f"Regression models trained for {len(results)} employees.")