import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import os
import pickle


warnings.filterwarnings("ignore")


def train_classification_models(
    employee_processed_data: dict,
    random_state: int = 42
):
    """
    Train and evaluate classification models for each employee.

    Models:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting

    Models are tuned using GridSearchCV with 5-fold Stratified CV.
    The best model for each employee is selected using F1-score.
    """

    # Candidate models and hyperparameter grids
    model_param_grids = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000,
                random_state=random_state
            ),
            {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"]
            },
        ),

        "DecisionTree": (
            DecisionTreeClassifier(
                random_state=random_state
            ),
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            },
        ),

        "RandomForest": (
            RandomForestClassifier(
                random_state=random_state
            ),
            {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None]
            },
        ),

        "GradientBoosting": (
            GradientBoostingClassifier(
                random_state=random_state
            ),
            {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            },
        ),
    }

    # Classification features
    feature_columns = [
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

    # 5-fold Stratified Cross Validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    trained_models = {}

    # Train separately for each employee
    for employee_id, data in employee_processed_data.items():

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        X_train_model = X_train[feature_columns]
        X_test_model = X_test[feature_columns]

        employee_results = {}

        best_model = None
        best_model_name = None
        best_f1 = -1

        print("=" * 60)
        print(f"TRAINING MODELS FOR EMPLOYEE: {employee_id}")
        print("=" * 60)

        # Train every candidate model
        for model_name, (model, param_grid) in model_param_grids.items():

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
            )

            grid_search.fit(X_train_model, y_train)

            tuned_model = grid_search.best_estimator_

            # Test prediction
            y_pred = tuned_model.predict(X_test_model)

            # Evaluation metrics
            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            # Probability for ROC-AUC
            if hasattr(tuned_model, "predict_proba"):
                y_probability = tuned_model.predict_proba(
                    X_test_model
                )[:, 1]
            else:
                y_probability = y_pred

            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(
                    y_test,
                    y_probability
                )
            else:
                roc_auc = np.nan

            # Confusion matrix
            cm = confusion_matrix(
                y_test,
                y_pred
            )

            # Classification report
            clf_report = classification_report(
                y_test,
                y_pred,
                zero_division=0
            )

            # Store model results
            employee_results[model_name] = {
                "model": tuned_model,
                "best_params": grid_search.best_params_,
                "confusion_matrix": cm,
                "classification_report": clf_report,
                "f1_score": f1,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
            }

            print(f"\n--- {model_name} ---")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"F1 Score       : {f1:.4f}")
            print(f"Accuracy       : {accuracy:.4f}")
            print(f"Precision      : {precision:.4f}")
            print(f"Recall         : {recall:.4f}")
            print(f"ROC-AUC        : {roc_auc:.4f}")
            print(f"Confusion Matrix:\n{cm}")

            # Select best model based on F1
            if f1 > best_f1:
                best_f1 = f1
                best_model = tuned_model
                best_model_name = model_name

        # Store all results for employee
        trained_models[employee_id] = {
            "models": employee_results,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_f1_score": best_f1,
        }

        # Save the best classification model and preprocessing objects
        os.makedirs("models/classification", exist_ok=True)

        classification_model_data = {
            "model": best_model,
            "model_name": best_model_name,
            "scaler": data["scaler"],
            "task_encoder": data["task_encoder"],
        }

        model_path = (
            f"models/classification/{employee_id}.pkl"
        )

        with open(model_path, "wb") as file:
            pickle.dump(classification_model_data, file)

        
        print("\n" + "-" * 60)
        print(f"BEST MODEL: {best_model_name}")
        print(f"BEST F1   : {best_f1:.4f}")
        print("-" * 60)

    return trained_models

if __name__ == "__main__":
    from src.synthetic_dataset.data_loader import load_raw_data
    from src.synthetic_dataset.preprocessing import preprocess_employee_task
    from src.synthetic_dataset.train_test_split import split_employee_tasks
    from src.synthetic_dataset.encoding_scaling import encode_and_scale

    df = load_raw_data()
    df = preprocess_employee_task(df)

    feature_columns = [
        "task_type",
        "priority",
        "volume_metric",
        "dependency_score",
        "error_risk",
        "given_day_of_week",
        "given_month",
        "given_year",
        "days_to_deadline",
    ]

    employee_splits = split_employee_tasks(
        df=df,
        employee_column="employee_id",
        target_column="is_completed",
        feature_columns=feature_columns,
    )

    employee_processed_data = encode_and_scale(
        employee_splits,
        scale_columns=["days_to_deadline"],
    )

    results = train_classification_models(
        employee_processed_data
    )

    print(
        f"Classification models trained for "
        f"{len(results)} employees."
    )