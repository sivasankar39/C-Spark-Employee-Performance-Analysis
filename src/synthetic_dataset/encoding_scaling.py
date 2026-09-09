from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import pandas as pd
import os
import numpy as np

from src.synthetic_dataset.data_loader import load_raw_data
from src.synthetic_dataset.preprocessing import preprocess_employee_task
from src.synthetic_dataset.train_test_split import split_employee_tasks


def encode_and_scale(
    employee_splits: dict,
    scale_columns: list[str] | None = None
):
    """
    Encode categorical features and scale numerical features
    separately for each employee.

    Encoders and scalers are fitted only on training data
    and then used to transform both training and testing data.

    Parameters
    ----------
    employee_splits : dict
        Dictionary containing employee-wise train/test splits.

    scale_columns : list[str], optional
        Numerical columns to scale.
        Default: ["days_to_deadline"]

    Returns
    -------
    employee_processed_data : dict
        Dictionary containing processed train/test data,
        targets, encoders and scalers for each employee.
    """

    if scale_columns is None:
        scale_columns = ["days_to_deadline"]

    employee_processed_data = {}

    # ======================================================
    # Process each employee separately
    # ======================================================

    for employee_id, splits in employee_splits.items():

        X_train = splits["X_train"].copy()
        X_test = splits["X_test"].copy()

        # ==================================================
        # 1. Encode priority
        # ==================================================

        priority_mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3
        }

        # Clean priority values
        X_train["priority"] = (
            X_train["priority"]
            .astype(str)
            .str.strip()
            .str.capitalize()
        )

        X_test["priority"] = (
            X_test["priority"]
            .astype(str)
            .str.strip()
            .str.capitalize()
        )

        # Apply ordinal mapping
        X_train["priority_encoded"] = (
            X_train["priority"]
            .map(priority_mapping)
        )

        X_test["priority_encoded"] = (
            X_test["priority"]
            .map(priority_mapping)
        )

        # Handle missing/unknown priority values
        X_train["priority_encoded"] = (
            X_train["priority_encoded"]
            .fillna(1)
            .astype(int)
        )

        X_test["priority_encoded"] = (
            X_test["priority_encoded"]
            .fillna(1)
            .astype(int)
        )

        # ==================================================
        # 2. Encode task_type
        # ==================================================

        task_encoder = LabelEncoder()

        # Fit ONLY on training data
        X_train["task_type_encoded"] = (
            np.asarray(
                task_encoder.fit_transform(
                    X_train["task_type"]
                ),
                dtype=int
            ) + 1
        )

        # Check for unseen task types in test data
        unseen_tasks = (
            set(X_test["task_type"])
            - set(task_encoder.classes_)
        )

        if unseen_tasks:
            raise ValueError(
                f"Employee {employee_id}: "
                f"X_test contains unseen task_type values: "
                f"{unseen_tasks}"
            )

        # Transform test using the training encoder
        X_test["task_type_encoded"] = (
            np.asarray(
                task_encoder.transform(
                    X_test["task_type"]
                ),
                dtype=int
            ) + 1
        )

        # ==================================================
        # 3. Drop original categorical columns
        # ==================================================

        X_train = X_train.drop(
            columns=["priority", "task_type"]
        )

        X_test = X_test.drop(
            columns=["priority", "task_type"]
        )

        # ==================================================
        # 4. Scale numerical features
        # ==================================================

        scaler = MinMaxScaler()

        # Fit ONLY on training data
        X_train[scale_columns] = scaler.fit_transform(
            X_train[scale_columns]
        )

        # Transform test using training scaler
        X_test[scale_columns] = scaler.transform(
            X_test[scale_columns]
        )

        # ==================================================
        # 5. Store processed data
        # ==================================================

        employee_processed_data[employee_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": splits["y_train"],
            "y_test": splits["y_test"],
            "task_encoder": task_encoder,
            "scaler": scaler
        }

    return employee_processed_data


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # 1. Load raw data
    # ======================================================

    df = load_raw_data()

    # ======================================================
    # 2. Preprocess data
    # ======================================================

    df = preprocess_employee_task(df)

    # ======================================================
    # 3. Define Stage 1 features
    # ======================================================

    feature_columns = [
        "task_type",
        "priority",
        "volume_metric",
        "dependency_score",
        "error_risk",
        "given_day_of_week",
        "given_month",
        "given_year",
        "days_to_deadline"
    ]

    target_column = "is_completed"

    # ======================================================
    # 4. Employee-wise train-test split
    # ======================================================

    employee_splits = split_employee_tasks(
        df=df,
        employee_column="employee_id",
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=0.2,
        random_state=42,
        min_samples=30,
        stratify_data=True
    )

    # ======================================================
    # 5. Encoding and scaling
    # ======================================================

    employee_processed_data = encode_and_scale(
        employee_splits=employee_splits,
        scale_columns=["days_to_deadline"]
    )

    # ======================================================
    # 6. Display results
    # ======================================================

    print("\nEncoding and Scaling Results")
    print("=" * 60)

    for employee_id, data in employee_processed_data.items():

        print(f"\nEmployee {employee_id}")

        print(
            f"  X_train: {data['X_train'].shape}"
            f" | y_train: {data['y_train'].shape}"
        )

        print(
            f"  X_test : {data['X_test'].shape}"
            f" | y_test : {data['y_test'].shape}"
        )

        print("\n  Task type classes:")
        print(data["task_encoder"].classes_)

        print("\n  Scaled days_to_deadline:")

        print(
            f"  Train min: "
            f"{data['X_train']['days_to_deadline'].min():.4f}"
        )

        print(
            f"  Train max: "
            f"{data['X_train']['days_to_deadline'].max():.4f}"
        )

        print("\n  X_train preview:")
        print(data["X_train"].head())

    # ==========================================================
    # 7. Combine all employees and save final processed CSV
    # ==========================================================

    processed_dataframes = []

    for employee_id, data in employee_processed_data.items():

        # ------------------------------------------------------
        # Training data
        # ------------------------------------------------------

        train_df = data["X_train"].copy()

        train_df["is_completed"] = data["y_train"].values
        train_df["employee_id"] = employee_id
        train_df["data_split"] = "train"

        # ------------------------------------------------------
        # Testing data
        # ------------------------------------------------------

        test_df = data["X_test"].copy()

        test_df["is_completed"] = data["y_test"].values
        test_df["employee_id"] = employee_id
        test_df["data_split"] = "test"

        # Add both datasets to the list
        processed_dataframes.append(train_df)
        processed_dataframes.append(test_df)

    # ==========================================================
    # 8. Combine all employees
    # ==========================================================

    final_processed_df = pd.concat(
        processed_dataframes,
        ignore_index=True
    )

    # ==========================================================
    # 9. Create output directory
    # ==========================================================

    output_directory = "data/processed"

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ==========================================================
    # 10. Save final CSV
    # ==========================================================

    output_path = os.path.join(
        output_directory,
        "employee_task_processed.csv"
    )

    final_processed_df.to_csv(
        output_path,
        index=False
    )

    # ==========================================================
    # 11. Display final CSV information
    # ==========================================================

    print("\n" + "=" * 60)
    print("FINAL PROCESSED DATASET")
    print("=" * 60)

    print(
        f"\nFile saved successfully:"
        f"\n{output_path}"
    )

    print(
        f"\nDataset shape: "
        f"{final_processed_df.shape}"
    )

    print("\nFinal columns:")
    print(
        final_processed_df.columns.tolist()
    )

    print("\nTrain/Test distribution:")
    print(
        final_processed_df["data_split"]
        .value_counts()
    )

    print("\nEmployee distribution:")
    print(
        final_processed_df["employee_id"]
        .value_counts()
    )

    print("\nFinal processed dataset preview:")
    print(
        final_processed_df.head()
    )
