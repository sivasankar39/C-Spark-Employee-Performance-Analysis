import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_employee_task


def split_employee_tasks(
    df: pd.DataFrame,
    employee_column: str,
    target_column: str,
    feature_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    min_samples: int = 30,
    stratify_data: bool = True
):
    """
    Split each employee's tasks separately into training and testing sets.

    Each employee's tasks are split according to the specified test_size.
    Stratification can be used to preserve the target-class distribution
    for each employee.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed employee-task dataframe.

    employee_column : str
        Column containing employee IDs.

    target_column : str
        Target column to be predicted.

    feature_columns : list[str]
        Columns used as input features.

    test_size : float, default=0.2
        Proportion of each employee's tasks used for testing.

    random_state : int, default=42
        Ensures reproducible splitting.

    min_samples : int, default=30
        Minimum number of tasks required for an employee
        to be included in the split.

    stratify_data : bool, default=True
        Whether to preserve target-class proportions.

    Returns
    -------
    employee_splits : dict
        Dictionary containing train/test data for every employee.
    """

    employee_splits = {}

    # Check that required columns exist
    required_columns = [employee_column, target_column] + feature_columns

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataframe: {missing_columns}"
        )

    # Process each employee separately
    for employee_id in df[employee_column].unique():

        employee_df = df[
            df[employee_column] == employee_id
        ].copy()

        # Remove rows where target is missing
        employee_df = employee_df.dropna(
            subset=[target_column]
        )

        # Skip employees with insufficient data
        if len(employee_df) < min_samples:
            continue

        # Stratification requires at least two target classes
        if stratify_data and employee_df[target_column].nunique() < 2:
            continue

        # Separate features and target
        X = employee_df[feature_columns]
        y = employee_df[target_column]

        # Stratification
        stratify = y if stratify_data else None

        # 80:20 split for this employee
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        # Store the split
        employee_splits[employee_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    return employee_splits


if __name__ == "__main__":

    # --------------------------------------------------
    # Load raw data
    # --------------------------------------------------
    df = load_raw_data()

    # --------------------------------------------------
    # Preprocess data
    # --------------------------------------------------
    df = preprocess_employee_task(df)

    # --------------------------------------------------
    # Features used by the model
    # --------------------------------------------------
    feature_columns = [
        "task_type",
        "days_to_deadline",
        "priority",
        "perceived_difficulty",
        "primary_skill_matching",
        "secondary_skill_matching",
        "ternary_skill_matching",
        "volume_metric",
        "dependency_score",
        "error_risk",
        "given_day_of_week",
        "given_month",
        "given_year"
    ]

    # --------------------------------------------------
    # Target
    # --------------------------------------------------
    target_column = "is_completed"

    # --------------------------------------------------
    # Employee-wise 80:20 split
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Display split information
    # --------------------------------------------------
    print("\nEmployee-wise Train-Test Split")
    print("=" * 50)

    for employee_id, splits in employee_splits.items():

        print(f"\nEmployee {employee_id}:")

        print(
            f"  X_train: {splits['X_train'].shape}"
            f" | y_train: {splits['y_train'].shape}"
        )

        print(
            f"  X_test : {splits['X_test'].shape}"
            f" | y_test : {splits['y_test'].shape}"
        )

        print("\n  Training class distribution:")
        print(
            splits["y_train"]
            .value_counts(normalize=True)
        )

        print("\n  Testing class distribution:")
        print(
            splits["y_test"]
            .value_counts(normalize=True)
        )