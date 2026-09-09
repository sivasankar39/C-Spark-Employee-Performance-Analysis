from src.synthetic_dataset.data_loader import load_raw_data
from src.synthetic_dataset.preprocessing import preprocess_employee_task
from src.synthetic_dataset.train_test_split import split_employee_tasks
from src.synthetic_dataset.encoding_scaling import encode_and_scale
from src.modeling.classification import train_classification_models
from src.modeling.regression import train_regression_models

def run_pipeline():
    # 1. Load raw data
    df = load_raw_data()

    # 2. Basic preprocessing
    df = preprocess_employee_task(df)

    # 3. Train/test split for classification
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

    # 4. Encoding and scaling
    employee_processed_data = encode_and_scale(
        employee_splits,
        scale_columns=["days_to_deadline"],
    )

    # 5. Classification
    classification_results = train_classification_models(
        employee_processed_data
    )

    # 6. Regression
    regression_results = train_regression_models(
        df=df
    )

    return classification_results, regression_results

if __name__ == "__main__":
    classification_results, regression_results = run_pipeline()

    print(
        f"Classification completed for "
        f"{len(classification_results)} employees."
    )

    print(
        f"Regression completed for "
        f"{len(regression_results)} employees."
    )