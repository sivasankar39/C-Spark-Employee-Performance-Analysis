import pandas as pd
from src.data_loader import load_raw_data
from pathlib import Path

def preprocess_employee_task(df: pd.DataFrame) -> pd.DataFrame:    
    """Preprocess the employee task dataset."""
    df = df.copy()

    # Remove duplicate rows
    print("Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates()

    # Remove unnecessary identifier column
    df = df.drop(columns=["task_id"])

    # Check missing values before handling
    print("\nMissing values before handling:")
    print(df.isna().sum())


    # Handle missing numerical input features
    numerical_columns = [
     "perceived_difficulty",
     "volume_metric",
     "dependency_score",
     "error_risk"
    ]

    for col in numerical_columns:
       if df[col].isna().sum() > 0:
           df[col] = df[col].fillna(df[col].median())


    # Handle categorical missing values
    categorical_columns = [
        "task_type",
        "primary_skill_matching",
        "secondary_skill_matching",
        "ternary_skill_matching"
    ]

    

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")
    
    # Check missing values after handling
    print("\nMissing values after handling:")
    print(df.isna().sum())

    # Convert date columns to datetime format
    df['task_given_date'] = pd.to_datetime(df['task_given_date'],  errors="coerce")

    # Remove rows with invalid/missing dates
    df = df.dropna(subset=["task_given_date"])

    # Feature engineering: date components
    df["given_day_of_week"] = df["task_given_date"].dt.dayofweek
    df["given_month"] = df["task_given_date"].dt.month
    df["given_year"] = df["task_given_date"].dt.year

    print("\nFinal missing values:")
    print(df.isna().sum())
   
    return df


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = preprocess_employee_task(raw_df)

    print(clean_df.head())

    # Save preprocessed data to intermediate directory
    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "intermediate"
        / "employee_task_intermediate.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to: {output_path}")
    print(f"Rows: {len(clean_df)}")
