# src/data_loader.py
import pandas as pd
from pathlib import Path

def load_raw_data(file_path: str | Path | None = None) -> pd.DataFrame:
    
    """Loads raw Emplyee task dataset from data/raw/ directory."""

    if file_path is None:
        file_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "employee_tasks_dataset_3.csv"
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path.resolve()}")
    
    df = pd.read_csv(path)
    print(f"Successfully loaded {len(df)} rows from {path}")
    return df

if __name__ == "__main__":
    # Test execution
    df = load_raw_data()
    print(df.head(2))
    print(df["rating"].dtype)