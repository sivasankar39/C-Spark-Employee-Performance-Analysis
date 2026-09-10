from models.model_loader import load_all_models

from src.synthetic_dataset.prediction.predictor import EmployeePredictor


def run_pipeline():
    """
    Load saved ML models and create the employee predictor.
    """

    # Load saved classification and regression models
    classification_models, regression_models = load_all_models()

    # Create predictor using loaded models
    predictor = EmployeePredictor(
        classification_models=classification_models,
        regression_models=regression_models,
    )

    return predictor


if __name__ == "__main__":

    predictor = run_pipeline()

    print(
        "Employee prediction pipeline initialized "
        "using saved pickle models."
    )