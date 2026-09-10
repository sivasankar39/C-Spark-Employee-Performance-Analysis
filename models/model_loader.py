import os
import pickle


def load_classification_models():
    """
    Load all saved classification models.
    """

    models = {}

    model_folder = "models/classification"

    for filename in os.listdir(model_folder):

        if not filename.endswith(".pkl"):
            continue

        employee_id = filename.replace(".pkl", "")
        model_path = os.path.join(model_folder, filename)

        with open(model_path, "rb") as file:
            models[employee_id] = pickle.load(file)

    return models


def load_regression_models():
    """
    Load all saved regression models.
    """

    models = {}

    model_folder = "models/regression"

    for filename in os.listdir(model_folder):

        if not filename.endswith(".pkl"):
            continue

        employee_id = filename.replace(".pkl", "")
        model_path = os.path.join(model_folder, filename)

        with open(model_path, "rb") as file:
            models[employee_id] = pickle.load(file)

    return models


def load_all_models():
    """
    Load both classification and regression models.
    """

    classification_models = load_classification_models()
    regression_models = load_regression_models()

    return classification_models, regression_models


if __name__ == "__main__":

    classification_models, regression_models = load_all_models()

    print(
        f"Classification models loaded: "
        f"{len(classification_models)}"
    )

    print(
        f"Regression models loaded: "
        f"{len(regression_models)}"
    )
