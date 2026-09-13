import os
import pickle


# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def load_classification_models():
    """
    Classification models are not currently used
    in the SYNQ project.

    Return an empty dictionary so that existing
    pipeline code remains compatible.
    """

    return {}


def load_regression_models():
    """
    Load all saved employee regression models.

    Models are stored in:
        models/regression/
    """

    models = {}

    model_folder = os.path.join(
        BASE_DIR,
        "models",
        "regression"
    )

    if not os.path.exists(model_folder):

        print(
            f"Warning: Regression model folder not found: "
            f"{model_folder}"
        )

        return models

    for filename in os.listdir(model_folder):

        if not filename.endswith(".pkl"):
            continue

        employee_id = filename.replace(
            ".pkl",
            ""
        )

        model_path = os.path.join(
            model_folder,
            filename
        )

        try:

            with open(
                model_path,
                "rb"
            ) as file:

                models[employee_id] = pickle.load(
                    file
                )

            print(
                f"Loaded regression model: "
                f"{employee_id}"
            )

        except Exception as e:

            print(
                f"Failed to load model "
                f"{employee_id}: {e}"
            )

    return models


def load_all_models():
    """
    Load all SYNQ ML models.

    Classification models are currently
    unavailable/not used.

    Employee-specific regression models
    are loaded from models/regression/.
    """

    classification_models = {}
    regression_models = load_regression_models()

    return (
        classification_models,
        regression_models
    )