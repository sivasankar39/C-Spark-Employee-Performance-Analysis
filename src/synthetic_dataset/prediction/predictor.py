import pandas as pd


class EmployeePredictor:
    """
    Predicts the most suitable employee for a new task.

    Stage 1:
        Classification models identify employees predicted
        to complete the task.

    Stage 2:
        Regression models predict the performance rating
        of eligible employees.

    Final:
        Employees are ranked by predicted rating and the
        highest-rated employee is recommended.
    """

    def __init__(
        self,
        classification_models,
        regression_models,
        ):
        self.classification_models = classification_models
        self.regression_models = regression_models

        self.classification_features = [
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

        self.regression_features = [
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

        self.scale_columns = [
            "days_to_deadline"
        ]

    def _difficulty_to_number(self, value):
        if value is None:
            return 3.0

        if isinstance(value, (int, float)):
            return float(value)

        mapping = {
          "low": 1.0,
          "easy": 1.0,
          "medium": 3.0,
          "moderate": 3.0,
         "high": 5.0,
          "hard": 5.0,
        }

        text = str(value).strip().lower()

        if text in mapping:
         return mapping[text]

        try:
         return float(text)
        except ValueError:
         return 3.0

    def _create_task_input(self, task):
        """
        Convert a new task into the feature format
        required by the ML models.
        """

        given_dt = pd.to_datetime(
            task["task_given_date"]
        )

        deadline_dt = pd.to_datetime(
            task["task_deadline"]
        )

        days_to_deadline = (
            deadline_dt - given_dt
        ).days

        task_input = pd.DataFrame([{
            "task_type": task["task_type"],
            "priority": task["priority"],
            "volume_metric": task["volume_metric"],
            "dependency_score": task["dependency_score"],
            "error_risk": task["error_risk"],
            "perceived_difficulty": self._difficulty_to_number(task["perceived_difficulty"]),
            "primary_skill_matching": task[
                "primary_skill_matching"
            ],
            "secondary_skill_matching": task[
                "secondary_skill_matching"
            ],
            "given_day_of_week": given_dt.dayofweek,
            "given_month": given_dt.month,
            "given_year": given_dt.year,
            "days_to_deadline": days_to_deadline,
        }])

        return task_input

    def _get_eligible_employees(self, task):
        """
        Stage 1:
        Use saved classification models to identify
        employees predicted to complete the task.
        """

        task_input = self._create_task_input(task)

        eligible_employees = set()

        priority_mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }

        for emp_id, model_info in self.classification_models.items():

            model = model_info["model"]
            scaler = model_info["scaler"]
            task_encoder = model_info["task_encoder"]

            X_clf = task_input.copy()

            # Priority encoding
            X_clf["priority_encoded"] = (
                X_clf["priority"]
                .map(priority_mapping)
                .fillna(1)
            )

            # Check task type
            task_type = X_clf["task_type"].iloc[0]

            if task_type not in task_encoder.classes_:
                continue

            # Task type encoding
            X_clf["task_type_encoded"] = (
                task_encoder.transform(
                    X_clf["task_type"]
                ) + 1
            )

            # Scale days_to_deadline
            X_clf[self.scale_columns] = (
                scaler.transform(
                    X_clf[self.scale_columns]
                )
            )

            # Remove original categorical columns
            X_clf = X_clf.drop(
                columns=[
                    "task_type",
                    "priority",
                ]
            )

            # Keep exact feature order
            X_clf = X_clf[
                self.classification_features
            ]

            prediction = model.predict(X_clf)[0]

            if prediction == 1:
                eligible_employees.add(emp_id)

        return eligible_employees

    def _predict_ratings(
        self,
        task,
        eligible_employees,
    ):
        """
        Stage 2:
        Predict performance ratings for eligible employees
        using saved Random Forest regression models.
        """

        task_input = self._create_task_input(task)

        predicted_ratings = {}

        priority_mapping = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }

        for emp_id in eligible_employees:

            if emp_id not in self.regression_models:
                continue

            model_info = self.regression_models[
                emp_id
            ]

            model = model_info["model"]
            scaler = model_info["scaler"]
            task_encoder = model_info["task_encoder"]

            X_reg = task_input.copy()

            # Priority encoding
            X_reg["priority_encoded"] = (
                X_reg["priority"]
                .map(priority_mapping)
                .fillna(1)
            )

            # Check task type
            task_type = X_reg["task_type"].iloc[0]

            if task_type not in task_encoder.classes_:
                continue

            # Task type encoding
            X_reg["task_type_encoded"] = (
                task_encoder.transform(
                    X_reg["task_type"]
                ) + 1
            )

            # Remove original categorical columns
            X_reg = X_reg.drop(
                columns=[
                    "task_type",
                    "priority",
                ]
            )

            # Scale days_to_deadline
            X_reg[self.scale_columns] = (
                scaler.transform(
                    X_reg[self.scale_columns]
                )
            )

            # Keep exact Random Forest feature order
            X_reg = X_reg[
                self.regression_features
            ]

            predicted_rating = model.predict(
                X_reg
            )[0]

            predicted_ratings[emp_id] = round(
                predicted_rating,
                2
            )

        return predicted_ratings

    def predict(self, task):
        """
        Run the complete employee prediction pipeline.

        Returns:
            Dictionary containing eligible employees,
            ranking and recommended employee.
        """

        # Stage 1: Classification
        eligible_employees = (
            self._get_eligible_employees(task)
        )

        # No eligible employees
        if not eligible_employees:
            return {
                "eligible_employees": [],
                "ranked_employees": [],
                "recommended_employee": None,
            }

        # Stage 2: Regression
        predicted_ratings = (
            self._predict_ratings(
                task,
                eligible_employees,
            )
        )

        # Rank employees by predicted rating
        rankings = sorted(
            predicted_ratings.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        ranked_employees = [
            {
                "employee_id": emp_id,
                "predicted_rating": rating,
            }
            for emp_id, rating in rankings
        ]

        recommended_employee = (
            ranked_employees[0]["employee_id"]
            if ranked_employees
            else None
        )

        return {
            "eligible_employees": list(
                eligible_employees
            ),
            "ranked_employees": ranked_employees,
            "recommended_employee": (
                recommended_employee
            ),
        }