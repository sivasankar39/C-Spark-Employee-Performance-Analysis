import pandas as pd


class EmployeePredictor:
    """
    Predicts the most suitable employee for a new task.

    Stage 1:
        Classification models identify employees predicted
        to complete the task.

    Stage 2:
        Regression models predict the performance rating of
        the eligible employees.

    Final:
        Employees are ranked by predicted rating and the
        highest-rated employee is recommended.
    """

    def __init__(
        self,
        best_classification_models,
        eligible_regression_models,
        regression_features,
        reg_scale_cols,
    ):
        self.best_classification_models = best_classification_models
        self.eligible_regression_models = eligible_regression_models
        self.regression_features = regression_features
        self.reg_scale_cols = reg_scale_cols

    def _create_task_input(self, task):
        """
        Convert a new task into the feature format required
        by the ML models.
        """

        given_dt = pd.to_datetime(task["task_given_date"])
        deadline_dt = pd.to_datetime(task["task_deadline"])

        days_to_deadline = (deadline_dt - given_dt).days

        task_input = pd.DataFrame([{
            "task_type_encoded": task["task_type_encoded"],
            "priority_encoded": task["priority_encoded"],
            "volume_metric": task["volume_metric"],
            "dependency_score": task["dependency_score"],
            "error_risk": task["error_risk"],
            "perceived_difficulty": task["perceived_difficulty"],
            "primary_skill_matching": task["primary_skill_matching"],
            "secondary_skill_matching": task["secondary_skill_matching"],
            "given_day_of_week": given_dt.dayofweek,
            "given_month": given_dt.month,
            "given_year": given_dt.year,
            "days_to_deadline": days_to_deadline,
        }])

        return task_input

    def _get_eligible_employees(self, task):
        """
        Stage 1:
        Use classification models to identify employees
        predicted to complete the task.
        """

        given_dt = pd.to_datetime(task["task_given_date"])
        deadline_dt = pd.to_datetime(task["task_deadline"])

        classification_features = [
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

        scale_cols = ["days_to_deadline"]

        days_to_deadline = (deadline_dt - given_dt).days

        task_input = pd.DataFrame([{
            "task_type_encoded": task["task_type_encoded"],
            "priority_encoded": task["priority_encoded"],
            "volume_metric": task["volume_metric"],
            "dependency_score": task["dependency_score"],
            "error_risk": task["error_risk"],
            "given_day_of_week": given_dt.dayofweek,
            "given_month": given_dt.month,
            "given_year": given_dt.year,
            "days_to_deadline": days_to_deadline,
        }])

        eligible_employees = set()

        for emp_id, clf_info in self.best_classification_models.items():

            model = clf_info["model"]
            scaler = clf_info["scaler"]

            X_clf = task_input[classification_features].copy()

            X_clf[scale_cols] = scaler.transform(
                task_input[scale_cols]
            )

            prediction = model.predict(X_clf)[0]

            if prediction == 1:
                eligible_employees.add(emp_id)

        return eligible_employees

    def _predict_ratings(self, task, eligible_employees):
        """
        Stage 2:
        Predict performance ratings for eligible employees
        using their regression models.
        """

        task_input = self._create_task_input(task)

        predicted_ratings = {}

        for emp_id in eligible_employees:

            if emp_id not in self.eligible_regression_models:
                continue

            reg_info = self.eligible_regression_models[emp_id]

            model = reg_info["model"]
            scaler = reg_info["scaler"]

            X_reg = task_input[self.regression_features].copy()

            X_reg[self.reg_scale_cols] = scaler.transform(
                task_input[self.reg_scale_cols]
            )

            predicted_rating = model.predict(X_reg)[0]

            predicted_ratings[emp_id] = round(
                predicted_rating, 2
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
        eligible_employees = self._get_eligible_employees(task)

        # If no employee is predicted to complete the task
        if not eligible_employees:
            return {
                "eligible_employees": [],
                "ranked_employees": [],
                "recommended_employee": None,
            }

        # Stage 2: Regression
        predicted_ratings = self._predict_ratings(
            task,
            eligible_employees
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
            "eligible_employees": list(eligible_employees),
            "ranked_employees": ranked_employees,
            "recommended_employee": recommended_employee,
        }