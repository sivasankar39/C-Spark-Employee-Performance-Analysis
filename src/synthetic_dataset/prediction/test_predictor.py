from src.pipeline import run_pipeline


# ============================================================
# LOAD SAVED MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING SAVED ML MODELS")
print("=" * 70)

predictor = run_pipeline()

print("EmployeePredictor created successfully.")


# ============================================================
# DUMMY TASKS FOR PREDICTION TESTING
# ============================================================

dummy_tasks = [

    {
        "name": "Dummy Task 1",
        "task_type": "Feature Development",
        "priority": "High",
        "volume_metric": 5,
        "dependency_score": 4,
        "error_risk": 3,
        "perceived_difficulty": 4,
        "primary_skill_matching": 0.95,
        "secondary_skill_matching": 0.70,
        "task_given_date": "2026-10-01",
        "task_deadline": "2026-10-06",
    },

    {
        "name": "Dummy Task 3",
        "task_type": "Bug Fix",
        "priority": "High",
        "volume_metric": 3,
        "dependency_score": 1,
        "error_risk": 3,
        "perceived_difficulty": 3,
        "primary_skill_matching": 0.88,
        "secondary_skill_matching": 0.65,
        "task_given_date": "2026-10-10",
        "task_deadline": "2026-10-13",
    },

    {
        "name": "Dummy Task 5",
        "task_type": "Bug Fix",
        "priority": "High",
        "volume_metric": 2,
        "dependency_score": 1,
        "error_risk": 3,
        "perceived_difficulty": 3,
        "primary_skill_matching": 0.92,
        "secondary_skill_matching": 0.75,
        "task_given_date": "2026-11-20",
        "task_deadline": "2026-11-22",
    },

    {
        "name": "Dummy Task 6",
        "task_type": "Feature Development",
        "priority": "Medium",
        "volume_metric": 4,
        "dependency_score": 3,
        "error_risk": 2,
        "perceived_difficulty": 3,
        "primary_skill_matching": 0.94,
        "secondary_skill_matching": 0.80,
        "task_given_date": "2026-12-01",
        "task_deadline": "2026-12-10",
    },

    {
        "name": "Dummy Task 7",
        "task_type": "Documentation",
        "priority": "Low",
        "volume_metric": 1,
        "dependency_score": 1,
        "error_risk": 0,
        "perceived_difficulty": 1,
        "primary_skill_matching": 0.85,
        "secondary_skill_matching": 0.50,
        "task_given_date": "2026-12-10",
        "task_deadline": "2026-12-25",
    },

    {
        "name": "Dummy Task 8",
        "task_type": "System Maintenance",
        "priority": "Medium",
        "volume_metric": 3,
        "dependency_score": 4,
        "error_risk": 2,
        "perceived_difficulty": 4,
        "primary_skill_matching": 0.96,
        "secondary_skill_matching": 0.70,
        "task_given_date": "2026-12-15",
        "task_deadline": "2026-12-23",
    },
]


# ============================================================
# TEST ALL DUMMY TASKS
# ============================================================

print("\n" + "=" * 70)
print("TESTING MULTIPLE EMPLOYEE PREDICTIONS")
print("=" * 70)


for task in dummy_tasks:

    task_name = task["name"]

    # Remove name before sending task to predictor
    prediction_task = {
        key: value
        for key, value in task.items()
        if key != "name"
    }

    print("\n" + "-" * 70)
    print(task_name)
    print("-" * 70)

    print("\nTask Input:")
    print(prediction_task)

    # Run prediction
    result = predictor.predict(prediction_task)

    # --------------------------------------------------------
    # Eligible employees
    # --------------------------------------------------------

    print("\nEligible Employees:")

    if result["eligible_employees"]:
        print(result["eligible_employees"])
    else:
        print("No eligible employees.")

    # --------------------------------------------------------
    # Ranked employees
    # --------------------------------------------------------

    print("\nRanked Employees:")

    if result["ranked_employees"]:

        for employee in result["ranked_employees"]:

            print(
                f"Employee: {employee['employee_id']} | "
                f"Predicted Rating: "
                f"{employee['predicted_rating']}"
            )

    else:

        print("No employees available for ranking.")

    # --------------------------------------------------------
    # Recommended employee
    # --------------------------------------------------------

    print("\nRecommended Employee:")

    if result["recommended_employee"]:

        recommended_employee = (
            result["recommended_employee"]
        )

        recommended_rating = next(
            employee["predicted_rating"]
            for employee in result["ranked_employees"]
            if employee["employee_id"]
            == recommended_employee
        )

        print(
            f"Employee: {recommended_employee} | "
            f"Predicted Rating: {recommended_rating}"
        )

    else:

        print("No employee recommended.")


# ============================================================
# TEST COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ALL DUMMY TASK TESTS COMPLETED")
print("=" * 70)