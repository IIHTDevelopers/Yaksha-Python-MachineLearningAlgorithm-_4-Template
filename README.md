# Machine Learning Assessment

This assessment consists of two Python files with skeleton code that you need to implement:
1. `employee.py` - Employee attrition prediction using logistic regression
2. `student.py` - Student score prediction using linear regression

## Dataset Information

### Employee Attrition Dataset (`employee_attrition.csv`)

This dataset contains information about employees and whether they left the company (attrition).

Key columns:
- `satisfaction_level`: Employee satisfaction level (0-1)
- `last_evaluation`: Score of last performance evaluation (0-1)
- `number_project`: Number of projects the employee is involved in
- `average_monthly_hours`: Average monthly working hours
- `time_spend_company`: Years at the company
- `work_accident`: Whether the employee had a workplace accident (0/1)
- `left`: Whether the employee left the company (0/1) - This is the target variable
- `promotion_last_5years`: Whether the employee was promoted in the last 5 years (0/1)
- `department`: Department the employee works in
- `salary_level`: Salary level (low/medium/high)

### Student Scores Dataset (`student_scores.csv`)

This dataset contains information about students and their final exam scores.

Key columns:
- `hours_studied`: Number of hours studied
- `previous_score`: Score in the previous exam
- `assignments_completed`: Number of assignments completed
- `final_score`: Final exam score - This is the target variable

## Task Instructions

Your task is to implement the functions in both files according to the TODO comments. Each function has detailed instructions on what it should do.

### For `employee.py`:

1. `load_and_prepare_data(path)`: Load and preprocess the employee attrition dataset
2. `hypothesis_demo()`: Demonstrate the logistic regression hypothesis function
3. `sigmoid_demo()`: Demonstrate the sigmoid activation function
4. `cost_function(y_true, y_pred_prob)`: Implement the log loss cost function
5. `train_and_evaluate(X_train, y_train, X_test, y_test, path)`: Train and evaluate a logistic regression model

### For `student.py`:

1. `load_and_preprocess(path)`: Load and preprocess the student scores dataset
2. `show_key_stats(df)`: Display key statistics about the dataset
3. `prepare_data(df, features, target)`: Prepare the data for model training
4. `train_and_save_model(X_train, y_train, model_path)`: Train and save a linear regression model
5. `evaluate_model(model, X_test, y_test)`: Evaluate the model performance

## Implementation Guidelines

1. Read the TODO comments carefully to understand what each function should do
2. Implement each function according to the specifications
3. Make sure your implementation passes all the test cases in `test/test_functional.py`
4. Do not modify the function signatures or return types

## Testing Your Implementation

After implementing the functions, you can run the tests to check if your implementation is correct:

```bash
python -m test.test_functional
```

All tests should pass after your implementation is complete.

## Additional Notes

- The `employee.py` file implements a logistic regression model for binary classification
- The `student.py` file implements a linear regression model for predicting continuous values
- Both files include a main section that demonstrates the full workflow
- Make sure to print the required messages as specified in the TODO comments
