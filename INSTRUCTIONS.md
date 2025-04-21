# Detailed Implementation Instructions

This document provides detailed instructions for implementing the functions in `employee.py` and `student.py`. Follow these guidelines to complete the assessment successfully.

## Employee Attrition Prediction (`employee.py`)

### Function 1: `load_and_prepare_data(path="employee_attrition.csv")`

**Purpose**: Load and preprocess the employee attrition dataset.

**Implementation Steps**:
1. Use `pd.read_csv(path)` to load the dataset
2. Print statistics about average monthly hours using:
   ```python
   print("\n📊 Avg Monthly Hours - Mean: {:.2f}, Max: {:.2f}".format(
       df['average_monthly_hours'].mean(), df['average_monthly_hours'].max()))
   ```
3. For categorical columns ('department', 'salary_level'), check if they are object type and encode them using `LabelEncoder()`
4. Scale all features (except 'left') using `StandardScaler()`
5. Print a success message: `print("✅ Dataset loaded and preprocessed.")`
6. Return the prepared DataFrame

### Function 2: `hypothesis_demo()`

**Purpose**: Demonstrate the logistic regression hypothesis function.

**Implementation Steps**:
1. Create a sample feature vector, e.g., `x_sample = np.array([0.5, -1.2, 0.8])`
2. Define weights, e.g., `weights = np.array([1.5, -0.8, 2.0])`
3. Define bias, e.g., `bias = 0.3`
4. Calculate z = dot product of weights and x_sample plus bias
5. Calculate h(x) = sigmoid(z) = 1 / (1 + np.exp(-z))
6. Print information about the hypothesis:
   ```python
   print(f"\n📐 Hypothesis h(x) = sigmoid(w·x + b)")
   print(f"🧮 z = {z:.4f}")
   print(f"🔢 Probability that employee will leave = {h_x:.4f}")
   ```

### Function 3: `sigmoid_demo()`

**Purpose**: Demonstrate the sigmoid activation function.

**Implementation Steps**:
1. Define z = 2.0
2. Calculate sigmoid(z) = 1 / (1 + np.exp(-z))
3. Print the result: `print(f"\n🧠 Sigmoid(2.0) = {sigmoid:.4f}")`
   - Note: The result should be 0.8808

### Function 4: `cost_function(y_true, y_pred_prob)`

**Purpose**: Implement the log loss cost function.

**Implementation Steps**:
1. Add a small epsilon (e.g., 1e-15) to prevent log(0)
2. Clip prediction probabilities using `np.clip(y_pred_prob, epsilon, 1 - epsilon)`
3. Calculate binary cross-entropy:
   ```python
   -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
   ```
4. Return the calculated cost

### Function 5: `train_and_evaluate(X_train, y_train, X_test, y_test, path="attrition_model.pkl")`

**Purpose**: Train and evaluate a logistic regression model.

**Implementation Steps**:
1. Create a LogisticRegression model with max_iter=1000
2. Train the model on X_train and y_train
3. Save the model to the specified path using `joblib.dump(model, path)`
4. Print a success message: `print(f"\n✅ Model trained and saved to '{path}'")`
5. Make predictions on X_test (both class predictions and probabilities)
6. Calculate the cost using your custom cost_function
7. Print the cost and sample predictions:
   ```python
   print(f"\n🎯 Log Loss (Custom Cost): {cost:.4f}")
   print("📌 Sample Predictions:", y_pred[:10])
   ```

## Student Score Prediction (`student.py`)

### Function 1: `load_and_preprocess(path)`

**Purpose**: Load and preprocess the student scores dataset.

**Implementation Steps**:
1. Use `pd.read_csv(path)` to load the dataset
2. Convert column names to lowercase and strip whitespace: `df.columns = df.columns.str.lower().str.strip()`
3. Remove rows with missing values using `df.dropna()`
4. Print a success message: `print("📚 Student data loaded and cleaned.")`
5. Return the cleaned DataFrame

### Function 2: `show_key_stats(df)`

**Purpose**: Display key statistics about the dataset.

**Implementation Steps**:
1. Calculate the standard deviation of hours_studied: `hours_std = df['hours_studied'].std()`
2. Find the maximum value of previous_score: `max_previous_score = df['previous_score'].max()`
3. Print these statistics:
   ```python
   print(f"\n📊 Standard Deviation of Study Hours: {hours_std:.2f}")
   print(f"🏅 Max Previous Score: {max_previous_score}")
   ```

### Function 3: `prepare_data(df, features, target)`

**Purpose**: Prepare the data for model training.

**Implementation Steps**:
1. Extract features (X) and target (y) from the DataFrame
2. Create a StandardScaler and scale the features
3. Split the data into training and testing sets (80/20 split) with random_state=42
4. Print a success message: `print("\n🧪 Data prepared and split.")`
5. Return X_train, X_test, y_train, y_test, and the scaler

### Function 4: `train_and_save_model(X_train, y_train, model_path="student_score_model.pkl")`

**Purpose**: Train and save a linear regression model.

**Implementation Steps**:
1. Create a LinearRegression model
2. Train the model on X_train and y_train
3. Save the model to the specified path using `joblib.dump(model, model_path)`
4. Print a success message: `print(f"\n✅ Model trained and saved to '{model_path}'")`
5. Return the trained model

### Function 5: `evaluate_model(model, X_test, y_test)`

**Purpose**: Evaluate the model performance.

**Implementation Steps**:
1. Make predictions on X_test
2. Calculate the mean squared error using `mean_squared_error(y_test, y_pred)`
3. Print the MSE and sample predictions:
   ```python
   print(f"\n🎯 Mean Squared Error: {mse:.2f}")
   print("📈 Sample Predictions:", y_pred[:5])
   ```

## Testing Your Implementation

After implementing all functions, run the tests to check if your implementation is correct:

```bash
python -m test.test_functional
```

If all tests pass, your implementation is correct. If any tests fail, review the error messages and fix your implementation accordingly.

## Common Pitfalls to Avoid

1. **Not printing the exact expected messages**: Make sure your print statements match exactly what's expected in the tests.
2. **Incorrect function signatures**: Don't change the function parameters or return types.
3. **Not handling edge cases**: Make sure your functions handle potential errors gracefully.
4. **Incorrect scaling or encoding**: Follow the instructions carefully for preprocessing steps.
5. **Not saving models correctly**: Make sure you're using joblib.dump correctly to save models.

Good luck with your implementation!
