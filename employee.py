import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Function 1: Load, clean, and prepare dataset
def load_and_prepare_data(path="employee_attrition.csv"):
    """
    TODO: Implement this function to load and prepare the employee attrition dataset.
    
    This function should:
    1. Load the CSV file from the given path
    2. Print statistics about average monthly hours
    3. Encode categorical variables (department, salary_level) using LabelEncoder
    4. Scale numerical features using StandardScaler
    5. Print a success message
    6. Return the prepared DataFrame
    
    Parameters:
    path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Prepared DataFrame
    """
    pass

# Function 2: Hypothesis function demo (returns probability)
def hypothesis_demo():
    """
    TODO: Implement this function to demonstrate the hypothesis function.
    
    This function should:
    1. Create a sample feature vector x_sample
    2. Define weights and bias
    3. Calculate z = dot(weights, x_sample) + bias
    4. Calculate h(x) = sigmoid(z)
    5. Print information about the hypothesis function
    6. Print the calculated probability
    """
    pass

# Function 3: Sigmoid activation demo
def sigmoid_demo():
    """
    TODO: Implement this function to demonstrate the sigmoid activation function.
    
    This function should:
    1. Define z = 2.0
    2. Calculate sigmoid(z) = 1 / (1 + exp(-z))
    3. Print the result as "Sigmoid(2.0) = 0.8808"
    """
    pass

# Function 4: Custom log loss cost function
def cost_function(y_true, y_pred_prob):
    """
    TODO: Implement this function to calculate the log loss cost function.
    
    This function should:
    1. Add a small epsilon to prevent log(0)
    2. Clip prediction probabilities to avoid numerical issues
    3. Calculate the binary cross-entropy: -mean(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
    
    Parameters:
    y_true (array): True labels (0 or 1)
    y_pred_prob (array): Predicted probabilities
    
    Returns:
    float: Calculated cost
    """
    pass

# Function 5: Train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="attrition_model.pkl"):
    """
    TODO: Implement this function to train and evaluate a logistic regression model.
    
    This function should:
    1. Create a LogisticRegression model
    2. Train the model on X_train and y_train
    3. Save the model to the specified path using joblib
    4. Print a success message
    5. Make predictions on X_test
    6. Calculate the cost using the custom cost_function
    7. Print the cost and sample predictions
    
    Parameters:
    X_train (array): Training features
    y_train (array): Training labels
    X_test (array): Test features
    y_test (array): Test labels
    path (str): Path to save the model
    """
    pass

# --------- Main Logic ---------
if __name__ == "__main__":
    df = load_and_prepare_data("employee_attrition.csv")

    hypothesis_demo()
    sigmoid_demo()

    X = df.drop(columns=['left'])
    y = df['left']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate(X_train, y_train, X_test, y_test)
