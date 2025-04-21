import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Function 1: Load and preprocess the dataset
def load_and_preprocess(path):
    """
    TODO: Implement this function to load and preprocess the student scores dataset.
    
    This function should:
    1. Load the CSV file from the given path
    2. Convert column names to lowercase and strip whitespace
    3. Remove rows with missing values
    4. Print a success message "Student data loaded and cleaned."
    5. Return the cleaned DataFrame
    
    Parameters:
    path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    pass

# Function 2: Show basic stats
def show_key_stats(df):
    """
    TODO: Implement this function to display key statistics about the dataset.
    
    This function should:
    1. Calculate the standard deviation of hours_studied
    2. Find the maximum value of previous_score
    3. Print these statistics with appropriate labels
    
    Parameters:
    df (pd.DataFrame): The student scores DataFrame
    """
    pass

# Function 3: Prepare data
def prepare_data(df, features, target):
    """
    TODO: Implement this function to prepare the data for model training.
    
    This function should:
    1. Extract features (X) and target (y) from the DataFrame
    2. Create a StandardScaler and scale the features
    3. Split the data into training and testing sets (80/20 split)
    4. Print a success message "Data prepared and split."
    5. Return X_train, X_test, y_train, y_test, and the scaler
    
    Parameters:
    df (pd.DataFrame): The student scores DataFrame
    features (list): List of feature column names
    target (str): Target column name
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    pass

# Function 4: Train and save model
def train_and_save_model(X_train, y_train, model_path="student_score_model.pkl"):
    """
    TODO: Implement this function to train and save a linear regression model.
    
    This function should:
    1. Create a LinearRegression model
    2. Train the model on X_train and y_train
    3. Save the model to the specified path using joblib
    4. Print a success message
    5. Return the trained model
    
    Parameters:
    X_train (array): Training features
    y_train (array): Training target values
    model_path (str): Path to save the model
    
    Returns:
    LinearRegression: Trained model
    """
    pass

# Function 5: Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    TODO: Implement this function to evaluate the model performance.
    
    This function should:
    1. Make predictions on X_test
    2. Calculate the mean squared error
    3. Print the MSE and sample predictions
    
    Parameters:
    model (LinearRegression): Trained model
    X_test (array): Test features
    y_test (array): Test target values
    """
    pass

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    features = ['hours_studied', 'previous_score', 'assignments_completed']
    target = 'final_score'

    df = load_and_preprocess("student_scores.csv")
    show_key_stats(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, target)
    model = train_and_save_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
