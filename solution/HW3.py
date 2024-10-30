import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
   the library specified in question instruction available. Importing additional libraries will result in 
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional 
   default arguments or helper functions if necessary, but ensure the input/output format matches.

3. Save your best model as 'svm_best_model.pkl' as shown in the main function.
'''

def create_binary_label(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''
    Convert the values in the specified column to binary labels:
    - Values greater than the median will be labeled as 1.
    - Values less than or equal to the median will be labeled as 0.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name based on which the binary label is created.

    Returns:
        pd.DataFrame: DataFrame with an additional binary label column.
    '''

def split_data(df: pd.DataFrame, features: list, label: str) -> tuple:
    '''
    Split the data into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        features (list): List of column names to use as features.
        label (str): The column name for the label.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Seed for reproducibility (default is 42).

    Returns:
        tuple: X_train, X_test, y_train, y_test (numpy arrays)
    '''

def train_svm_model(X_train: np.ndarray, y_train: np.ndarray, kernel: str = 'linear') -> SVC:
    '''
    Train an SVM model using the specified kernel.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        kernel (str): The kernel type to be used in the algorithm (default is 'linear').
    '''

if __name__ == "__main__":

    # For each SVM kernel, train a model with possible feature combinations and store the best model in the best_model variable
    best_model = None

    # Save the best model to the pickle file
    with open('svm_best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    # Attach the pkl file with the submission