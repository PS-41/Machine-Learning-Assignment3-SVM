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
    median_value = df[column].median()
    df['binary_label'] = np.where(df[column] > median_value, 1, 0)
    return df

def normalize_features(X: np.ndarray) -> np.ndarray:
    '''
    Normalize the feature data to have a mean of 0 and standard deviation of 1.

    Parameters:
        X (np.ndarray): Feature data to be normalized.

    Returns:
        np.ndarray: Normalized feature data.
    '''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def split_data(df: pd.DataFrame, features: list, label: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
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
    X = df[features].values
    y = df[label].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_svm_model(X_train: np.ndarray, y_train: np.ndarray, kernel: str = 'linear') -> SVC:
    '''
    Train an SVM model using the specified kernel.

    Parameters:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        kernel (str): The kernel type to be used in the algorithm (default is 'linear').
    '''
    if kernel == 'poly':
        model = SVC(kernel=kernel, degree=3)
    else:
        model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model

def find_best_config(df: pd.DataFrame, label: str, feature_combinations: list, kernels: list) -> tuple:
    '''
    Finds the best feature combination and kernel configuration based on the highest accuracy score.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        label (str): The column name for the label.
        feature_combinations (list): List of feature combinations to test.
        kernels (list): List of kernels to test.

    Returns:
        tuple: Best features and best kernel
    '''
    best_accuracy = 0
    best_features = None
    best_kernel = None

    for features in feature_combinations:
        X_train, X_test, y_train, y_test = split_data(df, features, label)
        X_train = normalize_features(X_train)
        X_test = normalize_features(X_test)
        for kernel in kernels:
            model = train_svm_model(X_train, y_train, kernel=kernel)
            accuracy = model.score(X_test, y_test)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = features
                best_kernel = kernel

    print(best_accuracy, best_features, best_kernel)

    return best_features, best_kernel

def get_all_combinations(features: list) -> list:
    """
    Generate all possible combinations of the provided features.

    Parameters:
        features (list): List of feature names.

    Returns:
        list: List of all possible feature combinations.
    """
    all_combos = []
    n = len(features)
    for r in range(1, n + 1):
        for i in range(1 << n):
            combo = []
            for j in range(n):
                if (i & (1 << j)) > 0:
                    combo.append(features[j])
            if len(combo) == r:
                all_combos.append(combo)
    return all_combos

if __name__ == "__main__":

    df = pd.read_csv('data_train-2.csv')
    df = create_binary_label(df, column='Chance of Admit ')

    features_list = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
    feature_combinations = get_all_combinations(features_list)

    kernels = ['linear', 'rbf', 'poly']

    # For each SVM kernel, train a model with possible feature combinations and store the best model in the best_model variable
    best_features, best_kernel = find_best_config(df, 'binary_label', feature_combinations, kernels)

    X_data = df[best_features].values
    y_data = df['binary_label'].values

    X_data = normalize_features(X_data)

    best_model = train_svm_model(X_data, y_data, kernel=best_kernel)

    # Save the best model to the pickle file
    with open('svm_best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    # Attach the pkl file with the submission