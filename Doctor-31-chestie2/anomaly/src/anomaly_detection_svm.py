# src/anomaly_detection_svm.py
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def apply_one_class_svm(df: pd.DataFrame, features: list, nu: float = 0.05, kernel: str = 'rbf', gamma: str = 'auto'):
    """
    Applies One-Class SVM to detect anomalies.
    'nu' is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    It roughly corresponds to the expected proportion of outliers.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of feature column names to use.
        nu (float): The nu parameter for OneClassSVM.
        kernel (str): Kernel type for SVM.
        gamma (str): Kernel coefficient.

    Returns:
        pd.DataFrame: DataFrame with 'svm_anomaly' (-1 for anomaly, 1 for normal)
                      and 'svm_anomaly_score' (signed distance to the separating hyperplane).
    """
    df_copy = df.copy()

    # Prepare data
    X_candidate = df_copy[features].replace([float('inf'), float('-inf')], pd.NA)
    # Fill NaNs - consider a more sophisticated strategy if needed (e.g., imputation)
    X_input = X_candidate.fillna(0)

    if X_input.empty or X_input.isnull().all().all():
        df_copy["svm_anomaly_score"] = pd.NA
        df_copy["svm_anomaly"] = 1 # Default to normal if no data
        return df_copy

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_input)

    # Initialize and fit One-Class SVM
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    try:
        predictions = model.fit_predict(X_scaled) # -1 for outliers, 1 for inliers
        scores = model.decision_function(X_scaled) # Signed distance. Negative for outliers.
    except Exception as e:
        print(f"Error during OneClassSVM fitting or prediction: {e}") # Or use st.error in Streamlit context
        df_copy["svm_anomaly_score"] = pd.NA
        df_copy["svm_anomaly"] = 1 # Default to normal
        return df_copy

    df_copy["svm_anomaly"] = predictions
    df_copy["svm_anomaly_score"] = scores

    return df_copy