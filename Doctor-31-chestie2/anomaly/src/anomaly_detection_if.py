# src/anomaly_detection_if.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def apply_isolation_forest(df: pd.DataFrame, features: list, contamination: float = 0.05, random_state: int = 42):
    """
    Applies Isolation Forest to detect anomalies.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of feature column names.
        contamination (float): The expected proportion of outliers in the data set.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with 'if_anomaly' (-1 for anomaly, 1 for normal)
                      and 'if_anomaly_score' (the lower, the more abnormal).
    """
    df_copy = df.copy()

    X_candidate = df_copy[features].replace([float('inf'), float('-inf')], pd.NA)
    X_input = X_candidate.fillna(0)

    if X_input.empty or X_input.isnull().all().all():
        df_copy["if_anomaly_score"] = pd.NA
        df_copy["if_anomaly"] = 1 # Default to normal
        return df_copy

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_input)

    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100) # n_estimators can be tuned

    try:
        predictions = model.fit_predict(X_scaled) # -1 for outliers, 1 for inliers
        # decision_function: Higher score = more normal, lower score = more anomalous.
        scores = model.decision_function(X_scaled)
    except Exception as e:
        print(f"Error during Isolation Forest fitting or prediction: {e}")
        df_copy["if_anomaly_score"] = pd.NA
        df_copy["if_anomaly"] = 1 # Default to normal
        return df_copy

    df_copy["if_anomaly"] = predictions
    df_copy["if_anomaly_score"] = scores

    return df_copy