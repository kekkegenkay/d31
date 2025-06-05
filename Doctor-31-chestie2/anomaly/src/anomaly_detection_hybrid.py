# src/anomaly_detection_hybrid.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Assuming apply_isolation_forest and apply_one_class_svm are importable
# from .anomaly_detection_if import apply_isolation_forest # Use relative import if in same package
# from .anomaly_detection_svm import apply_one_class_svm

# For direct use in Streamlit or if they are in sys.path:
from src.anomaly_detection_if import apply_isolation_forest
from src.anomaly_detection_svm import apply_one_class_svm


def apply_hybrid_if_svm(df: pd.DataFrame, features: list,
                        if_contamination: float = 0.05, svm_nu: float = 0.05,
                        weight_if: float = 0.5, weight_svm: float = 0.5):
    """
    Applies a hybrid model combining Isolation Forest and One-Class SVM.
    Normalizes their anomaly scores and combines them with specified weights.
    Higher combined score indicates higher anomaly likelihood.
    """
    df_copy = df.copy()

    # 1. Apply Isolation Forest
    df_if_results = apply_isolation_forest(df_copy, features, contamination=if_contamination)

    # 2. Apply One-Class SVM (pass the df that now has IF results, or df_copy if preferred)
    df_svm_results = apply_one_class_svm(df_if_results, features, nu=svm_nu)

    if 'if_anomaly_score' not in df_svm_results.columns or 'svm_anomaly_score' not in df_svm_results.columns:
        print("Error: Could not retrieve scores from IF or SVM for hybrid model.")
        df_svm_results['hybrid_anomaly_score'] = pd.NA
        df_svm_results['hybrid_anomaly'] = 1 # Default normal
        return df_svm_results

    # Impute NaNs in scores if any (e.g., with median of respective scores)
    # This is important before scaling.
    median_if_score = df_svm_results['if_anomaly_score'].median()
    median_svm_score = df_svm_results['svm_anomaly_score'].median()

    if_scores_series = df_svm_results['if_anomaly_score'].fillna(median_if_score if pd.notna(median_if_score) else 0)
    svm_scores_series = df_svm_results['svm_anomaly_score'].fillna(median_svm_score if pd.notna(median_svm_score) else 0)

    # 3. Normalize scores (MinMaxScaler to 0-1 range)
    # We want HIGHER scores to indicate MORE ANOMALOUS for the combined score.
    # IF: decision_function gives lower scores for anomalies. So, use -score.
    # OCSVM: decision_function gives lower (more negative) scores for anomalies. So, use -score.

    scaler = MinMaxScaler()

    # Ensure there's variance for the scaler to work, otherwise, it might produce NaNs or all 0s/1s.
    # Reshape for scaler: .values.reshape(-1, 1)
    if if_scores_series.nunique() > 1:
        norm_if_scores = scaler.fit_transform( (-if_scores_series).values.reshape(-1, 1) ).flatten()
    else: # Handle no variance case (all scores are the same)
        norm_if_scores = np.zeros(len(if_scores_series)) if if_scores_series.iloc[0] >= 0 else np.ones(len(if_scores_series))


    if svm_scores_series.nunique() > 1:
        norm_svm_scores = scaler.fit_transform( (-svm_scores_series).values.reshape(-1, 1) ).flatten()
    else: # Handle no variance case
        norm_svm_scores = np.zeros(len(svm_scores_series)) if svm_scores_series.iloc[0] >= 0 else np.ones(len(svm_scores_series))


    df_svm_results['norm_if_score_for_hybrid'] = norm_if_scores
    df_svm_results['norm_svm_score_for_hybrid'] = norm_svm_scores

    # 4. Combine scores (higher combined score = more anomalous)
    df_svm_results['hybrid_anomaly_score'] = (weight_if * df_svm_results['norm_if_score_for_hybrid'] +
                                              weight_svm * df_svm_results['norm_svm_score_for_hybrid'])

    # 5. Determine anomaly label based on combined score
    # This threshold determines what combined score is considered an anomaly.
    # A simple approach is to use a percentile based on the weighted average of input 'contamination'/'nu'.
    # E.g., if average expected anomaly is 5%, flag top 5% of hybrid scores.
    # This is an estimation. More robust thresholding might involve cross-validation or domain knowledge.

    # Calculate the percentile for flagging anomalies. If contamination=0.05, flag top 5%.
    # So, we need the (100 - combined_contamination_percentage)-th percentile of scores.
    # The scores are now 0-1, higher is more anomalous.
    effective_contamination = (if_contamination * weight_if + svm_nu * weight_svm)
    threshold_percentile = 100 * (1 - effective_contamination)

    if df_svm_results['hybrid_anomaly_score'].notna().any() and df_svm_results['hybrid_anomaly_score'].nunique() > 0 :
        # Ensure threshold_percentile is within reasonable bounds
        threshold_percentile = max(0, min(100, threshold_percentile))
        anomaly_threshold_value = np.percentile(df_svm_results['hybrid_anomaly_score'].dropna(), threshold_percentile)

        # If all scores are 0 (e.g. after scaling uniform data), this threshold might be 0.
        # Handle edge case where threshold_percentile is 100 (meaning 0% anomalies expected)
        if threshold_percentile >= 100:
             df_svm_results['hybrid_anomaly'] = 1 # All normal
        elif threshold_percentile <=0: # all anomalies
             df_svm_results['hybrid_anomaly'] = -1
        else:
            # Higher score is more anomalous, so > threshold_value is anomaly (-1)
            df_svm_results['hybrid_anomaly'] = np.where(df_svm_results['hybrid_anomaly_score'] > anomaly_threshold_value, -1, 1)
    else:
        df_svm_results['hybrid_anomaly'] = 1 # Default to normal if scores are all NaN or uniform

    return df_svm_results