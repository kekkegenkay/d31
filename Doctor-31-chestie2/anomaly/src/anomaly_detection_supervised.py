import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def generate_labels_from_rule_based_output(df_with_is_anomaly_and_common_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generates labels for supervised learning based on the 'is_anomaly' column
    produced by the rule-based detector.
    Assumes df_with_is_anomaly_and_common_features has common feature column names
    ('age', 'weight', 'height', 'bmi') and an 'is_anomaly' boolean column.
    """
    df = df_with_is_anomaly_and_common_features.copy()

    if "is_anomaly" not in df.columns:
        raise ValueError("Input DataFrame must contain an 'is_anomaly' column from rule-based detection.")

    # Convert boolean 'is_anomaly' to integer labels (1 for anomaly, 0 for normal)
    df["is_anomaly"] = df["is_anomaly"].fillna(False) # Ensure boolean before conversion
    df["label"] = df["is_anomaly"].astype(int)

    return df


def train_supervised_anomaly_model(df_labeled: pd.DataFrame):
    """
    Antrenează un model supervizat (XGBoost) pe datele etichetate.
    df_labeled must contain the 'label' column and features: 'age', 'weight', 'height', 'bmi'.
    """
    features = ["age", "weight", "height", "bmi"]

    # Preprocesare
    # Ensure no inf values, and drop rows where essential features or label are NaN
    X_candidate = df_labeled[features].replace([float('inf'), float('-inf')], pd.NA)
    y_candidate = df_labeled["label"]

    # Keep only rows where all features and the label are non-null
    valid_indices = X_candidate.dropna().index.intersection(y_candidate.dropna().index)

    if len(valid_indices) == 0:
        raise ValueError("No valid data remaining after dropping NaNs from features and labels. Cannot train model.")

    X = X_candidate.loc[valid_indices]
    y = y_candidate.loc[valid_indices].astype(int)

    if len(X) < 20: # Threshold for minimum samples
        raise ValueError(f"Not enough valid samples ({len(X)}) to train the model. Need at least 20.")
    if y.nunique() < 2:
        raise ValueError(f"Only one class present in the labels after preprocessing. Label distribution:\n{y.value_counts()}. Cannot train.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)

    return model, scaler


def apply_supervised_model(df: pd.DataFrame, model, scaler):
    """
    Aplică modelul supervizat pe toate datele și returnează scorurile.
    df should have common feature names.
    """
    df_copy = df.copy()
    features = ["age", "weight", "height", "bmi"]

    X_pred_candidate = df_copy[features].replace([float('inf'), float('-inf')], pd.NA)
    X_pred = X_pred_candidate.fillna(0) # Fill NaNs for prediction

    X_scaled = scaler.transform(X_pred)

    scores = model.predict_proba(X_scaled)[:, 1]
    df_copy["ai_anomaly_score"] = pd.Series(scores, index=X_pred.index)

    return df_copy


if __name__ == "__main__":
    try:
        from data_loader import load_and_prepare_data
        from rule_based_detector import get_anomaly_detection_pipeline, initialize_anomaly_columns, finalize_anomaly_data
    except ImportError:
        # Fallback if running from project root (e.g. python anomaly/src/...)
        from src.data_loader import load_and_prepare_data
        from src.rule_based_detector import get_anomaly_detection_pipeline, initialize_anomaly_columns, finalize_anomaly_data
    import os

    # Determine project root dynamically to find the data file
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)
    project_root_dir = os.path.dirname(src_dir) # This should be 'anomaly' directory

    # Path to the CSV file from the project root
    csv_file_path_main = os.path.join(project_root_dir, "data", "doctor31_cazuri(1).csv")

    print(f"Attempting to load data from: {csv_file_path_main}")
    df_orig, df_comm = load_and_prepare_data(csv_file_path_main)

    if df_orig is not None and df_comm is not None:
        print("Successfully loaded original and common DataFrames.")
        print(f"df_orig shape: {df_orig.shape}, df_comm shape: {df_comm.shape}")

        print("\nRunning rule-based detection to generate 'is_anomaly' flags...")
        df_rb_processed = df_orig.copy()
        df_rb_processed = initialize_anomaly_columns(df_rb_processed)
        rules_pipeline = get_anomaly_detection_pipeline()
        for i, (message, rule_func) in enumerate(rules_pipeline):
            print(f"  Applying rule: {message}")
            df_rb_processed = rule_func(df_rb_processed)
        df_rb_final = finalize_anomaly_data(df_rb_processed)
        print(f"Rule-based detection complete. 'is_anomaly' counts:\n{df_rb_final['is_anomaly'].value_counts(dropna=False)}")

        # Merge 'is_anomaly' from rule-based output (df_rb_final) into df_comm
        # Ensure ID columns exist
        if 'id_cases' not in df_rb_final.columns:
            print("ERROR: 'id_cases' not found in rule-based processed data (df_rb_final). Cannot merge for AI training.")
        elif 'id_case' not in df_comm.columns:
            print("ERROR: 'id_case' not found in common features data (df_comm). Cannot merge for AI training.")
        else:
            df_for_labeling = df_comm.merge(
                df_rb_final[['id_cases', 'is_anomaly']], # Select only necessary columns
                left_on='id_case',      # From df_comm
                right_on='id_cases',    # From df_rb_final
                how='left'              # Keep all records from df_comm
            )
            # Handle cases where merge might not find a match (should be rare if IDs are consistent)or if 'is_anomaly' was somehow NaN from rules (should be False/True)
            df_for_labeling['is_anomaly'] = df_for_labeling['is_anomaly'].fillna(False)
            print(f"\nMerged 'is_anomaly' into common features DataFrame. Shape: {df_for_labeling.shape}")
            print(f"'is_anomaly' counts after merge:\n{df_for_labeling['is_anomaly'].value_counts(dropna=False)}")

            # Ensure essential features for AI are present and not all NaN before dropping
            ai_features = ['age', 'weight', 'height', 'bmi']
            missing_ai_features = [f for f in ai_features if f not in df_for_labeling.columns]
            if missing_ai_features:
                print(f"ERROR: Essential AI features missing from df_for_labeling: {missing_ai_features}")
            else:
                df_for_labeling_cleaned = df_for_labeling.dropna(subset=ai_features)
                print(f"DataFrame shape after dropping rows with NaN in AI features: {df_for_labeling_cleaned.shape}")

                if df_for_labeling_cleaned.empty:
                    print("No data available for AI training after cleaning NaNs from features.")
                else:
                    print("\nGenerating labels for AI model from 'is_anomaly' flags...")
                    df_labeled = generate_labels_from_rule_based_output(df_for_labeling_cleaned)
                    print(f"📊 Cazuri etichetate (din reguli): {len(df_labeled)}")
                    print(f"Distribuție etichete ('label' column):\n{df_labeled['label'].value_counts(dropna=False)}")

                    try:
                        print("\nAttempting to train the supervised AI model...")
                        model, scaler = train_supervised_anomaly_model(df_labeled)
                        print("✅ Supervised AI Model trained successfully.")

                        print("\nApplying trained AI model to the full common dataset...")
                        # Apply to a fresh copy of df_comm to ensure all original rows are processed
                        df_comm_with_scores = apply_supervised_model(df_comm.copy(), model, scaler)

                        print("\n🔴 Top 5 cazuri cu scor de anomalie AI (bazat pe reguli) mare:")
                        print(df_comm_with_scores.sort_values("ai_anomaly_score", ascending=False)[
                                  ["id_case", "age", "weight", "height", "bmi", "ai_anomaly_score"]].head())

                    except ValueError as e:
                        print(f"Nu s-a putut antrena modelul AI: {e}")
                    except Exception as e_gen:
                        print(f"O eroare generală a apărut în timpul antrenării sau aplicării modelului AI: {e_gen}")
    else:
        print("Eroare la încărcarea datelor inițiale. Nu se poate continua.")