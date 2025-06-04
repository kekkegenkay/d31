# File: /src/rule_based_detector.py
import pandas as pd
import numpy as np

# --- Constants for Anomaly Detection Rules & Severity/Scoring ---
SEVERITY_LEVEL_MAP = {"Red": 3, "Orange": 2, "Yellow": 1, "": 0}
REVERSE_SEVERITY_MAP = {v: k for k, v in SEVERITY_LEVEL_MAP.items()}

RULE_DEFINITIONS = {
    "R1.1": (f"Age > 120 years", "Red", 100),
    "R1.2": (f"Age <= 0 years", "Red", 100),
    "R_AGE_CORRUPT": (f"Age is extremely high (potential data corruption)", "Red", 110),
    "R2.1": ("Negative weight", "Red", 90),
    "R2.2": (f"Adult (Age >= 18) with Weight < 20 kg", "Red", 90),
    "R2.3": (f"Adult (Age >= 18) with Weight > 400 kg", "Red", 90),
    "R2.4": (f"Child (0 < Age < 18) with Weight < 2 kg", "Red", 90),
    "R3.1": (f"Height < 50 cm", "Red", 95),
    "R3.2": (f"Height > 250 cm", "Red", 95),
    "R4.1": (f"BMI < 10 (using provided imcINdex)", "Red", 100),
    "R4.2": (f"BMI > 70 (using provided imcINdex)", "Red", 100),
    "R5.1": (f"Age > 85 and IMC in ['Obese', 'Extremly Obese']", "Orange", 50),
    "R7.1": (f"Calculated BMI differs from imcINdex by > 1.0", "Orange", 40),
    "R6.1": (f"Potential duplicate (same age/weight/height within 1 hr)", "Orange", 30),
    "R_MISSING_CRITICAL": ("Missing critical data (age_v, greutate, or inaltime)", "Orange", 60),
}

MAX_VALID_AGE = 120
MIN_VALID_AGE = 0
ADULT_AGE_THRESHOLD = 18
MIN_ADULT_WEIGHT_KG = 20
MAX_ADULT_WEIGHT_KG = 400
MIN_CHILD_WEIGHT_KG = 2
MIN_WEIGHT_KG = 0
MIN_HEIGHT_CM = 50
MAX_HEIGHT_CM = 250
MIN_BMI_THRESHOLD = 10
MAX_BMI_THRESHOLD = 70
ELDERLY_AGE_THRESHOLD = 85
SUSPICIOUS_OBESITY_CATEGORIES = ['Obese', 'Extremly Obese']
DUPLICATE_TIMEFRAME_HOURS = 1
BMI_CALCULATION_TOLERANCE = 1.0

def _add_anomaly_reason(df_row, reason_code):
    if reason_code not in RULE_DEFINITIONS:
        desc, severity_level_str, score_value = "Unknown rule violation", "Yellow", 10
    else:
        desc, severity_level_str, score_value = RULE_DEFINITIONS[reason_code]
    full_reason_display = f"{reason_code}: {desc}"
    if not isinstance(df_row['anomaly_reason'], list): df_row['anomaly_reason'] = []
    if not isinstance(df_row['applied_rule_codes'], list): df_row['applied_rule_codes'] = []
    if reason_code not in df_row['applied_rule_codes']:
        df_row['anomaly_reason'].append(full_reason_display)
        df_row['applied_rule_codes'].append(reason_code)
        df_row['total_anomaly_score'] += score_value
    current_max_severity_val = df_row.get('max_anomaly_severity_value', 0)
    new_severity_val = SEVERITY_LEVEL_MAP.get(severity_level_str, 0)
    if new_severity_val > current_max_severity_val:
        df_row['max_anomaly_severity_value'] = new_severity_val
    df_row['is_anomaly'] = True
    return df_row

def apply_data_integrity_rules(df):
    critical_cols = ['age_v', 'greutate', 'inaltime'] # Using original column names
    for col in critical_cols:
        if col in df.columns:
            condition = df[col].isna()
            df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R_MISSING_CRITICAL"), axis=1)
    if 'age_v' in df.columns:
        condition = (df['age_v'].notna()) & (df['age_v'] > 2000000000)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R_AGE_CORRUPT"), axis=1)
    return df

def apply_age_rules(df):
    if 'age_v' in df.columns:
        condition = (df['age_v'].notna()) & (df['age_v'] > MAX_VALID_AGE) & (df['age_v'] <= 2000000000)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R1.1"), axis=1)
        condition = (df['age_v'].notna()) & (df['age_v'] <= MIN_VALID_AGE)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R1.2"), axis=1)
    return df

def apply_weight_rules(df):
    if 'greutate' in df.columns and 'age_v' in df.columns:
        condition = (df['greutate'].notna()) & (df['greutate'] < MIN_WEIGHT_KG)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R2.1"), axis=1)
        condition = (df['age_v'] >= ADULT_AGE_THRESHOLD) & (df['greutate'].notna()) & (df['greutate'] < MIN_ADULT_WEIGHT_KG)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R2.2"), axis=1)
        condition = (df['age_v'] >= ADULT_AGE_THRESHOLD) & (df['greutate'].notna()) & (df['greutate'] > MAX_ADULT_WEIGHT_KG)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R2.3"), axis=1)
        condition = (df['age_v'] < ADULT_AGE_THRESHOLD) & (df['age_v'] > MIN_VALID_AGE) & \
                    (df['greutate'].notna()) & (df['greutate'] < MIN_CHILD_WEIGHT_KG)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R2.4"), axis=1)
    return df

def apply_height_rules(df):
    if 'inaltime' in df.columns:
        condition = (df['inaltime'].notna()) & (df['inaltime'] < MIN_HEIGHT_CM)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R3.1"), axis=1)
        condition = (df['inaltime'].notna()) & (df['inaltime'] > MAX_HEIGHT_CM)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R3.2"), axis=1)
    return df

def apply_bmi_rules(df): # Works on imcINdex from original data
    if 'imcINdex' in df.columns:
        # Use a temporary cleaned column for BMI rules if not already present by this module
        if 'rule_based_imcINdex_cleaned' not in df.columns:
            df['rule_based_imcINdex_cleaned'] = pd.to_numeric(df['imcINdex'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        
        condition = (df['rule_based_imcINdex_cleaned'].notna()) & (df['rule_based_imcINdex_cleaned'] < MIN_BMI_THRESHOLD)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R4.1"), axis=1)
        condition = (df['rule_based_imcINdex_cleaned'].notna()) & (df['rule_based_imcINdex_cleaned'] > MAX_BMI_THRESHOLD)
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R4.2"), axis=1)
    return df

def apply_elderly_obesity_rule(df):
    if 'age_v' in df.columns and 'IMC' in df.columns:
        condition = (df['age_v'] > ELDERLY_AGE_THRESHOLD) & \
                    (df['IMC'].notna() & df['IMC'].isin(SUSPICIOUS_OBESITY_CATEGORIES))
        df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R5.1"), axis=1)
    return df

def apply_duplicate_case_rule(df):
    # Assumes 'age_v', 'greutate', 'inaltime', 'data1' exist
    if not all(col in df.columns for col in ['age_v', 'greutate', 'inaltime', 'data1']):
        return df # Cannot apply rule if essential columns are missing
    if df['data1'].isnull().all(): # Cannot proceed if all dates are NaT
        return df

    df_sorted = df.sort_values(by=['age_v', 'greutate', 'inaltime', 'data1']).copy()
    df_sorted['prev_data1'] = df_sorted.groupby(['age_v', 'greutate', 'inaltime'])['data1'].shift(1)
    
    mask_not_nat = df_sorted['prev_data1'].notna() & df_sorted['data1'].notna()
    df_sorted['time_diff_to_prev_hours'] = np.nan
    if mask_not_nat.any():
        df_sorted.loc[mask_not_nat, 'time_diff_to_prev_hours'] = \
            (df_sorted.loc[mask_not_nat, 'data1'] - df_sorted.loc[mask_not_nat, 'prev_data1']).dt.total_seconds() / 3600

    duplicate_indices = df_sorted[
        (df_sorted['time_diff_to_prev_hours'].notna()) &
        (df_sorted['time_diff_to_prev_hours'] >= 0) &
        (df_sorted['time_diff_to_prev_hours'] < DUPLICATE_TIMEFRAME_HOURS)
    ].index
    df.loc[df.index.isin(duplicate_indices)] = df.loc[df.index.isin(duplicate_indices)].apply(
        lambda row: _add_anomaly_reason(row, "R6.1"), axis=1
    )
    # Clean up temporary columns added by this function if they were added to the main df
    if 'prev_data1' in df.columns: df.drop(columns=['prev_data1'], inplace=True, errors='ignore')
    if 'time_diff_to_prev_hours' in df.columns: df.drop(columns=['time_diff_to_prev_hours'], inplace=True, errors='ignore')
    return df

def apply_bmi_consistency_rule(df):
    # Uses original column names: 'inaltime', 'greutate', 'imcINdex'
    if not all(col in df.columns for col in ['inaltime', 'greutate', 'imcINdex']):
        return df

    if 'rule_based_inaltime_m' not in df.columns:
        df['rule_based_inaltime_m'] = pd.to_numeric(df['inaltime'], errors='coerce') / 100
    if 'rule_based_calculated_bmi' not in df.columns:
        df['rule_based_calculated_bmi'] = np.where(
            (df['rule_based_inaltime_m'].notna()) & (df['rule_based_inaltime_m'] > 0) & (df['greutate'].notna()),
            pd.to_numeric(df['greutate'], errors='coerce') / (df['rule_based_inaltime_m'] ** 2),
            np.nan
        )
        df['rule_based_calculated_bmi'] = df['rule_based_calculated_bmi'].replace([np.inf, -np.inf], np.nan)
    if 'rule_based_imcINdex_cleaned' not in df.columns: # This was created in apply_bmi_rules
        df['rule_based_imcINdex_cleaned'] = pd.to_numeric(df['imcINdex'], errors='coerce').replace([np.inf, -np.inf], np.nan)

    condition = (
        df['rule_based_calculated_bmi'].notna() &
        df['rule_based_imcINdex_cleaned'].notna() &
        (df['rule_based_imcINdex_cleaned'] <= MAX_BMI_THRESHOLD) &
        (abs(df['rule_based_calculated_bmi'] - df['rule_based_imcINdex_cleaned']) > BMI_CALCULATION_TOLERANCE)
    )
    df.loc[condition] = df.loc[condition].apply(lambda row: _add_anomaly_reason(row, "R7.1"), axis=1)
    return df

def calculate_anomaly_score_percentage(row):
    if not row.get('is_anomaly', False): return 0
    max_severity_val = row.get('max_anomaly_severity_value', 0)
    total_score = row.get('total_anomaly_score', 0)
    percentage = 0
    if max_severity_val == SEVERITY_LEVEL_MAP["Red"]:
        percentage = 80 + min(total_score / 10, 20) # Scaled for typical red scores
    elif max_severity_val == SEVERITY_LEVEL_MAP["Orange"]:
        percentage = 50 + min(total_score / 5, 29)  # Scaled for typical orange scores
    elif max_severity_val == SEVERITY_LEVEL_MAP["Yellow"]:
        percentage = 20 + min(total_score / 2, 29)  # Scaled for typical yellow scores
    return min(int(percentage), 100)

def get_anomaly_detection_pipeline():
    return [
        ("Applying Data Integrity Checks", apply_data_integrity_rules),
        ("Applying Age Rules", apply_age_rules),
        ("Applying Weight Rules", apply_weight_rules),
        ("Applying Height Rules", apply_height_rules),
        ("Applying BMI Rules (from imcINdex)", apply_bmi_rules),
        ("Applying Elderly Obesity Rule", apply_elderly_obesity_rule),
        ("Applying Duplicate Case Rule", apply_duplicate_case_rule),
        ("Applying BMI Consistency Rule", apply_bmi_consistency_rule)
    ]

def initialize_anomaly_columns(df):
    df['is_anomaly'] = False
    df['anomaly_reason'] = [[] for _ in range(len(df))]
    df['applied_rule_codes'] = [[] for _ in range(len(df))]
    df['total_anomaly_score'] = 0
    df['max_anomaly_severity_value'] = 0
    return df

def finalize_anomaly_data(df):
    if 'is_anomaly' in df.columns:
        df['max_anomaly_severity_category'] = df['max_anomaly_severity_value'].map(REVERSE_SEVERITY_MAP).fillna("")
        df['anomaly_score_percentage'] = df.apply(calculate_anomaly_score_percentage, axis=1)
        # Clean up temporary columns used internally by rule-based detector
        cols_to_drop_internal = ['rule_based_imcINdex_cleaned', 'rule_based_inaltime_m', 'rule_based_calculated_bmi']
        for col in cols_to_drop_internal:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors='ignore')
    return df