import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import time 
import shap 

# --- Scikit-learn imports ---
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest # For unsupervised, already added in previous step

# --- Attempt to import AgGrid ---
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False


# --- Project-Specific Imports (Revised for src layout) ---
try:
    from src.rule_based_detector import (
        get_anomaly_detection_pipeline,
        initialize_anomaly_columns,
        finalize_anomaly_data,
        RULE_DEFINITIONS, 
        SEVERITY_LEVEL_MAP 
    )
except ImportError as e:
    st.error(f"Error importing from src.rule_based_detector: {e}. Ensure 'src/rule_based_detector.py' exists and src/ contains an __init__.py file.")
    st.stop()

try:
    from src.data_loader import load_and_prepare_data, COLUMN_MAPPING_TO_COMMON
    from src.validators import validate_data as validate_data_for_ai # This is the simple validator for AI prediction inputs
    from src.anomaly_detection_supervised import (
        generate_labels_from_rule_based_output, # UPDATED IMPORT
        train_supervised_anomaly_model,
        apply_supervised_model
    )
    from src.explain import explain_instance
except ImportError as e:
    st.error(f"Error importing new system components: {e}. Ensure all necessary files are in src/ and src/ contains an __init__.py file.")
    st.stop()

# --- Global Configuration ---
st.set_page_config(layout="wide")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_ORIGINAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'doctor31_cazuri(1).csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SUPERVISED_MODEL_PATH = os.path.join(MODEL_DIR, "model_supervised_doctor31.pkl")
SUPERVISED_SCALER_PATH = os.path.join(MODEL_DIR, "scaler_supervised_doctor31.pkl")


@st.cache_data
def load_initial_data(file_path):
    df_orig, df_comm = load_and_prepare_data(file_path)
    if df_orig is None:
        st.error(f"Failed to load data from {file_path}")
    return df_orig, df_comm

# Helper function to run rule-based detection (to avoid duplicating this logic within cache function)
def run_full_rule_based_detection_for_training(df_orig_data_for_rules):
    df_processed = df_orig_data_for_rules.copy()
    df_processed = initialize_anomaly_columns(df_processed) # From rule_based_detector
    rules_pipeline = get_anomaly_detection_pipeline() # From rule_based_detector
    for rule_name, rule_func in rules_pipeline: # Iterate through rules
        # st.write(f"Applying rule for training: {rule_name}") # Optional: for debugging
        df_processed = rule_func(df_processed)
    df_final_rules = finalize_anomaly_data(df_processed) # From rule_based_detector
    return df_final_rules


@st.cache_resource
# Signature updated to accept original and common DFs for the new training process
def get_or_train_ai_model_cached(_df_original_cols_for_rules, _df_common_cols_for_features):
    if os.path.exists(SUPERVISED_MODEL_PATH) and os.path.exists(SUPERVISED_SCALER_PATH):
        st.info("Loading pre-trained supervised AI model and scaler...")
        model = joblib.load(SUPERVISED_MODEL_PATH)
        scaler = joblib.load(SUPERVISED_SCALER_PATH)
        st.success("Pre-trained AI model and scaler loaded.")
        return model, scaler
    else:
        st.info("No pre-trained AI model found. Preparing data for new two-step AI model training...")
        
        # Step 1: Run Rule-Based Detector on original_cols to get 'is_anomaly' flags
        with st.spinner("Running rule-based detection for AI training labels..."):
            # Use a copy to avoid modifying the original DataFrame passed in
            df_rule_processed = run_full_rule_based_detection_for_training(_df_original_cols_for_rules.copy())
            # df_rule_processed now has 'is_anomaly' and uses original column names (e.g., 'id_cases', 'age_v')

        # Step 2: Prepare data for label generation
        # DataFrame with common feature names ('age', 'weight', etc.) AND the 'is_anomaly' flag.
        # Merge 'is_anomaly' from df_rule_processed.
        
        if 'id_case' not in _df_common_cols_for_features.columns or 'id_cases' not in df_rule_processed.columns:
            st.error("Critical ID columns ('id_case' in common data or 'id_cases' in rule-processed data) missing for merging. Cannot train AI model.")
            return None, None

        # Select only necessary columns from rule_processed to avoid duplicate feature columns after merge and to ensure we are picking up the correct 'is_anomaly'.
        df_rules_subset = df_rule_processed[['id_cases', 'is_anomaly']].copy()

        # Merge 'is_anomaly' into the common features DataFrame. Use a copy of _df_common_cols_for_features.
        df_for_labeling_merged = _df_common_cols_for_features.copy().merge(
            df_rules_subset,
            left_on='id_case', # From common_cols
            right_on='id_cases', # From rule_processed
            how='left' # Keep all rows from common_cols, match 'is_anomaly' where possible
        )
        
        # Handle cases where a common_col entry might not have a match in rule_processed.
        # For training, 'is_anomaly' should not be NaN. If no match, assume it was not flagged as anomaly by rules.
        df_for_labeling_merged['is_anomaly'] = df_for_labeling_merged['is_anomaly'].fillna(False)
        
        # Ensure crucial AI features and 'is_anomaly' are present
        required_cols_for_ai_features = ['age', 'weight', 'height', 'bmi']
        required_cols_for_labeling = required_cols_for_ai_features + ['is_anomaly']
        
        if not all(col in df_for_labeling_merged.columns for col in required_cols_for_labeling):
            missing_cols = [col for col in required_cols_for_labeling if col not in df_for_labeling_merged.columns]
            st.error(f"Missing columns after merge, required for AI training: {missing_cols}. Cannot train AI.")
            return None, None
            
        # Drop rows where essential features for AI model (age, weight, height, bmi) are NaN.
        # The 'is_anomaly' column should be boolean by now (True/False).
        df_for_labeling_cleaned = df_for_labeling_merged.dropna(subset=required_cols_for_ai_features)
        
        if df_for_labeling_cleaned.empty:
            st.error("No data left for AI training after dropping NaNs in essential features (age, weight, height, bmi).")
            return None, None

        with st.spinner("Generating labels for AI model training from rule-based output..."):
            # generate_labels_from_rule_based_output expects 'is_anomaly' column and common feature names
            df_labeled = generate_labels_from_rule_based_output(df_for_labeling_cleaned)
        
        if df_labeled.empty or "label" not in df_labeled.columns:
            st.warning("Label generation failed or produced an empty DataFrame.")
            return None, None
        
        # Further checks are now inside train_supervised_anomaly_model (e.g., enough samples, >1 class)
            
        with st.spinner("Training supervised AI (XGBoost) based on rule-derived labels..."):
            # train_supervised_anomaly_model expects 'label' and common feature names
            try:
                model, scaler = train_supervised_anomaly_model(df_labeled) # This will raise ValueError if issues
                joblib.dump(model, SUPERVISED_MODEL_PATH)
                joblib.dump(scaler, SUPERVISED_SCALER_PATH)
                st.success("New AI model trained (on rule-based labels) and saved.")
                return model, scaler
            except ValueError as ve: # Catch specific errors from training function
                st.error(f"AI Model training aborted: {ve}")
                return None, None
            except Exception as e_train: # Catch other unexpected errors
                st.error(f"An unexpected error occurred during AI model training: {e_train}")
                st.exception(e_train)
                return None, None

def highlight_severity_rule_based(row):
    severity_category = row.get('max_anomaly_severity_category', '')
    text_color = 'black'
    bg_color = 'white'
    if severity_category == 'Red': bg_color = '#FFCCCB'
    elif severity_category == 'Orange': bg_color = '#FFD580'
    elif severity_category == 'Yellow': bg_color = '#FFFFE0'
    return [f'background-color: {bg_color}; color: {text_color}' for _ in row]

def get_ai_risk_color_text(score):
    if score > 0.9: return "Red"
    elif score > 0.5: return "Orange"
    elif score > 0.2: return "Yellow"
    else: return "Green"


# --- Main App ---
st.title("Doctor31 Data Anomaly Detection Platform")

# Initialize session state for messages if not already present
if 'retrain_message' not in st.session_state:
    st.session_state.retrain_message = ""
if 'retrain_message_type' not in st.session_state: # 'success', 'error', 'info'
    st.session_state.retrain_message_type = "info"


df_original_cols, df_common_cols = load_initial_data(DATA_FILE_ORIGINAL_CSV)

if df_original_cols is not None and df_common_cols is not None:
    st.sidebar.header("⚙️ Detection Mode")
    # When detection mode changes, clear any lingering retrain messages
    current_detection_mode = st.session_state.get('detection_mode_tracker', None)
    new_detection_mode = st.sidebar.radio(
        "Choose Anomaly Detection Approach:",
        ("Rule-Based (Severity Scoring)", "AI Supervised (XGBoost)"),
        key="detection_mode_radio" # Add a key for consistent access
    )
    if current_detection_mode != new_detection_mode:
        st.session_state.retrain_message = "" # Clear message on mode change
        st.session_state.detection_mode_tracker = new_detection_mode


    st.sidebar.markdown("---")

    # --- Add Retrain Button ---
    st.sidebar.header("🛠️ AI Model Management")

    # Display persistent messages from session state
    if st.session_state.retrain_message:
        if st.session_state.retrain_message_type == "success":
            st.sidebar.success(st.session_state.retrain_message)
        elif st.session_state.retrain_message_type == "error":
            st.sidebar.error(st.session_state.retrain_message)
        else: # info
            st.sidebar.info(st.session_state.retrain_message)
        # Optionally, provide a way to clear the message or clear it after some actions
        # For now, it will persist until the next retrain attempt or mode change.

    if st.sidebar.button("🔄 Retrain AI Model", key="retrain_ai_button"):
        st.session_state.retrain_message = "Processing: Retraining AI model..." # Initial message
        st.session_state.retrain_message_type = "info"
        
        # Delete existing model and scaler files to force retraining
        files_removed_log = []
        files_ok = True
        if os.path.exists(SUPERVISED_MODEL_PATH):
            try:
                os.remove(SUPERVISED_MODEL_PATH)
                files_removed_log.append(f"Removed old model: {os.path.basename(SUPERVISED_MODEL_PATH)}")
            except OSError as e:
                st.session_state.retrain_message += f"\nError removing model file: {e}"
                files_ok = False
        if os.path.exists(SUPERVISED_SCALER_PATH):
            try:
                os.remove(SUPERVISED_SCALER_PATH)
                files_removed_log.append(f"Removed old scaler: {os.path.basename(SUPERVISED_SCALER_PATH)}")
            except OSError as e:
                st.session_state.retrain_message += f"\nError removing scaler file: {e}"
                files_ok = False
        
        if not files_ok:
            st.session_state.retrain_message_type = "error"
            st.rerun() # Rerun to show the error message from session state

        if files_removed_log: # Add successfully removed files to the message
            st.session_state.retrain_message += "\n" + "\n".join(files_removed_log)

        # Clear the cache for the model loading function
        get_or_train_ai_model_cached.clear()
        st.session_state.retrain_message += "\nCleared model cache."

        # Attempt to retrain by calling the function.
        training_success = False
        try:
            with st.spinner("Retraining AI model now... please wait."):
                new_model, new_scaler = get_or_train_ai_model_cached(df_original_cols, df_common_cols)
            if new_model and new_scaler:
                st.session_state.retrain_message = (
                    "✅ AI Model Retrained and Reloaded Successfully!\n"
                    "The AI Supervised (XGBoost) model has been updated. "
                    "Please switch to that mode or re-select it to use the new model."
                )
                st.session_state.retrain_message_type = "success"
                training_success = True
            else:
                st.session_state.retrain_message = "⚠️ AI Model Retraining Failed. Check logs or app messages in the main panel if any."
                st.session_state.retrain_message_type = "error"
        except Exception as e:
            st.session_state.retrain_message = f"An unexpected error occurred during retraining trigger: {e}"
            st.session_state.retrain_message_type = "error"
            # Display the exception in the main panel for detailed debugging
            st.exception(e)
        
        st.rerun() # Rerun to display the final message from session state
    
    st.sidebar.markdown("---")
    # --- End of Retrain Button ---


    st.subheader("Original Data Sample (First 5 Rows)")
    st.dataframe(df_original_cols.head())
    st.markdown("---")

    # Update the tracker for detection mode at the end of the sidebar setup
    # This ensures 'detection_mode_tracker' in session_state always reflects the latest radio button choice
    st.session_state.detection_mode_tracker = st.session_state.detection_mode_radio


    # --- Main Panel Logic ---
    # Use the value directly from the radio button's current state (via its key) for determining which panel to show. This is the most reliable source after a rerun.
    selected_mode_for_main_panel = st.session_state.detection_mode_radio

    if selected_mode_for_main_panel == "Rule-Based (Severity Scoring)":
        st.header("Rule-Based Anomaly Detection Results")
        df_processed_rb = df_original_cols.copy()
        df_processed_rb = initialize_anomaly_columns(df_processed_rb) # From rule_based_detector
        st.subheader("Detection Progress")
        progress_bar_rb = st.progress(0)
        status_text_rb = st.empty()
        rules_pipeline_rb = get_anomaly_detection_pipeline() # From rule_based_detector
        total_steps_rb = len(rules_pipeline_rb)
        for i, (message, rule_func) in enumerate(rules_pipeline_rb):
            status_text_rb.text(f"{message}...")
            try: df_processed_rb = rule_func(df_processed_rb)
            except Exception as e: status_text_rb.error(f"Error in '{message}': {e}"); st.exception(e); st.stop()
            progress_bar_rb.progress((i + 1) / total_steps_rb)
        df_final_rb = finalize_anomaly_data(df_processed_rb) # From rule_based_detector
        status_text_rb.success("Rule-based anomaly detection complete!")
        anomalous_df_rb = df_final_rb[df_final_rb['is_anomaly']].copy()
        if 'anomaly_reason' in anomalous_df_rb.columns:
            anomalous_df_rb['anomaly_reason_str'] = anomalous_df_rb['anomaly_reason'].apply(lambda x: '; '.join(x) if isinstance(x, list) and x else "N/A")
        else: anomalous_df_rb['anomaly_reason_str'] = "Reason error"
        if not anomalous_df_rb.empty and 'max_anomaly_severity_value' in anomalous_df_rb.columns and 'anomaly_score_percentage' in anomalous_df_rb.columns:
            anomalous_df_rb = anomalous_df_rb.sort_values(by=['max_anomaly_severity_value', 'anomaly_score_percentage'], ascending=[False, False])
        
        if not anomalous_df_rb.empty and 'applied_rule_codes' in anomalous_df_rb.columns:
            all_codes = [code for sl in anomalous_df_rb['applied_rule_codes'] for code in sl if isinstance(sl, list)]
            if all_codes:
                counts = pd.Series(all_codes).value_counts().rename(index=lambda rc: f"{rc}: {RULE_DEFINITIONS.get(rc, ('Unknown Rule', '', 0))[0]}")
                st.subheader("Top Anomaly Reasons Counts (Rule-Based):"); st.dataframe(counts)
        
        st.subheader(f"Anomalous Rows Found (Rule-Based): {len(anomalous_df_rb)}")
        if not anomalous_df_rb.empty:
            st.markdown("""
            **Legend for Anomaly Severity (Rule-Based):**
            - <span style='background-color:#FFCCCB;color:black;padding:2px 5px;border-radius:3px;'>Red</span>: Critical data errors or biologically implausible values.
              > *Examples: BMI outside [12-60], Age > 120 or <= 0, Height < 50cm or > 220cm, Weight < 20kg or > 300kg.*
            - <span style='background-color:#FFD580;color:black;padding:2px 5px;border-radius:3px;'>Orange</span>: Suspicious data, inconsistencies, or potential data management issues.
              > *Examples: Elderly (Age > 85) & Obese, Potential duplicate, BMI calculation mismatch, Missing critical age/weight/height.*
            - <span style='background-color:#FFFFE0;color:black;padding:2px 5px;border-radius:3px;'>Yellow</span>: Warnings for values at the edge of typical ranges.
              > *Examples: Age 0-18 or 100-120, Height 50-150cm.*
            """, unsafe_allow_html=True)
            st.markdown("---")

            display_cols_rb = ['id_cases', 'age_v', 'greutate', 'inaltime', 'IMC', 'imcINdex',
                               'max_anomaly_severity_category', 'anomaly_score_percentage',
                               'data1', 'anomaly_reason_str']
            # The 'calculated_bmi' for rule-based is now 'rule_based_calculated_bmi'
            if 'rule_based_calculated_bmi' in anomalous_df_rb.columns:
                if 'data1' in display_cols_rb: display_cols_rb.insert(display_cols_rb.index('data1'), 'rule_based_calculated_bmi')
                else: display_cols_rb.append('rule_based_calculated_bmi')
            
            actual_cols_rb = [col for col in display_cols_rb if col in anomalous_df_rb.columns]
            rows_to_show_rb = 1000
            show_all_rb = st.checkbox("Show all anomalous rows (Rule-Based)?", value=len(anomalous_df_rb) <= rows_to_show_rb, key="show_all_rb_checkbox")
            display_subset_rb = anomalous_df_rb if show_all_rb else anomalous_df_rb.head(rows_to_show_rb)
            if not show_all_rb and len(anomalous_df_rb) > rows_to_show_rb:
                st.caption(f"Displaying first {rows_to_show_rb} of {len(anomalous_df_rb)} anomalous rows.")
            
            if AGGRID_AVAILABLE: # Use AgGrid for Rule-Based Anomalous Table
                gb_rb = GridOptionsBuilder.from_dataframe(display_subset_rb[actual_cols_rb])
                gb_rb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
                gb_rb.configure_default_column(editable=False, filterable=True, sortable=True, resizable=True, wrapText=True, autoHeight=True)
                cell_style_jscode_rb = JsCode("""
                function(params) {
                    if (params.data.max_anomaly_severity_category === 'Red') { return {'backgroundColor': '#FFCCCB', 'color': 'black'}; }
                    else if (params.data.max_anomaly_severity_category === 'Orange') { return {'backgroundColor': '#FFD580', 'color': 'black'}; }
                    else if (params.data.max_anomaly_severity_category === 'Yellow') { return {'backgroundColor': '#FFFFE0', 'color': 'black'}; }
                    return {'color': 'black'}; }; """)
                for col_name in actual_cols_rb: gb_rb.configure_column(col_name, cellStyle=cell_style_jscode_rb)
                gridOptions_rb = gb_rb.build()
                AgGrid(display_subset_rb[actual_cols_rb], gridOptions=gridOptions_rb, height=600, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, theme='streamlit', key='aggrid_rb')
            elif actual_cols_rb: # Fallback
                st.warning("streamlit-aggrid not installed. Using st.dataframe (might be slow).")
                st.dataframe(display_subset_rb[actual_cols_rb].style.apply(highlight_severity_rule_based, axis=1))
            
            @st.cache_data
            def convert_df_to_csv_r_local(input_df, cols): return input_df[cols].to_csv(index=False).encode('utf-8') # Renamed for clarity
            csv_anom_rb_dl = convert_df_to_csv_r_local(anomalous_df_rb, actual_cols_rb)
            st.download_button("Download ALL Anomalous Data (Rule-Based)", csv_anom_rb_dl, "anomalies_rule_based.csv", "text/csv", key="dl_anom_rb")

        valid_df_rb = df_final_rb[~df_final_rb['is_anomaly']].copy()
        cols_to_drop_rb_valid = list(set(valid_df_rb.columns) - set(df_original_cols.columns))
        valid_df_rb.drop(columns=cols_to_drop_rb_valid, inplace=True, errors='ignore')
        st.subheader(f"Valid Rows (Rule-Based) - First 5: {len(valid_df_rb)}")
        st.dataframe(valid_df_rb.head())
        if not valid_df_rb.empty:
            csv_valid_rb_dl = valid_df_rb.to_csv(index=False).encode('utf-8')
            st.download_button("Download Valid Data (Rule-Based)", csv_valid_rb_dl, "valid_rule_based.csv", "text/csv", key="dl_valid_rb")

    elif selected_mode_for_main_panel == "AI Supervised (XGBoost)":
        st.header("AI Supervised Anomaly Detection (Trained on Rule-Based Classification)")

        if df_original_cols is None or df_common_cols is None:
            st.error("Initial data (original or common) not loaded. Cannot proceed with AI mode.")
            st.stop()

        model_ai, scaler_ai = get_or_train_ai_model_cached(df_original_cols, df_common_cols)

        if model_ai and scaler_ai:
            # Create tabs for Supervised Results and Unsupervised Exploration
            tab1_supervised, tab2_unsupervised_explore = st.tabs([
                "📈 Supervised AI Results",
                "🔬 Unsupervised Exploration"
            ])

            with tab1_supervised:
                st.subheader("Supervised AI Model Output")
                # For applying the model, we use df_common_cols
                df_ai_input = df_common_cols.copy()
                df_validated_for_ai_prediction = validate_data_for_ai(df_ai_input)
                st.write(f"Data pre-validated for AI prediction: {len(df_validated_for_ai_prediction[df_validated_for_ai_prediction['valid']])} valid rows, "
                         f"{len(df_validated_for_ai_prediction[~df_validated_for_ai_prediction['valid']])} invalid rows (by simple src.validators). "
                         f"AI model will be applied to all {len(df_validated_for_ai_prediction)} rows.")

                with st.spinner("Applying supervised AI model to all data..."):
                    df_ai_processed = apply_supervised_model(df_validated_for_ai_prediction.copy(), model_ai, scaler_ai)
                df_ai_processed["anomaly_risk_category_text"] = df_ai_processed["ai_anomaly_score"].apply(get_ai_risk_color_text)

                # Note: AI Mode Filtering in the sidebar will apply to this tab's content.
                # ask the mtf if you want separate filters for unsupervised exploration or if global AI filters are okay.
                # For simplicity, current sidebar AI filters will affect this supervised results table.

                df_show_ai = df_ai_processed.copy() # Start with all processed data
                # Apply sidebar filters (this code is already present and should work here)
                if df_show_ai["age"].notna().any() and 'age_min_ai' in st.session_state and 'age_max_ai' in st.session_state : # Check if filters are set
                    df_show_ai = df_show_ai[(df_show_ai["age"] >= st.session_state.age_min_ai) & (df_show_ai["age"] <= st.session_state.age_max_ai)]
                if df_show_ai["bmi"].notna().any() and 'bmi_min_ai' in st.session_state and 'bmi_max_ai' in st.session_state:
                    df_show_ai = df_show_ai[(df_show_ai["bmi"] >= st.session_state.bmi_min_ai) & (df_show_ai["bmi"] <= st.session_state.bmi_max_ai)]
                if 'sex_ai' in st.session_state and st.session_state.sex_ai != "Toate" and df_show_ai["sex"].notna().any():
                    df_show_ai = df_show_ai[df_show_ai["sex"] == st.session_state.sex_ai]
                if df_show_ai["ai_anomaly_score"].notna().any() and 'score_ai' in st.session_state:
                    df_show_ai = df_show_ai[df_show_ai["ai_anomaly_score"] >= st.session_state.score_ai]
                df_show_ai = df_show_ai.sort_values(by="ai_anomaly_score", ascending=False)


                st.subheader("📊 Tabel cu Date și Scoruri AI Supervizate")
                cols_to_show_ai = ['id_case', 'age', 'sex', 'weight', 'height', 'bmi_category', 'bmi', 'timestamp', 'ai_anomaly_score', 'anomaly_risk_category_text', 'valid', 'suspect_elderly_obese']
                actual_cols_ai = [col for col in cols_to_show_ai if col in df_show_ai.columns]

                if AGGRID_AVAILABLE and actual_cols_ai:
                    gb = GridOptionsBuilder.from_dataframe(df_show_ai[actual_cols_ai])
                    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
                    gb.configure_default_column(editable=False, filterable=True, sortable=True, resizable=True, wrapText=True, autoHeight=True)
                    cell_style_jscode_ai = JsCode("""
                    function(params) {
                        if (params.data.anomaly_risk_category_text === 'Red') { return {'backgroundColor': '#FFCCCB', 'color': 'black'}; }
                        else if (params.data.anomaly_risk_category_text === 'Orange') { return {'backgroundColor': '#FFD580', 'color': 'black'}; }
                        else if (params.data.anomaly_risk_category_text === 'Yellow') { return {'backgroundColor': '#FFFFE0', 'color': 'black'}; }
                        else if (params.data.anomaly_risk_category_text === 'Green') { return {'backgroundColor': 'lightgreen', 'color': 'black'}; }
                        return {'color': 'black'}; }; """)
                    for col_name_ag in actual_cols_ai: gb.configure_column(col_name_ag, cellStyle=cell_style_jscode_ai)
                    gridOptions = gb.build()
                    AgGrid(df_show_ai[actual_cols_ai], gridOptions=gridOptions, height=600, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, theme='streamlit', key='aggrid_ai_supervised_tab')
                elif actual_cols_ai:
                    st.warning("streamlit-aggrid not installed for optimal table display.")
                    st.dataframe(df_show_ai[actual_cols_ai])
                else:
                    st.info("No data to display with current AI supervised filters or columns missing.")


                st.markdown("---"); st.subheader("🧠 Explică un Caz cu SHAP (Model Supervizat)")
                if not df_show_ai.empty:
                    max_idx_shap = len(df_show_ai)-1
                    if max_idx_shap >=0:
                        idx_explain_ai = st.number_input("Alege un index de rând din tabelul filtrat:", 0, max_idx_shap, 0, 1, key="shap_idx_ai_tab")
                        if 0 <= idx_explain_ai <= max_idx_shap:
                            case_series_ai = df_show_ai.iloc[idx_explain_ai]
                            features_ai_shap = ["age", "weight", "height", "bmi"] # Renamed to avoid conflict
                            instance_shap = case_series_ai[features_ai_shap].copy()
                            if instance_shap.isna().any():
                                st.warning(f"Instanța SHAP are valori lipsă: {instance_shap[instance_shap.isna()].index.tolist()}. Se înlocuiesc cu 0.")
                                instance_shap.fillna(0, inplace=True)
                            with st.spinner("Generare SHAP..."):
                                try:
                                    explainer_s = shap.Explainer(model_ai, feature_names=features_ai_shap)
                                    X_selected_scaled_shap = scaler_ai.transform(instance_shap.values.reshape(1, -1))
                                    shap_values_s = explainer_s(X_selected_scaled_shap)
                                    st.write("Valori intrare SHAP:"); st.dataframe(instance_shap.to_frame().T)
                                    st.write("Plot SHAP Waterfall:")
                                    fig_s, ax_s = plt.subplots(figsize=(10,4)); shap.plots.waterfall(shap_values_s[0], show=False, max_display=10); st.pyplot(fig_s); plt.close(fig_s)
                                except Exception as e_s: st.error(f"Eroare SHAP: {e_s}")
                        else: st.info("Index SHAP invalid.")
                    else: st.info("Tabel filtrat gol pentru SHAP.")
                else: st.info("Tabel AI Supervizat gol pentru SHAP (verificați filtrele).")


                st.markdown("---"); st.subheader("📊 Diagrame Suplimentare (Mod AI)")
                if 'ai_anomaly_score' in df_ai_processed.columns:
                    col1_tab, col2_tab = st.columns(2)
                    with col1_tab:
                        fig_score_ai_tab = px.histogram(df_ai_processed, x="ai_anomaly_score", nbins=50, title="Distribuție Scor AI")
                        st.plotly_chart(fig_score_ai_tab, use_container_width=True)
                    with col2_tab:
                        fig_scatter_ai_tab = px.scatter(df_ai_processed, x="age", y="bmi", color="ai_anomaly_score", title="Vârstă vs. BMI (Scor AI)", color_continuous_scale=px.colors.sequential.Viridis)
                        st.plotly_chart(fig_scatter_ai_tab, use_container_width=True)
                else:
                    st.info("Coloana 'ai_anomaly_score' lipsește din datele procesate pentru diagrame.")


            with tab2_unsupervised_explore:
                st.subheader("Unsupervised Anomaly Exploration on 'Rule-Normal' Data")
                st.write("""
                This section helps find potential new anomalies that your current rules might miss.
                It applies an Isolation Forest model to data that the rule-based system considered 'normal'.
                Lower 'unsupervised_iso_score' values (more negative) indicate higher anomaly likelihood.
                """)

                if st.button("🚀 Run Unsupervised Exploration", key="unsupervised_explore_button_tab"):
                    if df_original_cols is None or df_common_cols is None:
                        st.error("Original or common data not loaded. Cannot perform unsupervised exploration.")
                    else:
                        with st.spinner("Preparing data for unsupervised exploration (running rules if needed)..."):
                            df_rb_for_unsupervised = df_original_cols.copy()
                            df_rb_for_unsupervised = initialize_anomaly_columns(df_rb_for_unsupervised)
                            rules_pipeline_temp = get_anomaly_detection_pipeline()
                            for _, rule_func_temp in rules_pipeline_temp:
                                df_rb_for_unsupervised = rule_func_temp(df_rb_for_unsupervised)
                            df_rules_output_for_unsupervised = finalize_anomaly_data(df_rb_for_unsupervised)
                        
                        df_rule_normal_orig_cols = df_rules_output_for_unsupervised[~df_rules_output_for_unsupervised['is_anomaly']].copy()

                        if df_rule_normal_orig_cols.empty:
                            st.warning("No data classified as 'normal' by rules found to explore.")
                        else:
                            if 'id_cases' not in df_rule_normal_orig_cols.columns or 'id_case' not in df_common_cols.columns:
                                st.error("ID columns missing for merging rule-normal data with common features.")
                            else:
                                df_rule_normal_common_cols = df_common_cols.merge(
                                    df_rule_normal_orig_cols[['id_cases']], 
                                    left_on='id_case',
                                    right_on='id_cases',
                                    how='inner' 
                                )
                                if df_rule_normal_common_cols.empty:
                                    st.warning("No 'rule-normal' data found after merging with common features data.")
                                else:
                                    st.write(f"Exploring {len(df_rule_normal_common_cols)} records considered 'normal' by rules.")
                                    features_for_unsupervised = ["age", "weight", "height", "bmi"]
                                    if not all(f in df_rule_normal_common_cols.columns for f in features_for_unsupervised):
                                        st.error(f"One or more features for unsupervised model ({features_for_unsupervised}) not found in the 'rule-normal' common data.")
                                    else:
                                        df_unsupervised_input = df_rule_normal_common_cols[features_for_unsupervised].copy()
                                        df_unsupervised_input = df_unsupervised_input.replace([np.inf, -np.inf], pd.NA).fillna(0)

                                        scaler_unsupervised = StandardScaler()
                                        X_scaled_unsupervised = scaler_unsupervised.fit_transform(df_unsupervised_input)

                                        iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42) 
                                        with st.spinner("Fitting Isolation Forest model..."):
                                            iso_forest.fit(X_scaled_unsupervised)
                                        
                                        unsupervised_scores = iso_forest.decision_function(X_scaled_unsupervised)
                                        
                                        df_exploration_results = df_rule_normal_common_cols.copy()
                                        df_exploration_results["unsupervised_iso_score"] = unsupervised_scores
                                        df_exploration_results["unsupervised_is_potential_anomaly"] = iso_forest.predict(X_scaled_unsupervised) == -1
                                        st.session_state.df_exploration_results = df_exploration_results # Store in session state

                # Display exploration results if available in session state
                if 'df_exploration_results' in st.session_state and st.session_state.df_exploration_results is not None:
                    df_to_show_exploration = st.session_state.df_exploration_results
                    df_potential_new_anomalies = df_to_show_exploration[df_to_show_exploration["unsupervised_is_potential_anomaly"] == True]
                    df_potential_new_anomalies = df_potential_new_anomalies.sort_values(by="unsupervised_iso_score", ascending=True)

                    st.subheader(f"Potential New Anomalies Found: {len(df_potential_new_anomalies)}")
                    st.write("These records were considered 'normal' by rules but flagged by Isolation Forest. Review them carefully.")
                    
                    cols_to_display_exploration = ['id_case', 'age', 'weight', 'height', 'bmi', 'unsupervised_iso_score'] + [
                        col for col in ['bmi_category', 'timestamp'] if col in df_potential_new_anomalies.columns
                    ]
                    actual_cols_exploration = [c for c in cols_to_display_exploration if c in df_potential_new_anomalies.columns]

                    if not df_potential_new_anomalies.empty and actual_cols_exploration:
                        if AGGRID_AVAILABLE:
                            gb_exp = GridOptionsBuilder.from_dataframe(df_potential_new_anomalies[actual_cols_exploration])
                            gb_exp.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
                            # gb_exp.configure_selection(selection_mode='multiple', use_checkbox=True) # Example for row selection
                            gb_exp.configure_default_column(editable=False, filterable=True, sortable=True, resizable=True, wrapText=True, autoHeight=True)
                            gridOptions_exp = gb_exp.build()
                            AgGrid(df_potential_new_anomalies[actual_cols_exploration], gridOptions=gridOptions_exp, height=400, fit_columns_on_grid_load=False, theme='streamlit', key='aggrid_exploration_tab')
                        else:
                            st.dataframe(df_potential_new_anomalies[actual_cols_exploration])
                        st.info("Next step: Manually review these cases. If confirmed as anomalies, their IDs can be used to augment the training data for the supervised AI model.")
                    elif not df_potential_new_anomalies.empty and not actual_cols_exploration:
                        st.warning("Potential new anomalies found, but no columns configured for display in the exploration table.")
                    else:
                        st.success("Isolation Forest did not flag any significant new outliers on the 'rule-normal' data with current settings (or no 'rule-normal' data was processed).")
        else: # This else is for the if model_ai and scaler_ai: confusing aaah code
            st.error("AI Model (supervised) or scaler not available. Cannot proceed with AI mode operations.")

else: # This else is for the if df_original_cols is not None and df_common_cols is not None:
    st.error("Datele inițiale nu au putut fi încărcate.")