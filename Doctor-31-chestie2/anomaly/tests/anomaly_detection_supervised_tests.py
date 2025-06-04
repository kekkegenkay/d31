import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.anomaly_detection_supervised import (
    generate_labels_from_rule_based_output,
    train_supervised_anomaly_model,
    apply_supervised_model
)

def test_generate_labels_from_rule_based_output_creates_labels():
    # Test case: DataFrame with 'is_anomaly' (boolean) and common features
    df_input = pd.DataFrame({
        "age": [25, 90, 30, 50],
        "weight": [70, 25, 80, np.nan], # Added a NaN to test feature handling later
        "height": [175, 160, 180, 170],
        "bmi": [22.0, 60.0, 24.7, 23.5],
        "is_anomaly": [False, True, True, False] # Mixed True/False
    })

    labeled_df = generate_labels_from_rule_based_output(df_input)

    assert "label" in labeled_df.columns
    assert "is_anomaly" in labeled_df.columns # Original column should persist
    assert set(labeled_df["label"].unique()) <= {0, 1} # Labels should be 0 or 1
    assert labeled_df["label"].tolist() == [0, 1, 1, 0] # Check conversion
    assert len(labeled_df) == 4 # Should process all rows

def test_generate_labels_handles_missing_is_anomaly_column():
    df_input = pd.DataFrame({
        "age": [25], "weight": [70], "height": [175], "bmi": [22.0]
    })
    try:
        generate_labels_from_rule_based_output(df_input)
        assert False, "Should have raised ValueError for missing 'is_anomaly' column"
    except ValueError as e:
        assert "must contain an 'is_anomaly' column" in str(e)

def test_generate_labels_handles_nan_in_is_anomaly():
    # NaNs in is_anomaly should be treated as False (0)
    df_input = pd.DataFrame({
        "age": [25, 90], "weight": [70, 25], "height": [175, 160], "bmi": [22.0, 60.0],
        "is_anomaly": [False, np.nan] 
    })
    labeled_df = generate_labels_from_rule_based_output(df_input)
    assert labeled_df["label"].tolist() == [0, 0]


def test_train_supervised_anomaly_model_trains_xgb_with_rule_labels():
    # Create a DataFrame suitable for training (has 'label' and features)
    # This data would be the output of generate_labels_from_rule_based_output
    df_labeled = pd.DataFrame({
        "age": [25, 90, 30, 85, 40, 22, 60, 70, 50, 45] * 3, # More data points
        "weight": [70, 25, 80, 150, 90, 65, 100, 30, 88, 78] * 3,
        "height": [175, 160, 180, 150, 170, 165, 155, 160, 177, 169] * 3,
        "bmi": [22.0, 10.0, 24.7, 66.6, 31.1, 23.8, 41.6, 11.7, 28.0, 27.3] * 3,
        "label": ([0, 1, 0, 1, 1, 0, 1, 1, 0, 0]) * 3 # Corresponding labels
    })
    # Add some NaNs to test robustness of training function's preprocessing
    df_labeled.loc[5, 'age'] = np.nan
    df_labeled.loc[10, 'bmi'] = np.nan
    df_labeled.loc[15, 'label'] = np.nan


    model, scaler = train_supervised_anomaly_model(df_labeled)

    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")
    # Check that the model was trained on non-NaN data
    assert model.n_features_in_ == 4 # Should be trained on 4 features

def test_train_model_raises_error_if_not_enough_data_or_one_class():
    # Not enough data
    df_few_rows = pd.DataFrame({
        "age": [25, 90], "weight": [70, 25], "height": [175, 160], "bmi": [22.0, 10.0], "label": [0, 1]
    })
    try:
        train_supervised_anomaly_model(df_few_rows)
        assert False, "Should raise error for too few rows"
    except ValueError as e:
        assert "Not enough valid samples" in str(e) or "Need at least 20" in str(e)

    # Only one class
    df_one_class = pd.DataFrame({
        "age": [25, 90, 30, 85, 40, 22, 60, 70, 50, 45] * 3,
        "weight": [70, 25, 80, 150, 90, 65, 100, 30, 88, 78] * 3,
        "height": [175, 160, 180, 150, 170, 165, 155, 160, 177, 169] * 3,
        "bmi": [22.0, 10.0, 24.7, 66.6, 31.1, 23.8, 41.6, 11.7, 28.0, 27.3] * 3,
        "label": ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * 3 # All label 0
    })
    try:
        train_supervised_anomaly_model(df_one_class)
        assert False, "Should raise error for only one class"
    except ValueError as e:
        assert "Only one class present" in str(e)


def test_apply_supervised_model_with_mocked_model_and_scaler():
    df_input = pd.DataFrame({
        "age": [25, np.nan, 30], # Test with NaN
        "weight": [70, 80, 90],
        "height": [175, 180, 185],
        "bmi": [22.0, 24.7, 26.3]
    })

    # Mock the model
    mock_model = MagicMock()
    # Expected predictions for 3 rows
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])

    # Mock the scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.random.rand(3, 4)


    result_df = apply_supervised_model(df_input, mock_model, mock_scaler)

    assert "ai_anomaly_score" in result_df.columns
    # Check that scores are assigned correctly based on model's predict_proba output (second column)
    pd.testing.assert_series_equal(
        result_df["ai_anomaly_score"],
        pd.Series([0.1, 0.8, 0.5], name="ai_anomaly_score"),
        check_dtype=False # Allow float differences
    )
    # Check that the scaler was called with data where NaNs were handled
    mock_scaler.transform.assert_called_once()
    # The argument passed to transform should be a numpy array of shape (3, 4)
    # and the first element of the first row (age) should be 0 if fillna(0) was used.
    call_args = mock_scaler.transform.call_args[0][0]
    assert call_args.shape == (3,4)
    assert call_args[0,0] == 0 # Assuming fillna(0) for age in the first row