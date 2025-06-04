import pandas as pd
from src.anomaly_detection import detect_anomalies

def test_ai_anomaly_added():
    df = pd.DataFrame({
        "age": [25, 80, 90],
        "weight": [70, 130, 150],
        "height": [170, 160, 150],
        "bmi": [24.2, 50.7, 66.6],
        "valid": [True, True, True],
        "suspect_elderly_obese": [False, True, True]
    })
    df_out = detect_anomalies(df)
    assert "ai_anomaly" in df_out.columns
    assert df_out["ai_anomaly"].dtype == bool
