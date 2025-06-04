import pandas as pd
from src.validators import validate_data

def test_bmi_invalid_cases():
    data = {
        "age": [30, 40],
        "weight": [40, 180],
        "height": [170, 160],
        "bmi": [11.5, 61.2]
    }
    df = pd.DataFrame(data)
    df_validated = validate_data(df)
    assert df_validated["valid"].tolist() == [False, False]
