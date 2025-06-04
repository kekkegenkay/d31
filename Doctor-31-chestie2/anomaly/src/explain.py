import shap
import pandas as pd
import numpy as np

def explain_instance(model, scaler, instance: pd.Series, feature_names):
    """
    Returnează explicația SHAP pentru o singură instanță (caz).
    """
    # Pregătește inputul
    X = instance[feature_names].values.reshape(1, -1)
    X_scaled = scaler.transform(X)

    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(X_scaled)

    # Returnăm ca dicționar pentru afișare
    return dict(zip(feature_names, shap_values.values[0]))
