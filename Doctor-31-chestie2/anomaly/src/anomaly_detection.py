import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.db import save_to_postgres


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplică Isolation Forest pentru a detecta anomalii în datele valide.
    """
    df = df.copy()

    # Ne uităm doar la coloanele relevante și la datele deja validate
    features = ["age", "weight", "height", "bmi"]
    df_valid = df[df["valid"]].dropna(subset=features)

    # Normalizare date
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_valid[features])

    # Aplicare Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    df_valid["ai_anomaly"] = model.fit_predict(X_scaled)

    # Convertim: -1 = anomalie, 1 = normal
    df_valid["ai_anomaly"] = df_valid["ai_anomaly"] == -1

    # Îmbinăm rezultatele înapoi în DataFrame-ul complet
    df = df.merge(df_valid[["ai_anomaly"]], left_index=True, right_index=True, how="left")
    df["ai_anomaly"] = df["ai_anomaly"].fillna(False)

    return df


if __name__ == "__main__":
    from src.data_loader import load_data_from_pdf
    from src.validators import validate_data

    # Încarcă și prelucrează datele
    df = load_data_from_pdf("dataset.pdf")
    df = validate_data(df)
    df = detect_anomalies(df)

    # Salvare în fișiere CSV
    df[df["valid"]].to_csv("export_date_valide.csv", index=False)
    df[df["ai_anomaly"]].to_csv("export_anomalii_ai.csv", index=False)

    print("\n📁 Datele au fost salvate cu succes în:")
    print("- export_date_valide.csv (doar datele valide)")
    print("- export_anomalii_ai.csv (anomaliile detectate de AI)")

    # Salvare în PostgreSQL
    save_to_postgres(df[df["valid"]], "date_valide")
    save_to_postgres(df[df["ai_anomaly"]], "anomalii_ai")

    # Afișează exemple
    print("\n🔎 Exemple detectate ca anomalii de AI:")
    print(df[df["ai_anomaly"]][["age", "weight", "height", "bmi", "ai_anomaly"]].head())
