import pandas as pd


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primește un DataFrame cu date medicale și returnează același tabel cu o coloană nouă: 'valid'.
    """
    df = df.copy()

    # Conversii numerice
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Reguli de validare
    df["valid"] = True

    # 1. Vârstă imposibilă
    df.loc[df["age"] > 120, "valid"] = False

    # 2. Greutate imposibilă
    df.loc[(df["weight"] < 30) | (df["weight"] > 300), "valid"] = False

    # 3. Înălțime imposibilă
    df.loc[(df["height"] < 120) | (df["height"] > 220), "valid"] = False

    # 4. BMI imposibil
    df.loc[(df["bmi"] < 12) | (df["bmi"] > 60), "valid"] = False

    # 5. Vârstnici supraponderali
    df["suspect_elderly_obese"] = False
    df.loc[(df["age"] > 85) & (df["bmi"] > 30), "suspect_elderly_obese"] = True

    return df


if __name__ == "__main__":
    from data_loader import load_data_from_pdf

    df = load_data_from_pdf("dataset.pdf")
    df_validated = validate_data(df)

    print("\n✅ Primele 5 rânduri validate:")
    print(df_validated[["age_v", "greutate", "inaltime", "imcINdex", "valid", "suspect_elderly_obese"]].head())
