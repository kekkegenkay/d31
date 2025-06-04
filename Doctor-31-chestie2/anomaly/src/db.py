# from sqlalchemy import create_engine
# import pandas as pd

# Schimbă aici parola dacă ai setat alta
# DB_USER = "postgres"
# DB_PASSWORD = "1234"
# DB_HOST = "localhost"
# DB_PORT = "5432"
# DB_NAME = "anomaly_db"

# def save_to_postgres(df: pd.DataFrame, table_name: str):
#    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
#    df.to_sql(table_name, engine, index=False, if_exists="replace")

import os
from sqlalchemy import create_engine
import pandas as pd

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "anomaly_db")

def save_to_postgres(df: pd.DataFrame, table_name: str):
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df.to_sql(table_name, engine, index=False, if_exists="replace")
