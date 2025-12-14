import os

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


POSTGRES_USER = "mluser"
POSTGRES_PASSWORD = "mlpass"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "ml_data"


# Folder models inside of ml-training
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # ml-training/
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")


def load_features():
    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    query = """
        SELECT
            customer_id,
            transaction_count_1m,
            avg_amount_1m,
            max_amount_1m,
            fraud_count_1m
        FROM transaction_features
    """

    df = pd.read_sql_query(query, engine)
    return df


def add_synthetic_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etiquetas sintéticas de ejemplo:
    - Marca como fraude (1) el 5% de ventanas con mayor max_amount_1m.
    - El resto, 0.
    """

    df = df.copy()

    if df.empty:
        raise ValueError("No hay datos en transaction_features para entrenar el modelo.")

    threshold = df["max_amount_1m"].quantile(0.95)
    df["is_fraud"] = (df["max_amount_1m"] >= threshold).astype(int)

    return df


def train_and_save_model(df: pd.DataFrame):
    feature_cols = [
        "transaction_count_1m",
        "avg_amount_1m",
        "max_amount_1m",
        "fraud_count_1m",
    ]

    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")


def main():
    df = load_features()
    df_labeled = add_synthetic_label(df)
    train_and_save_model(df_labeled)


if __name__ == "__main__":
    main()
