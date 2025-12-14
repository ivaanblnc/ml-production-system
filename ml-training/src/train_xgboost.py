import os
import time

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib


POSTGRES_USER = "mluser"
POSTGRES_PASSWORD = "mlpass"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "ml_data"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # ml-training/
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_fraud_model.pkl")


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
    df = df.copy()

    if df.empty:
        raise ValueError("No hay datos en transaction_features para entrenar el modelo.")

    threshold = df["max_amount_1m"].quantile(0.95)
    df["is_fraud"] = (df["max_amount_1m"] >= threshold).astype(int)

    return df


def train_xgboost(df: pd.DataFrame):
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

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": -1,
        "tree_method": "hist",
    }

    start = time.time()

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_time = time.time() - start

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"AUC: {auc:.4f}")
    print(f"Train time (s): {train_time:.2f}")
    print("Classification report:")
    print(classification_report(y_test, (y_proba >= 0.5).astype(int)))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo XGBoost guardado en: {MODEL_PATH}")


def main():
    df = load_features()
    df_labeled = add_synthetic_label(df)
    train_xgboost(df_labeled)


if __name__ == "__main__":
    main()
