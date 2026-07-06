"""
model_training.py
------------------
Loads application_record.csv + credit_record.csv, performs data cleaning,
feature engineering, encodes categorical variables, trains four
classification models (Logistic Regression, Decision Tree, Random Forest,
XGBoost), evaluates them, and saves the best-performing model (plus the
encoders / feature list needed for inference) to model/model.pkl.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = "/home/claude/credit_card_approval/data"
MODEL_DIR = "/home/claude/credit_card_approval/model"

CATEGORICAL_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
]

FEATURE_COLS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH",
    "DAYS_EMPLOYED", "CNT_FAM_MEMBERS", "EMI_PAID_OFF",
    "EMI_PASTDUES", "NUMBER_OF_LOANS",
]


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
def load_data():
    app_df = pd.read_csv(f"{DATA_DIR}/application_record.csv")
    credit_df = pd.read_csv(f"{DATA_DIR}/credit_record.csv")
    return app_df, credit_df


# ---------------------------------------------------------------------------
# 2. Feature engineering on the credit_record table
# ---------------------------------------------------------------------------
def build_credit_features(credit_df):
    """Convert the multi-class STATUS codes into applicant-level features:
    EMI_PAID_OFF, EMI_PASTDUES, NUMBER_OF_LOANS and the high-risk target.
    """
    overdue_statuses = {"1", "2", "3", "4", "5"}
    bad_statuses = {"2", "3", "4", "5"}  # 60+ days past due -> high risk

    grouped = credit_df.groupby("ID")

    emi_paid_off = grouped["STATUS"].apply(lambda s: (s == "C").sum())
    emi_pastdues = grouped["STATUS"].apply(lambda s: s.isin(overdue_statuses).sum())
    number_of_loans = grouped["STATUS"].apply(lambda s: (s != "X").sum())
    is_high_risk = grouped["STATUS"].apply(lambda s: int(s.isin(bad_statuses).any()))

    feats = pd.DataFrame({
        "ID": emi_paid_off.index,
        "EMI_PAID_OFF": emi_paid_off.values,
        "EMI_PASTDUES": emi_pastdues.values,
        "NUMBER_OF_LOANS": number_of_loans.values,
        "IS_HIGH_RISK": is_high_risk.values,
    })
    return feats


# ---------------------------------------------------------------------------
# 3. Clean + merge
# ---------------------------------------------------------------------------
def preprocess(app_df, credit_df):
    credit_feats = build_credit_features(credit_df)

    df = app_df.merge(credit_feats, on="ID", how="inner")

    # Drop duplicates
    df = df.drop_duplicates(subset="ID")

    # Handle missing values
    df["OCCUPATION_TYPE"] = df.get("OCCUPATION_TYPE", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
    df = df.dropna(subset=FEATURE_COLS)

    # Target: 1 = Approved, 0 = Rejected (rejected == high risk applicant)
    df["APPROVAL_STATUS"] = 1 - df["IS_HIGH_RISK"]

    return df


# ---------------------------------------------------------------------------
# 4. Encode categoricals
# ---------------------------------------------------------------------------
def encode_features(df):
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


# ---------------------------------------------------------------------------
# 5. Train + evaluate all four models
# ---------------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 55)
    print(f"{name} — Accuracy: {acc:.4f}")
    print("=" * 55)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return acc


def main():
    print("Loading data...")
    app_df, credit_df = load_data()

    print("Preprocessing & feature engineering...")
    df = preprocess(app_df, credit_df)
    df_enc, encoders = encode_features(df)

    X = df_enc[FEATURE_COLS]
    y = df_enc["APPROVAL_STATUS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=42, n_estimators=150, use_label_encoder=False,
            eval_metric="logloss",
        ),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        acc = evaluate_model(name, model, X_test, y_test)
        results[name] = acc
        trained_models[name] = model

    best_name = max(results, key=results.get)
    best_model = trained_models[best_name]
    print("\n" + "#" * 55)
    print(f"Best model: {best_name} (Accuracy: {results[best_name]:.4f})")
    print("#" * 55)

    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "encoders": encoders,
        "feature_cols": FEATURE_COLS,
        "results": results,
    }, f"{MODEL_DIR}/model.pkl")

    print(f"\nSaved best model to {MODEL_DIR}/model.pkl")


if __name__ == "__main__":
    main()
