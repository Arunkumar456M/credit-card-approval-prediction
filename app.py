"""
app.py
------
Flask web application for the Credit Card Approval Prediction system.
Loads the trained model (model/model.pkl) and serves:
  /            -> home page
  /predict     -> input form (GET) + prediction result (POST)
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model bundle once at startup
# ---------------------------------------------------------------------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
model_name = bundle["model_name"]

# Dropdown choices shown on the form (must match the values the encoders were
# fit on during training)
GENDER_CHOICES = ["FEMALE", "MALE"]
YES_NO_CHOICES = ["YES", "NO"]
INCOME_TYPE_CHOICES = sorted(encoders["NAME_INCOME_TYPE"].classes_.tolist())
EDUCATION_CHOICES = sorted(encoders["NAME_EDUCATION_TYPE"].classes_.tolist())
FAMILY_STATUS_CHOICES = sorted(encoders["NAME_FAMILY_STATUS"].classes_.tolist())
HOUSING_TYPE_CHOICES = sorted(encoders["NAME_HOUSING_TYPE"].classes_.tolist())


def encode_value(col, value):
    """Safely transform a raw categorical string using the fitted LabelEncoder,
    falling back to the first known class if an unseen value is submitted."""
    le = encoders[col]
    if value not in le.classes_:
        value = le.classes_[0]
    return le.transform([value])[0]


@app.route("/")
def home():
    return render_template("home.html", model_name=model_name)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None

    if request.method == "POST":
        form = request.form

        gender_raw = "F" if form.get("gender", "FEMALE").upper() == "FEMALE" else "M"
        own_car_raw = "Y" if form.get("own_car", "NO").upper() == "YES" else "N"
        own_realty_raw = "Y" if form.get("own_realty", "NO").upper() == "YES" else "N"

        income_type = form.get("income_type", INCOME_TYPE_CHOICES[0])
        education = form.get("education", EDUCATION_CHOICES[0])
        family_status = form.get("family_status", FAMILY_STATUS_CHOICES[0])
        housing_type = form.get("housing_type", HOUSING_TYPE_CHOICES[0])

        annual_income = float(form.get("annual_income", 0) or 0)
        days_birth = float(form.get("days_birth", 0) or 0)
        days_employed = float(form.get("days_employed", 0) or 0)
        family_members = float(form.get("family_members", 1) or 1)
        emi_paid_off = float(form.get("emi_paid_off", 0) or 0)
        emi_pastdues = float(form.get("emi_pastdues", 0) or 0)
        number_of_loans = float(form.get("number_of_loans", 0) or 0)

        row = {
            "CODE_GENDER": encode_value("CODE_GENDER", gender_raw),
            "FLAG_OWN_CAR": encode_value("FLAG_OWN_CAR", own_car_raw),
            "FLAG_OWN_REALTY": encode_value("FLAG_OWN_REALTY", own_realty_raw),
            "AMT_INCOME_TOTAL": annual_income,
            "NAME_INCOME_TYPE": encode_value("NAME_INCOME_TYPE", income_type),
            "NAME_EDUCATION_TYPE": encode_value("NAME_EDUCATION_TYPE", education),
            "NAME_FAMILY_STATUS": encode_value("NAME_FAMILY_STATUS", family_status),
            "NAME_HOUSING_TYPE": encode_value("NAME_HOUSING_TYPE", housing_type),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "CNT_FAM_MEMBERS": family_members,
            "EMI_PAID_OFF": emi_paid_off,
            "EMI_PASTDUES": emi_pastdues,
            "NUMBER_OF_LOANS": number_of_loans,
        }

        X = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][int(pred)] if hasattr(model, "predict_proba") else None

        prediction = "APPROVED" if pred == 1 else "REJECTED"
        probability = round(float(proba) * 100, 2) if proba is not None else None

    return render_template(
        "predict.html",
        gender_choices=GENDER_CHOICES,
        yes_no_choices=YES_NO_CHOICES,
        income_type_choices=INCOME_TYPE_CHOICES,
        education_choices=EDUCATION_CHOICES,
        family_status_choices=FAMILY_STATUS_CHOICES,
        housing_type_choices=HOUSING_TYPE_CHOICES,
        prediction=prediction,
        probability=probability,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
