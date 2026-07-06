"""
generate_dataset.py
--------------------
Generates a synthetic but realistic dataset that mirrors the structure of the
Kaggle "Credit Card Approval Prediction" dataset (application_record.csv +
credit_record.csv). This lets the whole pipeline run end-to-end offline.

If you have the real Kaggle dataset, just drop application_record.csv and
credit_record.csv into this /data folder and skip running this script.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_APPLICANTS = 6000


def generate_application_record(n=N_APPLICANTS):
    ids = np.arange(1000001, 1000001 + n)

    gender = np.random.choice(["M", "F"], size=n, p=[0.42, 0.58])
    own_car = np.random.choice(["Y", "N"], size=n, p=[0.4, 0.6])
    own_realty = np.random.choice(["Y", "N"], size=n, p=[0.65, 0.35])
    cnt_children = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.22, 0.15, 0.06, 0.02])

    income_type = np.random.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n, p=[0.5, 0.22, 0.17, 0.1, 0.01]
    )
    amt_income_total = np.round(np.random.lognormal(mean=11.8, sigma=0.5, size=n), -2)
    amt_income_total = np.clip(amt_income_total, 27000, 1600000)

    education_type = np.random.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        size=n, p=[0.55, 0.32, 0.08, 0.03, 0.02]
    )
    family_status = np.random.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.6, 0.17, 0.08, 0.08, 0.07]
    )
    housing_type = np.random.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.72, 0.11, 0.08, 0.06, 0.02, 0.01]
    )

    days_birth = -np.random.randint(20 * 365, 65 * 365, size=n)  # negative = age in days
    # Pensioners get a large positive DAYS_EMPLOYED sentinel like the real dataset (365243)
    days_employed = -np.random.randint(0, 40 * 365, size=n)
    pensioner_mask = income_type == "Pensioner"
    days_employed[pensioner_mask] = 365243

    flag_mobil = np.ones(n, dtype=int)
    flag_work_phone = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    flag_phone = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    flag_email = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

    occupation_type = np.random.choice(
        ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
         "High skill tech staff", "Accountants", "Medicine staff",
         "Security staff", "Cooking staff", "Cleaning staff", "Other"],
        size=n
    )
    cnt_fam_members = np.clip(cnt_children + np.random.choice([1, 2], size=n, p=[0.3, 0.7]), 1, 8)

    df = pd.DataFrame({
        "ID": ids,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": amt_income_total,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": flag_mobil,
        "FLAG_WORK_PHONE": flag_work_phone,
        "FLAG_PHONE": flag_phone,
        "FLAG_EMAIL": flag_email,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": cnt_fam_members,
    })
    return df


def generate_credit_record(app_df):
    """Generate month-by-month credit history rows per applicant, mirroring the
    STATUS coding scheme used by the real dataset:
      0     : 1-29 days past due
      1     : 30-59 days past due
      2     : 60-89 days past due
      3     : 90-119 days past due
      4     : 120-149 days past due
      5     : overdue / bad debt (>150 days)
      C     : paid off that month
      X     : no loan that month
    """
    rows = []
    statuses = ["C", "X", "0", "1", "2", "3", "4", "5"]

    for _id in app_df["ID"]:
        # Give each applicant a "risk propensity" so their history is internally consistent
        risk = np.random.beta(1.5, 6)  # skewed toward low risk
        n_months = np.random.randint(3, 25)
        probs = np.array([
            0.45 - 0.2 * risk,   # C paid off
            0.30,                # X no loan
            0.15 + 0.3 * risk,   # 0 (1-29 dpd)
            0.05 + 0.2 * risk,   # 1
            0.02 + 0.1 * risk,   # 2
            0.01 + 0.05 * risk,  # 3
            0.01 + 0.05 * risk,  # 4
            0.01 + 0.05 * risk,  # 5
        ])
        probs = np.clip(probs, 0.001, None)
        probs = probs / probs.sum()

        month_status = np.random.choice(statuses, size=n_months, p=probs)
        for m, status in enumerate(month_status):
            rows.append((_id, -m, status))

    credit_df = pd.DataFrame(rows, columns=["ID", "MONTHS_BALANCE", "STATUS"])
    return credit_df


if __name__ == "__main__":
    app_df = generate_application_record()
    credit_df = generate_credit_record(app_df)

    app_df.to_csv("/home/claude/credit_card_approval/data/application_record.csv", index=False)
    credit_df.to_csv("/home/claude/credit_card_approval/data/credit_record.csv", index=False)

    print("application_record.csv:", app_df.shape)
    print("credit_record.csv:", credit_df.shape)
