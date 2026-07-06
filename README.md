# Credit Card Approval Prediction System

An end-to-end Machine Learning project that predicts whether a credit card
application will be **approved** or **rejected**, based on an applicant's
financial and demographic profile. Includes data preprocessing, feature
engineering, four classification models, and a Flask web application for
real-time predictions.

## Project Structure

```
credit_card_approval/
├── app.py                     # Flask web application
├── model_training.py          # Preprocessing, training, evaluation, saves best model
├── requirements.txt
├── README.md
├── data/
│   ├── generate_dataset.py    # Creates a synthetic application/credit dataset
│   ├── application_record.csv
│   └── credit_record.csv
├── model/
│   └── model.pkl              # Best trained model + encoders (created by model_training.py)
├── templates/
│   ├── base.html
│   ├── home.html
│   └── predict.html
└── static/
    └── style.css
```

## Dataset

This project is built to work with the Kaggle **"Credit Card Approval
Prediction"** dataset (`application_record.csv` + `credit_record.csv`).

- If you have the real Kaggle files, download them from
  https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
  and place them in the `data/` folder (overwriting the sample files).
- If you don't, this project ships with `data/generate_dataset.py`, which
  creates a synthetic dataset with the same column structure so the whole
  pipeline runs out of the box.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the sample dataset
python data/generate_dataset.py

# 4. Train the models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
#    Evaluates each with accuracy / confusion matrix / classification report,
#    then saves the best-performing model to model/model.pkl
python model_training.py

# 5. Run the Flask web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- `/` — Home page with a project overview
- `/predict` — Enter applicant details and get an instant Approved / Rejected prediction

## How It Works

1. **Feature Engineering** — `credit_record.csv` contains a month-by-month
   payment `STATUS` per applicant (`C` = paid off, `X` = no loan that month,
   `0`–`5` = days-past-due buckets). These are aggregated per applicant into:
   - `EMI_PAID_OFF` — count of months paid off
   - `EMI_PASTDUES` — count of overdue months
   - `NUMBER_OF_LOANS` — count of active loan months
   - `IS_HIGH_RISK` — 1 if the applicant was ever 60+ days past due

2. **Target** — `APPROVAL_STATUS = 1 - IS_HIGH_RISK` (Approved vs. Rejected).

3. **Model Training** — Categorical fields (gender, income type, education,
   family status, housing type, own car/realty) are label-encoded, then four
   classifiers are trained and compared:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost

   The model with the highest test accuracy is saved to `model/model.pkl`
   along with its encoders, for use by the Flask app.

4. **Web App** — `app.py` loads `model.pkl` and exposes a form
   (`/predict`) matching the fields used during training: gender, own
   car/realty, annual income, income type, education, family status,
   housing type, days of birth/employment, family members, EMI paid off,
   EMI past dues, and number of loans.

## Notes

- Swap in the real Kaggle dataset any time — no code changes are needed as
  long as the column names match (`CODE_GENDER`, `FLAG_OWN_CAR`,
  `AMT_INCOME_TOTAL`, etc.). Just re-run `model_training.py` afterward.
- To deploy elsewhere (e.g. IBM Cloud, Render, Heroku), point their runtime
  at `app.py`'s Flask `app` object, and make sure `model/model.pkl` is
  included in the deployment package.
