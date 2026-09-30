from flask import Flask, send_from_directory, request, jsonify
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__, static_folder='.', template_folder='.')

MODEL_PATH = "loan_model.pkl"
DATA_PATH = "loan_feature_engineered.csv"

FEATURES = [
    "No_Of_Dependents",
    "Education",
    "Self_Employed",
    "Income_Annum",
    "Loan_Amount",
    "Loan_Term",
    "Cibil_Score",
    "Residential_Assets_Value",
    "Commercial_Assets_Value",
    "Luxury_Assets_Value",
    "Bank_Asset_Value",
]


def train_and_save_model():
    loan = pd.read_csv(DATA_PATH)
    X = loan[FEATURES]
    y = loan["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=3,
        criterion="entropy",
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_and_save_model()


@app.route("/")
def index():
    return send_from_directory('.', 'index.html')


@app.route("/predict", methods=["POST"])
def predict_loan_status():
    # JSON aur Form dono support karein
    if request.is_json:
        form_data = request.get_json()
    else:
        form_data = request.form

    customer = {}
    for feature in FEATURES:
        customer[feature] = int(form_data.get(feature, 0))

    model = load_model()
    df = pd.DataFrame([customer], columns=FEATURES)

    prediction = int(model.predict(df)[0])

    # Probabilities bhi nikaalein
    proba = model.predict_proba(df)[0]
    # Aapke model mein: 0 = Approved, 1 = Rejected
    approved_prob = round(proba[0] * 100, 2)
    rejected_prob = round(proba[1] * 100, 2)

    result = "Approved" if prediction == 0 else "Rejected"

    return jsonify({
        "status": result,
        "approved_probability": approved_prob,
        "rejected_probability": rejected_prob,
    })


if __name__ == "__main__":
    app.run(debug=True)