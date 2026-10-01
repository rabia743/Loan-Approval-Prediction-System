<div align="center">

<img src="bank-story.svg" width="100%" alt="Customer walks into bank and gets loan approved"/>

</div>

# 🏦 Loan Approval Prediction System

This is a machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on an applicant's personal and financial information.

I built this project using **Python, Pandas, Scikit-learn, Decision Tree, and Flask**.

## 📋 About the Project

The model uses information such as **income, loan amount, loan term, CIBIL score, education, employment status, and asset values** to predict the loan status.

This project covers the basic machine learning workflow, from preparing the dataset to training the model and making predictions through a web application.

## 🔹 What This Project Includes

* Data cleaning and preprocessing
* Feature engineering
* Label encoding
* Decision Tree classification
* Model evaluation
* Saving the trained model
* Flask web application for prediction

## 🤖 Machine Learning Model

**Algorithm:** Decision Tree Classifier

**Target Column:** `Loan_Status`

The model predicts two outcomes:

* **Approved**
* **Rejected**

## 📊 Features Used

* No. of Dependents
* Education
* Self Employed
* Income
* Loan Amount
* Loan Term
* CIBIL Score
* Residential Assets Value
* Commercial Assets Value
* Luxury Assets Value
* Bank Asset Value

## 🗂️ Project Structure

```text
loan-approval-prediction/
│
├── cleaned data loan.csv
├── loan_feature_engineered.csv
├── loan_model.pkl
├── education_encoder.pkl
├── self_employed_encoder.pkl
├── loan_status_encoder.pkl
├── app.py
├── train_model.py
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Flask
* HTML/CSS
* Joblib

## 🚀 How to Run

First, clone the repository:

```bash
git clone <https://github.com/rabia743/Loan-Approval-Prediction-System.git>
```

Then move into the project folder:

```bash
cd loan-approval-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
python app.py
```

Now open this in your browser:

```text
http://127.0.0.1:5000/
```

## 🎯 Project Goal

The main goal of this project is to practice **machine learning classification** and understand how a trained model can be connected to a simple web application for making predictions.

