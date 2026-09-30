import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load engineered dataset
loan = pd.read_csv("loan_feature_engineered.csv")

# Independent features
features = [
    "No_Of_Dependents", "Education", "Self_Employed", "Income_Annum",
    "Loan_Amount", "Loan_Term", "Cibil_Score",
    "Residential_Assets_Value", "Commercial_Assets_Value",
    "Luxury_Assets_Value", "Bank_Asset_Value"
]

X = loan[features]
y = loan["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Decision Tree model
model = DecisionTreeClassifier(
    max_depth=3,
    criterion="entropy",
    random_state=42
)
# if model train in random forest just change the libraray and add this syntax and remove the decision tree model train syntax
# model = RandomForestClassifier(
    # n_estimators=100,
    # random_state=42
# )

# Train model
model.fit(X_train, y_train)

# Predictions
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# Accuracy
print(f"Train Accuracy: {accuracy_score(y_train, train_prediction) * 100:.2f}%")
print(f"Test Accuracy:  {accuracy_score(y_test, test_prediction) * 100:.2f}%")

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_prediction))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, test_prediction))

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.2f}")

# Save model
# joblib.dump(model, "loan_model.pkl")
# print("\nModel saved: loan_model.pkl")

# New customer prediction using saved model
print("\nPredicting new customer loan status...\n")

# Because your dataset is already encoded:
# Education: 0 = Graduate, 1 = Not Graduate
# Self_Employed: 0 = No, 1 = Yes
# Loan_Status: 0 = Approved, 1 = Rejected

print("Customer")
name = input("Name: ")
customer_id = input("ID: ")
address = input("Address: ")
new_customer = {
    "No_Of_Dependents": int(input("No of Dependents: ")),
    "Education": int(input("Education (0 = Graduate, 1 = Not Graduate): ")),
    "Self_Employed": int(input("Self Employed (0 = No, 1 = Yes): ")),
    "Income_Annum": int(input("Annual Income: ")),
    "Loan_Amount": int(input("Loan Amount: ")),
    "Loan_Term": int(input("Loan Term: ")),
    "Cibil_Score": int(input("Cibil Score: ")),
    "Residential_Assets_Value": int(input("Residential Assets Value: ")),
    "Commercial_Assets_Value": int(input("Commercial Assets Value: ")),
    "Luxury_Assets_Value": int(input("Luxury Assets Value: ")),
    "Bank_Asset_Value": int(input("Bank Asset Value: "))
}

new_customer_df = pd.DataFrame([new_customer])
prediction = model.predict(new_customer_df[features])[0]

if prediction == 0:
    result = "Approved"
else:
    result = "Rejected"

print(f"Loan Status: {result}")
# print(f"Encoded Prediction: {prediction}")