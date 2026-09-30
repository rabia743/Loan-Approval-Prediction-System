import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

loan = pd.read_csv("cleaned data loan.csv")
# print(loan.head(5))


# String Convert

education_encoder = LabelEncoder()
self_employed_encoder = LabelEncoder()
loan_status_encoder = LabelEncoder()


loan["Education"] = education_encoder.fit_transform(loan["Education"])
loan["Self_Employed"] = self_employed_encoder.fit_transform(loan["Self_Employed"])
loan["Loan_Status"] = loan_status_encoder.fit_transform(loan["Loan_Status"])
# print(loan.head(5))
# No scaling needed because Decision Tree does not require scaling.

# Save feature-engineered data
# loan.to_csv("loan_feature_engineered.csv", index=False)

# Save Encoders
joblib.dump(education_encoder, "education_encoder.pkl")
joblib.dump(self_employed_encoder, "self_employed_encoder.pkl")
joblib.dump(loan_status_encoder, "loan_status_encoder.pkl")
print("Feature Engineering Complete")
print("Encoders Saved")