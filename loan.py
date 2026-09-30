import pandas as pd

loan = pd.read_csv("loan.csv")
print(loan.head(5))

# data cleaning
print(loan.isnull().sum())
print(loan.duplicated().sum())
loan.drop("loan_id", axis=1, inplace=True)
print(loan.columns)
print(loan.dtypes)

# text cleaning
loan.columns = loan.columns.str.strip().str.title()
# check rows and columns
print(loan.shape)
# SAVE CLEANED DATA BACK TO CSV FILE
# loan.to_csv("cleaned data loan.csv", index=False)
# print("Cleaned data saved successfully!")