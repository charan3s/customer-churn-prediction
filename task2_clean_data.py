import pandas as pd

# Load the original data
df = pd.read_csv("churnguard_data.csv")
print("Original shape:", df.shape)

# Remove duplicate rows and duplicate customer records
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["customerID"], keep="first")

# Removing extra spaces
text_columns = [
    "gender", "PhoneService", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod", "Churn"
]

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

# Normalize labels which have inconsistent capitalization
df["gender"] = df["gender"].str.title()
df["PhoneService"] = df["PhoneService"].str.lower().replace({"yes": "Yes", "no": "No"})
df["PaperlessBilling"] = df["PaperlessBilling"].str.lower().replace({"yes": "Yes", "no": "No"})
df["Churn"] = df["Churn"].str.lower().replace({"yes": "Yes", "no": "No"})
df["PaymentMethod"] = df["PaymentMethod"].str.lower().replace({
    "electronic check": "Electronic check",
    "mailed check": "Mailed check",
    "bank transfer": "Bank transfer",
    "credit card": "Credit card"
})

# Standardize contract labels
df["Contract"] = df["Contract"].str.lower().replace({
    "month-to-month": "Month-to-month",
    "month to month": "Month-to-month",
    "monthly": "Month-to-month",
    "one year": "One year",
    "1 year": "One year",
    "two year": "Two year",
    "2 year": "Two year"
})

# Standardize internet service labels
df["InternetService"] = df["InternetService"].str.lower().replace({
    "dsl": "DSL",
    "fiberoptic": "Fiber optic",
    "fibre optic": "Fiber optic",
    "fiber optic": "Fiber optic",
    "no": "No",
    "none": "No"
})

# Convert numeric columns invalid text becomes missing data
numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["tenure"].isna() | (df["tenure"] > 0)]
df = df[df["MonthlyCharges"].isna() | ((df["MonthlyCharges"] >= 10) & (df["MonthlyCharges"] <= 200))]

# Keep records with a valid churn label
df = df[df["Churn"].isin(["Yes", "No"])]
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

# Customer ID is not a predictive feature, so remove it before modeling
df = df.drop(columns=["customerID"])
df = df.reset_index(drop=True)

print("Cleaned shape:", df.shape)
print("\nMissing values left for the training pipeline to handle:")
print(df.isnull().sum())
print("\nChurn counts (0 = Stay, 1 = Churn):")
print(df["Churn"].value_counts())

df.to_csv("cleaned_churn_data.csv", index=False)
print("\nSaved cleaned data to cleaned_churn_data.csv")