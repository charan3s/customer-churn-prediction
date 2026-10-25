import pandas as pd

da=pd.read_csv("churnguard_data.csv")

df=pd.DataFrame(da)

print("Dataset shape:", df.shape) # return shape
print("\nFirst five rows:")
print(df.head())  # give first 5 rows

print("\nColumn details:")
print(df.info()) # gives column names,non null values counts,data type

print("\nMissing values:")
print(df.isnull().sum()) # null values counts.

print("\nDuplicate rows:", df.duplicated().sum()) # shows duplicated rows count

print("\nChurn distribution:")
print(df["Churn"].astype(str).str.strip().str.lower().value_counts()) 

print("\nContract values:")
print(df["Contract"].astype(str).str.strip().value_counts()) 

print("\nInternet service values:")
print(df["InternetService"].astype(str).str.strip().value_counts())