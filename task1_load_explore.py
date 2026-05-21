import pandas as pd

da=pd.read_csv("churnguard_data.csv")

df=pd.DataFrame(da)

print(df.shape) # return shape

print(df.head(5)) # give first 5 rows

print(df.info()) # gives column names,non null values counts,data type

print(df.isnull().sum()) # null values counts.

print(df.duplicated().sum()) # shows duplicated rows count

print(df['Churn'].value_counts()) # show count of all unique values.

print(df['Contract'].unique()) # shows unique values.