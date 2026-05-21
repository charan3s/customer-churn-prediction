import pandas as pd

da=pd.read_csv("churnguard_data.csv") # if not in same folder then use the path of that file.

df=pd.DataFrame(da)

df=df.drop("customerID",axis=1) # drop used to remove rows or columns.

df=df.drop_duplicates() # removes duplicated rows

col=["gender","PaymentMethod"]

for i in col:
    df[i]=df[i].str.strip() # to remove  left,right end spaces from multiple columns.

cols=["Churn","PhoneService","PaperlessBilling"]

for i in cols:
    df[i]=df[i].str.strip().str.title() # remove left,right end spaces and convert string to tiltle like first letter capital next all small letters.

df["Contract"]=df["Contract"].str.strip().str.lower() #remove left,right end spaces and convert string to lower case letters.

df["Contract"].unique() #gives unique values.

df["Contract"]=df["Contract"].replace({ # replace is used to modify the values mentioned and not mentioned values in columns stays as it is but if we use map they become NAN.
    "1 year":"One year",
    "2 year":"Two year",
    "one year":"One year",
    "two year":"Two year",
    "month to month":"Month-to-month",
    "monthly":"Month-to-month",
    "month-to-month":"Month-to-month"
})

df["InternetService"]=df["InternetService"].str.strip().str.lower()

df["InternetService"].unique()

df["InternetService"]=df["InternetService"].replace({
    "dsl": "DSL",
    "fiberoptic": "Fiber optic",
    "fibre optic": "Fiber optic",
    "fiber optic": "Fiber optic",
    "none": "No",
    "no": "No"
})

df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce") # convert "12"(string numbers) to actual numbers(12) because the array operations can't be performed on string they need to be numericals.
# errors=coerce is used to replace junk values to NAN like ("AB"," ","e3" are not numbers so they are junt so they are replaced with NAN)

df=df[~(df["tenure"]<=0)] # removes rows based on condition we must use ~(condition) or (if df["a"]< 10 need to be removed means do df[df["a"]>=10] or ~(df[df["a"]<10]))

df=df[~((df["MonthlyCharges"]<10) | (df["MonthlyCharges"]>200))] # removes rows based on two condition but to use or(|) for and(&)

df["MonthlyCharges"]=df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())

df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].mean())

df["tenure"]=df["tenure"].fillna(round(df["tenure"].median()))

print(df.shape)

print(df.isnull().sum())

df["Churn"]=df["Churn"].map({"Yes":1,"No":0})

df['Contract']=df['Contract'].map({"Month-to-month":0,"One year":1,"Two year":2})

df=df.drop(["gender","PhoneService","InternetService","PaperlessBilling","PaymentMethod"],axis=1)

x_train=df.drop("Churn",axis=1)

y_train=df["Churn"] # we train entire data set

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

# taking input for the required features.
tenure_months=int(input("Enter tenure (months):"))

monthly_charges=float(input("Enter Monthly Charges:"))

total_charges=float(input("Enter Total Charges:"))

senior_citizen=int(input("Senior Citizen? (1 = Yes, 0 = No):"))

contract_type=int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year):"))

x_test_input=[[senior_citizen,tenure_months,contract_type,monthly_charges,total_charges]]

predicted_output=model.predict(x_test_input) # model.predict() returns numpy array.

if(predicted_output[0]==0):
  print("Prediction: This customer is likely to STAY.")

else:
  print("Prediction: This customer is likely to CHURN.")