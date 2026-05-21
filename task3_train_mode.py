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

# In pd.get_dummies() ,each unique category(item) in a column becomes a new column.
df=pd.get_dummies(df,columns=["gender","PhoneService","InternetService","Contract","PaperlessBilling","PaymentMethod"],drop_first=True)

x=df.drop("Churn",axis=1)
y=df["Churn"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

acc=accuracy_score(y_test,y_pred)

print("Accuracy:",acc)

print(classification_report(y_test,y_pred,target_names=["Stay","Churn"]))