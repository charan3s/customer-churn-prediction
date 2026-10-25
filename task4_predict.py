from pathlib import Path

import joblib
import pandas as pd

model_path = Path("models/churn_pipeline.joblib")

if not model_path.exists():
    print("Model not found. Run python task3_train_model.py first.")
    raise SystemExit(1)

model = joblib.load(model_path)

print("Enter customer details using the category labels shown in the prompts.\n")

gender = input("Gender (Female/Male): ").strip().title()
senior_citizen = int(input("Senior citizen? (1 = Yes, 0 = No): "))
tenure = float(input("Tenure in months: "))
phone_service = input("Phone service (Yes/No): ").strip().title()
internet_service = input("Internet service (DSL/Fiber optic/No): ").strip().lower()
contract = input("Contract (Month-to-month/One year/Two year): ").strip().lower()
paperless_billing = input("Paperless billing (Yes/No): ").strip().title()
payment_method = input(
    "Payment method (Electronic check/Mailed check/Bank transfer/Credit card): "
).strip().lower()
monthly_charges = float(input("Monthly charges: "))
total_charges = float(input("Total charges: "))

# Normalize the user-entered categories in the same way as the training data
internet_service = internet_service.lower()
internet_service = internet_service.replace("fiberoptic", "fiber optic")
internet_service = internet_service.replace("fibre optic", "fiber optic")
internet_service = internet_service.replace("none", "no")
internet_service = {
    "dsl": "DSL",
    "fiber optic": "Fiber optic",
    "no": "No"
}.get(internet_service, internet_service)

contract = contract.lower()
contract = {
    "month to month": "Month-to-month",
    "month-to-month": "Month-to-month",
    "monthly": "Month-to-month",
    "1 year": "One year",
    "one year": "One year",
    "2 year": "Two year",
    "two year": "Two year"
}.get(contract, contract)

payment_method = payment_method.lower()
payment_method = {
    "electronic check": "Electronic check",
    "mailed check": "Mailed check",
    "bank transfer": "Bank transfer",
    "credit card": "Credit card"
}.get(payment_method, payment_method)

customer = pd.DataFrame([{
    "gender": gender,
    "PhoneService": phone_service,
    "InternetService": internet_service,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

churn_probability = model.predict_proba(customer)[0, 1]
prediction = int(churn_probability >= 0.5)

print("\nEstimated churn probability:", f"{churn_probability:.1%}")
if prediction == 1:
    print("Prediction: Higher churn risk at the default 0.50 threshold.")
else:
    print("Prediction: Lower churn risk at the default 0.50 threshold.")
print("This is a model estimate, not a guarantee of customer behavior.")
