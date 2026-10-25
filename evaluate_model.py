
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

model_path = Path("models/churn_pipeline.joblib")
test_path = Path("reports/test_data.csv")

Path("reports").mkdir(exist_ok=True)

if not model_path.exists() or not test_path.exists():
    print("Run python task3_train_model.py before evaluating.")
    raise SystemExit(1)

model = joblib.load(model_path)
test_data = pd.read_csv(test_path)

y_test = test_data.pop("Churn").astype(int)
X_test = test_data

predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, zero_division=0)
recall = recall_score(y_test, predicted, zero_division=0)
f1 = f1_score(y_test, predicted, zero_division=0)
roc_auc = roc_auc_score(y_test, probabilities)

print("\nCUSTOMER CHURN MODEL EVALUATION")
print("--------------------------------")
print(f"Accuracy:         {accuracy:.4f}")
print(f"Churn Precision:  {precision:.4f}")
print(f"Churn Recall:     {recall:.4f}")
print(f"Churn F1-score:   {f1:.4f}")
print(f"ROC-AUC:          {roc_auc:.4f}")

report = classification_report(
    y_test,
    predicted,
    labels=[0, 1],
    target_names=["Stay", "Churn"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open("reports/evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write("Customer Churn Model Evaluation\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Churn Precision: {precision:.4f}\n")
    file.write(f"Churn Recall: {recall:.4f}\n")
    file.write(f"Churn F1-score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
    file.write(report)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicted,
    labels=[0, 1],
    display_labels=["Stay", "Churn"]
)

plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()

RocCurveDisplay.from_predictions(y_test, probabilities)

plt.title("Customer Churn - ROC Curve")
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150)
plt.close()

print("\nEvaluation report saved in reports/evaluation_report.txt")
print("Confusion matrix saved in reports/confusion_matrix.png")
print("ROC curve saved in reports/roc_curve.png")
