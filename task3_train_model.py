from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# Create folders for generated files
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# Load cleaned data
df = pd.read_csv("cleaned_churn_data.csv")

categorical_columns = [
    "gender", "PhoneService", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
feature_columns = categorical_columns + numeric_columns

X = df[feature_columns]
y = df["Churn"].astype(int)

# Keep the test set separate from training and cross-validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
}

cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_score = -1
best_name = ""
best_pipeline = None

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("\nComparing models. Selection uses training-set cross-validation F1-score.\n")

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cross_validation, scoring="f1", n_jobs=None
    )

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    row = {
        "model": name,
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "test_accuracy": round(accuracy_score(y_test, predicted), 4),
        "test_precision_churn": round(precision_score(y_test, predicted, zero_division=0), 4),
        "test_recall_churn": round(recall_score(y_test, predicted, zero_division=0), 4),
        "test_f1_churn": round(f1_score(y_test, predicted, zero_division=0), 4),
        "test_roc_auc": round(roc_auc_score(y_test, probabilities), 4)
    }
    results.append(row)
    print(row)

    if cv_scores.mean() > best_score:
        best_score = cv_scores.mean()
        best_name = name
        best_pipeline = model

results_df = pd.DataFrame(results).sort_values("cv_f1_mean", ascending=False)
results_df.to_csv("reports/model_comparison.csv", index=False)

# Save the selected fitted pipeline, along with held-out test records
joblib.dump(best_pipeline, "models/churn_pipeline.joblib")
test_data = X_test.copy()
test_data["Churn"] = y_test.values
test_data.to_csv("reports/test_data.csv", index=False)

print("\nSelected model:", best_name)
print("Mean cross-validation F1-score:", round(best_score, 4))
print("Saved model: models/churn_pipeline.joblib")
print("Saved test records: reports/test_data.csv")
print("Saved comparison: reports/model_comparison.csv")