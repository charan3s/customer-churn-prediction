
# ChurnGuard - Customer Churn Prediction

## Overview

ChurnGuard is a machine learning project that predicts whether a customer is likely to leave a service based on subscription and billing information.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Project Files

- `churnguard_data.csv` - Customer dataset
- `task1_load_explore.py` - Explores the dataset
- `task2_clean_data.py` - Cleans and prepares the data
- `task3_train_model.py` - Trains and compares classification models
- `evaluate_model.py` - Evaluates model performance
- `task4_predict.py` - Predicts churn for a new customer
- `requirements.txt` - Required Python packages

## Machine Learning Workflow

1. Load and explore customer data.
2. Clean duplicate records and standardize categories.
3. Handle missing values and encode categorical variables.
4. Compare Logistic Regression, Decision Tree, and Random Forest.
5. Select a model using cross-validation on the training data.
6. Evaluate the selected model on held-out test data.
7. Predict churn risk for new customers.

## Evaluation Metrics

The project measures:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Setup

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the scripts from the project folder in this order:

```powershell
python task1_load_explore.py
python task2_clean_data.py
python task3_train_model.py
python evaluate_model.py
python task4_predict.py
```

## Generated Outputs

The training script creates:

- models/churn_pipeline.joblib
- reports/test_data.csv
- reports/model_comparison.csv

The evaluation script creates:

- reports/evaluation_report.txt
- reports/confusion_matrix.png
- reports/roc_curve.png

## Limitations

Model performance depends on the dataset and evaluation method. The predicted churn probability is an estimate, not a guarantee that a customer will leave.
