# =========================================================
# SECTION 1: LIBRARY IMPORT
# =========================================================

# Data Handling
import pandas as pd
import numpy as np

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Data Splitting
from sklearn.model_selection import train_test_split

# Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# SECTION 2: DATA LOADING
# =========================================================

# Load Dataset
df = pd.read_csv('raw_data/Telco-Customer-Churn.csv')

# Display First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# SECTION 3: DATA CLEANING
# =========================================================

# Convert TotalCharges into Numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Remove Missing Values
df = df.dropna()

# Verify Updated Data Types
print("\nUpdated Data Types:")
print(df.dtypes)

# Check Duplicate Records
duplicates = df.duplicated().sum()

print(f"\nDuplicate Records Found: {duplicates}")

# Remove Duplicates if Any
if duplicates > 0:
    df = df.drop_duplicates()
    print("\nDuplicate Records Removed!")
else:
    print("\nNo Duplicate Records Found.")

# Verify Dataset Shape After Cleaning
print("\nDataset Shape After Cleaning:")
print(df.shape)

# Churn Distribution
print("\nChurn Distribution:")
print(df['Churn'].value_counts())

# =========================================================
# SECTION 4: MACHINE LEARNING PREPROCESSING
# =========================================================

# Select Relevant Variables
model_df = df[[
    'tenure',
    'MonthlyCharges',
    'Contract',
    'PaymentMethod',
    'InternetService',
    'Churn'
]]

# Convert Categorical Variables into Numeric
model_df = pd.get_dummies(
    model_df,
    drop_first=True
)

# Define Input Features (X) and Target Variable (y)
X = model_df.drop('Churn_Yes', axis=1)
y = model_df['Churn_Yes']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Verify Shapes
print("\nModeling Data Prepared Successfully!")

print(f"\nX_train Shape: {X_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_test Shape: {y_test.shape}")

# =========================================================
# SECTION 5: LOGISTIC REGRESSION MODEL
# =========================================================

# Initialize Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)

# Train Model
log_model.fit(X_train, y_train)

# Prediction
y_pred_log = log_model.predict(X_test)

# Accuracy Score
log_accuracy = accuracy_score(
    y_test,
    y_pred_log
)

print("\nLogistic Regression Accuracy:")
print(round(log_accuracy, 4))

# Confusion Matrix
cm_log = confusion_matrix(
    y_test,
    y_pred_log
)

print("\nLogistic Regression Confusion Matrix:")
print(cm_log)

# Classification Report
print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_log
    )
)

# =========================================================
# SECTION 6: RANDOM FOREST MODEL
# =========================================================

# Initialize Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
rf_model.fit(X_train, y_train)

# Prediction
y_pred_rf = rf_model.predict(X_test)

# Accuracy Score
rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("\nRandom Forest Accuracy:")
print(round(rf_accuracy, 4))

# Confusion Matrix
cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

print("\nRandom Forest Confusion Matrix:")
print(cm_rf)

# Classification Report
print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)

# =========================================================
# SECTION 7: RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

# Extract Feature Importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

# Sort Features by Importance
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

# Display Feature Importance
print("\nRandom Forest Feature Importance:")
print(importance_df)

# =========================================================
# SECTION 8: MODEL COMPARISON
# =========================================================

# Create Comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest'
    ],
    
    'Accuracy': [
        round(log_accuracy, 4),
        round(rf_accuracy, 4)
    ]
})

# Display Comparison Table
print("\nModel Comparison:")
print(comparison_df)

# Identify Better Model
best_model = comparison_df.loc[
    comparison_df['Accuracy'].idxmax()
]

print("\nBest Performing Model:")
print(best_model)