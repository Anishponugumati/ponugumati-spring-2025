from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import shap

# ============================================
# 1. LOAD DATA
# ============================================
# Replace with the path to your land-use dataset CSV
data_path = "Preprocessed_Land_Use_Dataset.csv"
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)

# ============================================
# 2. DEFINE FEATURES & TARGET
# ============================================
# Target is the "target" column; features are all other columns.
target_col = "target"
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col].astype(int)  # Ensure the target is integer type

# (Optional) Print the first few rows to verify
print("First few rows of features:\n", X.head())
print("Target distribution:\n", y.value_counts())

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================
# Stratified split to preserve class distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 4. HYPERPARAMETER TUNING USING RANDOMIZEDSEARCHCV
# ============================================
# Define a parameter grid tailored for our dataset.
param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.25, 0.05, 0.1],
    "min_child_weight": [10, 11, 12],
    "gamma": [0.6, 0.7, 0.8],
    "subsample": [0.5, 0.6, 0.7],
    "colsample_bytree": [0.5, 0.6, 0.7],
    "reg_lambda": [50, 60, 70]
}

# Initialize the XGBoost classifier.
# Using "hist" tree_method for speed and GPU or CPU compatibility.
xgb_model = XGBClassifier(eval_metric="mlogloss", tree_method="hist", random_state=42)

# Set up stratified 10-fold cross-validation.
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Set up the randomized search.
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv_strategy,
    verbose=1,
    n_jobs=10,
    random_state=42
)

# Perform the search
print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)

# ============================================
# 5. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================
final_model = XGBClassifier(
    **random_search.best_params_,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)
final_model.fit(X_train, y_train)

# ============================================
# 6. EVALUATE THE MODEL
# ============================================
# Predictions
y_train_pred = final_model.predict(X_train)
y_pred = final_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Balanced Accuracy:", balanced_acc)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation scores
cv_scores = cross_val_score(final_model, X, y, cv=cv_strategy, scoring="accuracy")
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

# ============================================
# 7. SAVE FINAL MODEL
# ============================================
joblib.dump(final_model, "XGBoost_land_use_model.pkl")
print("Final land-use model saved successfully!")

# ============================================
# 8. VISUALIZATION OF FEATURE IMPORTANCE & SHAP VALUES
# ============================================
# Feature Importance from XGBoost
importances = final_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_cols, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.show()

# SHAP Feature Importance
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
