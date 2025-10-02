# main.py

# ------------------------------
# Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("/Users/Prathyusha/Downloads/Fraud_Analysis_Dataset(in).csv")

# ------------------------------
# EDA
# ------------------------------
print("Dataset Shape:", df.shape)
print("Null values:\n", df.isnull().sum())
print("Fraud counts:\n", df["isFraud"].value_counts())

# Plot transaction types
df["type"].value_counts().plot(kind="bar", title="Transaction Types", color="skyblue")
plt.xlabel("Transaction Type")
plt.show()

# Boxplot: Amount vs Fraud
sns.boxplot(data=df[df["amount"] < 50000], x="isFraud", y="amount")
plt.title("Amount vs isFraud (Filtered under 50k)")
plt.show()

# Create new features
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

# Frauds over time
frauds_per_step = df[df["isFraud"] == 1]["step"].value_counts().sort_index()
plt.plot(frauds_per_step.index, frauds_per_step.values, label="Frauds per Step")
plt.xlabel("Step (Time)")
plt.ylabel("Number of Frauds")
plt.title("Frauds over Time")
plt.grid(True)
plt.show()

# Correlation matrix
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# ------------------------------
# Feature Engineering
# ------------------------------
categorical = ["type"]
numeric = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

y = df["isFraud"]
X = df.drop("isFraud", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder="drop"
)

# ------------------------------
# Models
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

results = {}
financials = {}

for name, clf in models.items():
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Save the trained pipeline
    filename = f"fraud_detection_{name.replace(' ', '_')}.pkl"
    joblib.dump(pipeline, filename)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.decision_function(X_test)
    
    # Metrics
    acc = pipeline.score(X_test, y_test) * 100
    roc = roc_auc_score(y_test, y_proba)
    
    print(f"\n===== {name} =====")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", acc, "%")
    print("ROC-AUC Score:", roc)
    
    results[name] = {"Accuracy": acc, "ROC-AUC": roc}
    
    # Financial impact
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fraud_loss = 100
    saved = tp * fraud_loss
    lost = fn * fraud_loss
    profit = saved - lost
    financials[name] = {"Saved": saved, "Lost": lost, "Profit": profit}

# ------------------------------
# Results & Financials
# ------------------------------
results_df = pd.DataFrame(results).T
financial_df = pd.DataFrame(financials).T
print("\nModel Performance:\n", results_df)
print("\nFinancial Impact Analysis:\n", financial_df)

# ------------------------------
# Visualization
# ------------------------------
# ROC Curves
plt.figure(figsize=(8, 6))
for name, clf in models.items():
    pipeline = joblib.load(f"fraud_detection_{name.replace(' ', '_')}.pkl")
    if hasattr(clf, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test,y_proba):.2f})")

plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Models")
plt.legend()
plt.show()

# Financial impact bar plot
financial_df.plot(kind="bar", figsize=(10,6))
plt.title("Financial Impact of Models")
plt.ylabel("Amount")
plt.show()
# ---- Add to the END of main.py (to export analysis for the app) ----
results_df.to_csv("model_performance.csv", index=True)
financial_df.to_csv("financial_impact.csv", index=True)
