import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time

# Load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balance using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train models
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_res, y_train_res)

# Evaluate
print("=== Logistic Regression ===")
y_pred_lr = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))

print("\n=== XGBoost ===")
y_pred_xgb = xgb.predict(X_test)
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))

# ROC Curves
RocCurveDisplay.from_estimator(lr, X_test, y_test)
plt.title("Logistic Regression ROC Curve")
plt.figtext(0.5, -0.05, "Built by: Soham Suhas Gujar", wrap=True, horizontalalignment='center', fontsize=10)
plt.tight_layout()
plt.show()

RocCurveDisplay.from_estimator(xgb, X_test, y_test)
plt.title("XGBoost ROC Curve")
plt.show()

# Simulate Real-Time Detection
print("\nSimulating real-time fraud detection...")
for i in range(10):
    tx = X_test.iloc[i].values.reshape(1, -1)
    pred = xgb.predict(tx)
    print(f"Transaction {i+1}: {'FRAUD' if pred[0] == 1 else 'Legit'}")
    time.sleep(1)
