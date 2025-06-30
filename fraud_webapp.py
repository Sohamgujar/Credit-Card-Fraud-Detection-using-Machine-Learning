import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("💳 Credit Card Fraud Detection App")
st.subheader("By Soham Suhas Gujar")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df

df = load_data()
st.write("### Sample Data", df.head())

# Check class balance
st.write("### Class Distribution")
st.bar_chart(df["Class"].value_counts())

# Preprocessing
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test)

# Evaluation
st.write("### Model Evaluation")
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
st.success(f"ROC-AUC Score: {roc:.4f}")

# ROC Curve
from sklearn.metrics import RocCurveDisplay
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
plt.title("Logistic Regression ROC Curve\nBuilt by Soham Suhas Gujar")
st.pyplot(fig)

# Real-time simulation
st.write("### Simulated Real-Time Detection")
num_tx = st.slider("How many transactions to simulate?", 1, 20, 5)
for i in range(num_tx):
    tx = X_test.iloc[i].values.reshape(1, -1)
    pred = model.predict(tx)[0]
    st.write(f"Transaction {i+1}: {'🚨 FRAUD' if pred == 1 else '✅ Legit'}")
