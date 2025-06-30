# 💳 Credit Card Fraud Detection using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen">
</p>

Detecting fraudulent credit card transactions using Logistic Regression, SMOTE balancing, and performance evaluation techniques. Includes an interactive web app built using Streamlit.

---

## 🚀 Project Overview



This project aims to identify **fraudulent transactions** in credit card data. Due to the rarity of fraud cases, the dataset is highly imbalanced — a challenge tackled with **SMOTE** (Synthetic Minority Oversampling Technique). The model is deployed as a **web app** using **Streamlit** for easy interaction and real-time testing.

---

## 📂 Folder Structure

CreditCardFraudDetection/
│
├── data/
│ └── creditcard.csv # Dataset from Kaggle
├── fraud_detection.py # CLI model training and evaluation
├── fraud_webapp.py # Streamlit interactive web app
├── requirements.txt # All dependencies
└── README.md # Project documentation (this file)


---

## 🧠 Key Features

- ✅ Data preprocessing and cleaning
- ⚖️ Imbalanced data handling using SMOTE
- 📈 Model training using Logistic Regression
- 📊 Performance metrics: Confusion Matrix, F1-Score, ROC-AUC
- 🧪 Real-time transaction fraud simulation
- 🌐 Interactive web app with user controls (Streamlit)

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: V1–V28 (PCA anonymized), `Time`, `Amount`, and `Class`

> Class labels:  
> - `0` = Legit  
> - `1` = Fraud

---

## 📈 Model Performance (Logistic Regression)

| Metric          | Value     |
|-----------------|-----------|
| Precision       | 0.76      |
| Recall          | 0.86      |
| F1-Score        | 0.81      |
| ROC-AUC Score   | **0.9833** ✅ |

---

## 🖥️ How to Run Locally

### 📌 Clone the Repo

```bash
git clone https://github.com/yourusername/CreditCardFraudDetection.git
cd CreditCardFraudDetection

for requirment 
pip install -r requirements.txt

for streamlit 
streamlit run fraud_webapp.py


