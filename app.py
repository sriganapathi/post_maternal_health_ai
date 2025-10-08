# ==============================
# Streamlit App - Post-Maternal Health Risk Prediction
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Post-Maternal Health Monitoring AI", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("maternal_health_risk_full.csv")

st.title("🤖 AI-Driven Post-Maternal Health Monitoring System")
st.write("This system predicts postpartum health risk using AI models.")

# -----------------------------
# Show Correlation Heatmap
# -----------------------------
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("🔍 Feature Correlation Heatmap")
    plt.figure(figsize=(8,6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
    st.pyplot(plt)

# -----------------------------
# Data Preparation
# -----------------------------
data_clean = data.drop_duplicates().dropna()
encoder = LabelEncoder()
data_clean['RiskLevel'] = encoder.fit_transform(data_clean['RiskLevel'])

X = data_clean.drop('RiskLevel', axis=1)
y = data_clean['RiskLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Models
# -----------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

# -----------------------------
# Show Accuracy Comparison
# -----------------------------
st.subheader("📊 Model Accuracy Comparison")
fig, ax = plt.subplots()
ax.barh(list(accuracies.keys()), list(accuracies.values()), color=['skyblue', 'orange', 'lightgreen'])
ax.set_xlabel("Accuracy")
for i, v in enumerate(accuracies.values()):
    ax.text(v, i, f"{v*100:.2f}%", va='center', fontsize=10)
st.pyplot(fig)

st.write(f"🏆 Best Performing Model: **{best_model_name}** with Accuracy {accuracies[best_model_name]*100:.2f}%")

# -----------------------------
# User Input for Prediction
# -----------------------------
st.subheader("🩺 Enter Patient Details for Risk Prediction")
with st.form(key='risk_form'):
    Age = st.slider("Age", 18, 45, 30)
    SystolicBP = st.slider("Systolic BP", 90, 180, 120)
    DiastolicBP = st.slider("Diastolic BP", 50, 110, 80)
    BS = st.slider("Blood Sugar", 60, 180, 100)
    BodyTemp = st.slider("Body Temperature (°F)", 97.0, 104.0, 99.0, 0.1)
    HeartRate = st.slider("Heart Rate", 60, 130, 85)
    submit_button = st.form_submit_button(label='Predict Risk')

if submit_button:
    user_data = pd.DataFrame([[Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]],
                             columns=X.columns)
    user_scaled = scaler.transform(user_data)
    prediction = best_model.predict(user_scaled)
    risk = encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Post-Maternal Health Risk Level: **{risk.upper()}**")
