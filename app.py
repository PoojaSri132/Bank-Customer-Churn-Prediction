import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("bank_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.write("### Model Used")
st.sidebar.success("Random Forest Classifier")

st.sidebar.write("### Performance")

st.sidebar.write("Accuracy : **84.05%**")
st.sidebar.write("Precision : **59.56%**")
st.sidebar.write("Recall : **67.32%**")
st.sidebar.write("F1 Score : **63.21%**")
st.sidebar.write("ROC-AUC : **86.51%**")

st.sidebar.markdown("---")
st.sidebar.write("Developed using")
st.sidebar.write("- Streamlit")
st.sidebar.write("- Scikit-learn")
st.sidebar.write("- Python")

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "Predict whether a bank customer is likely to churn using a trained **Random Forest Classifier**."
)

st.markdown("---")

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

with col2:

    products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    credit_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    active = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

# -----------------------------
# Encoding
# -----------------------------
gender = 1 if gender == "Male" else 0

germany = 1 if geography == "Germany" else 0
spain = 1 if geography == "Spain" else 0

# -----------------------------
# Feature Engineering
# -----------------------------
balance_salary_ratio = balance / (salary + 1)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({

    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [products],
    'HasCrCard': [credit_card],
    'IsActiveMember': [active],
    'EstimatedSalary': [salary],
    'BalanceSalaryRatio': [balance_salary_ratio],
    'Geography_Germany': [germany],
    'Geography_Spain': [spain]

})

# -----------------------------
# Scale Continuous Features
# -----------------------------
continuous_features = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'EstimatedSalary'
]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    input_data[continuous_features] = scaler.transform(
        input_data[continuous_features]
    )

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    churn_probability = probability[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.subheader("Churn Probability")

    st.progress(float(churn_probability))

    st.write(f"### {churn_probability*100:.2f}%")

    st.subheader("Risk Level")

    if churn_probability >= 0.80:
        st.error("🔴 High Risk Customer")

    elif churn_probability >= 0.60:
        st.warning("🟡 Moderate Risk Customer")

    else:
        st.success("🟢 Low Risk Customer")

    st.markdown("---")

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Prediction",
            "Probability",
            "Risk Level",
            "Model"
        ],
        "Value": [
            "Churn" if prediction[0] == 1 else "Stay",
            f"{churn_probability*100:.2f}%",
            "High" if churn_probability >= 0.80 else
            "Moderate" if churn_probability >= 0.60 else
            "Low",
            "Random Forest"
        ]
    })

    st.table(summary)