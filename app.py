
# Streamlit App for Churn/Defaulter Prediction (Thesis Demo)
import streamlit as st
import numpy as np
import pickle
import gzip
from custom_churn_model import ChurnClassifierWithThreshold, ExtraTreesQuantileEnsemble, ExtraTreeQuantileClassifier
import joblib

# Load compressed joblib model
loaded_model = joblib.load('churn_model_with_threshold_compressed_________.joblib')



st.title("Churn / Defaulter Prediction App (Thesis Demo)")
st.write("This app predicts whether a customer will default or not using a threshold of 0.3.")

st.header("Enter Customer Information:")

# Numerical Features
current_loan_amount = st.number_input("Current Loan Amount", value=10000.0)
credit_score = st.number_input("Credit Score", value=700.0)
annual_income = st.number_input("Annual Income", value=50000.0)
monthly_debt = st.number_input("Monthly Debt", value=500.0)
years_credit_history = st.number_input("Years of Credit History", value=5.0)
open_accounts = st.number_input("Number of Open Accounts", value=3.0)
credit_problems = st.number_input("Number of Credit Problems", value=0.0)
current_credit_balance = st.number_input("Current Credit Balance", value=8000.0)
max_open_credit = st.number_input("Maximum Open Credit", value=15000.0)
bankruptcies = st.number_input("Bankruptcies", value=0.0)
tax_liens = st.number_input("Tax Liens", value=0.0)
credit_score_missing = st.selectbox("Is Credit Score Missing?", [0, 1])

# Categorical Features
term = st.selectbox("Term", ['Short Term', 'Long Term'])
years_in_job = st.selectbox("Years in Current Job", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
home_ownership = st.selectbox("Home Ownership", ['Rent', 'Own Home', 'HaveMortgage', 'Home Mortgage'])
purpose = st.selectbox("Purpose", ['Business Loan', 'Buy House', 'Buy a Car', 'Debt Consolidation', 'Educational Expenses', 'Home Improvements', 'Medical Bills', 'Other', 'Take a Trip', 'major_purchase', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation', 'wedding'])

if st.button("Predict"):
    # One-hot encoding for categorical features
    term_encoded = [1 if term == cat else 0 for cat in ['Long Term', 'Short Term']]
    job_encoded = [1 if years_in_job == cat else 0 for cat in ['1 year', '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '< 1 year']]
    home_encoded = [1 if home_ownership == cat else 0 for cat in ['HaveMortgage', 'Home Mortgage', 'Own Home', 'Rent']]
    purpose_encoded = [1 if purpose == cat else 0 for cat in ['Business Loan', 'Buy House', 'Buy a Car', 'Debt Consolidation', 'Educational Expenses', 'Home Improvements', 'Medical Bills', 'Other', 'Take a Trip', 'major_purchase', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation', 'wedding']]

    # Final feature vector (order matters)
    input_data = np.array([
        current_loan_amount,
        credit_score,
        annual_income,
        monthly_debt,
        years_credit_history,
        open_accounts,
        credit_problems,
        current_credit_balance,
        max_open_credit,
        bankruptcies,
        tax_liens,
        credit_score_missing
    ] + term_encoded + job_encoded + home_encoded + purpose_encoded).reshape(1, -1)

    prediction = loaded_model.predict(input_data)[0]
    prob = loaded_model.predict_proba(input_data)[:, 1][0]

    st.subheader("Prediction Result:")
    st.write("**Prediction:**", "Defaulter (1)" if prediction == 1 else "Non-Defaulter (0)")
    st.write(f"**Defaulter Probability:** {prob:.3f} (Threshold: 0.3)")
