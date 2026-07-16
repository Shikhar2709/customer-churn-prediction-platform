import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import setup_page, MODEL_PATH, COLORS
from utils.ml_pipeline import explain_prediction
from utils.insights import get_churn_risk_level

setup_page("Customer Churn Prediction Engine")

st.markdown("<h1 class='animate-fade-in'>🔮 Customer Churn Prediction Engine</h1>", unsafe_allow_html=True)
st.markdown("Enter customer details below to predict their churn risk and view customized retention actions.", unsafe_allow_html=True)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error("🚨 Saved model not found. Please run the model training step first to train and save the best model.")
    st.stop()

# Load model pipeline
try:
    pipeline = joblib.load(MODEL_PATH)
    # Extract model name if possible
    classifier_name = type(pipeline.named_steps['classifier']).__name__
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Form layout
st.markdown("### 📋 Customer Profile Input Form")

with st.form("prediction_form"):
    # Group inputs into columns for clean layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👥 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        
    with col2:
        st.subheader("💳 Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        tenure = st.slider("Tenure (Months)", 0, 72, 12, help="Number of months customer has stayed with company")
        
    with col3:
        st.subheader("🔌 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        
        # Conditional multiple lines
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            multiple_lines = "No phone service"
            st.info("Multiple Lines disabled (No Phone Service)")
            
        internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])
        
    # Divider for Internet Add-ons
    st.markdown("---")
    st.write("**Internet Add-ons & Features**")
    
    col4, col5, col6 = st.columns(3)
    
    if internet_service != "No":
        with col4:
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        with col5:
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        with col6:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        # Default value for customers without internet service
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        st.info("ℹ️ Internet add-ons disabled (No Internet Service)")
        
    # Charges Section
    st.markdown("---")
    col7, col8 = st.columns(2)
    with col7:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", 
            min_value=15.0, 
            max_value=150.0, 
            value=65.0, 
            step=0.5,
            help="Monthly subscription amount charged"
        )
    with col8:
        # Smart Auto-Calculate total charges
        suggested_total = round(monthly_charges * tenure, 2)
        total_charges = st.number_input(
            "Total Charges ($)", 
            min_value=0.0, 
            max_value=10000.0, 
            value=suggested_total,
            step=5.0,
            help="Total cumulative amount charged. Defaults to (Monthly Charges * Tenure)"
        )
        
    # Form Submit Button
    submit_button = st.form_submit_button(label="🔮 Predict Churn Risk")

# ----------------- INFERENCE & EXPLAINABLE AI -----------------
if submit_button:
    # 1. Package inputs into single-row DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })
    
    # Run prediction & explanation
    try:
        # Load explainer results
        explanation = explain_prediction(pipeline, input_data)
        prob = explanation["churn_probability"]
        pred_label = explanation["prediction"]
        risk_factors = explanation["risk_factors"]
        loyalty_factors = explanation["loyalty_factors"]
        
        # Get strategic details
        risk_level, risk_color, recommendation = get_churn_risk_level(prob)
        
        # Visual display banner for Churn / Loyal
        st.markdown("---")
        st.markdown("### 📊 Prediction Result")
        
        if pred_label == "Churn":
            st.markdown(f"""
            <div class='prediction-box churn animate-fade-in'>
                <h2 style='color: white; margin-bottom: 5px;'>⚠️ Customer Churn Predicted</h2>
                <div style='font-size: 1.5rem; font-weight: 700; margin-bottom: 10px;'>
                    Probability: {prob*100:.1f}% ({risk_level})
                </div>
                <p style='margin: 0px;'>Active retention action is highly recommended. Model: {classifier_name}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='prediction-box no-churn animate-fade-in'>
                <h2 style='color: white; margin-bottom: 5px;'>✅ Customer Will Remain Loyal</h2>
                <div style='font-size: 1.5rem; font-weight: 700; margin-bottom: 10px;'>
                    Probability of Churn: {prob*100:.1f}% ({risk_level})
                </div>
                <p style='margin: 0px;'>Customer shows high stability indicators. Model: {classifier_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Probability Gauge using Streamlit Progress Bar
        st.write("**Churn Risk Scale**")
        st.progress(prob)
        
        # Explainable AI: Drivers column split
        st.markdown("### 🧠 Explainable AI (XAI) - Local Driver Analysis")
        st.markdown("Analysis of which factors pushed this customer's risk profile towards Churn vs. Retention:")
        
        col_risk, col_loyalty = st.columns(2)
        
        with col_risk:
            st.markdown("<h4 style='color: #EF4444;'>🚨 Top Risk Factors (Pushes to Churn)</h4>", unsafe_allow_html=True)
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"""
                    <div class='kpi-card' style='margin-bottom: 0.75rem; border-left: 4px solid #EF4444;'>
                        <strong>{factor['factor']}</strong> <span style='font-size: 0.75rem; background-color: #FEE2E2; color: #991B1B; padding: 2px 6px; border-radius: 4px; font-weight:600;'>{factor['impact']} Impact</span>
                        <div style='font-size: 0.85rem; color: #64748B; margin-top: 4px;'>{factor['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No major risk factors detected.")
                
        with col_loyalty:
            st.markdown("<h4 style='color: #10B981;'>🛡️ Top Retention Factors (Pushes to Loyalty)</h4>", unsafe_allow_html=True)
            if loyalty_factors:
                for factor in loyalty_factors:
                    st.markdown(f"""
                    <div class='kpi-card' style='margin-bottom: 0.75rem; border-left: 4px solid #10B981;'>
                        <strong>{factor['factor']}</strong> <span style='font-size: 0.75rem; background-color: #D1FAE5; color: #065F46; padding: 2px 6px; border-radius: 4px; font-weight:600;'>{factor['impact']} Impact</span>
                        <div style='font-size: 0.85rem; color: #64748B; margin-top: 4px;'>{factor['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No major loyalty factors detected.")
                
        # Business Recommendations
        st.markdown("### 🎯 Strategic Retention Recommendation")
        st.markdown(f"""
        <div class='recommendation-card' style='border-left-color: {risk_color};'>
            <h4>Action Plan for Customer Support / CRM:</h4>
            <p>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Inference error: {e}")
        st.exception(e)
