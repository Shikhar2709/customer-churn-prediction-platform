import streamlit as st
import pandas as pd
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import setup_page, DATA_PATH, COLORS
from utils.insights import calculate_kpis

# Initialize page settings
setup_page("Home")

# Main Header
st.markdown("""
<div class='animate-fade-in'>
    <h1 style='text-align: center; margin-bottom: 0px;'>🎯 ChurnRadar</h1>
    <h3 style='text-align: center; color: #14B8A6; font-weight: 500; margin-top: 0px; margin-bottom: 2rem;'>
        Customer Churn Prediction & Business Analytics Platform
    </h3>
</div>
""", unsafe_allow_html=True)

# Main Banner/Hero Section
st.markdown("""
<div class='recommendation-card' style='border-left-color: #0F766E;'>
    <h4>Welcome to ChurnRadar</h4>
    <p>
        An enterprise-grade, end-to-end Machine Learning and Business Intelligence platform. 
        ChurnRadar enables subscription-based businesses to predict customer cancellation risk, analyze churn-driving behavior, 
        and obtain real-time, data-driven customer retention recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# Grid Layout: Left column is business problem & overview, Right is data summary
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("💼 The Business Problem")
    st.markdown("""
    Acquiring new customers is **5 to 25 times more expensive** than retaining existing ones. For subscription-based business models, 
    customer attrition—known as **churn**—directly impacts revenue stability and long-term growth.
    
    **ChurnRadar solves this by:**
    - **Identifying At-Risk Customers:** Predicting churn probabilities using state-of-the-art ensemble classifiers.
    - **Uncovering Churn Drivers:** Explaining predictions through global and local feature analysis (Explainable AI).
    - **Providing Strategic Interventions:** Recommending personalized retention steps based on billing, contract, and usage patterns.
    """)
    
    st.subheader("⚙️ System Architecture & Workflow")
    st.markdown("""
    ```mermaid
    graph TD
        A[Raw Customer Data] --> B[Data Validation & Cleaning]
        B --> C[Feature Engineering & Scale/Encode]
        C --> D[Model Training & Comparison]
        D -->|Best Model Pipeline| E[Joblib Model Storage]
        E --> F[Inference & Explanation Engine]
        F --> G[Strategic Business Recommendations]
        A --> H[Interactive BI Analytics Dashboard]
    ```
    """, unsafe_allow_html=True)

with col2:
    st.subheader("📊 Dataset Executive Summary")
    
    if os.path.exists(DATA_PATH):
        try:
            # Load raw data to show quick stats
            df = pd.read_csv(DATA_PATH)
            # Basic info
            total_cust = len(df)
            raw_churn = df['Churn'].value_counts(normalize=True)
            churn_pct = raw_churn.get('Yes', 0.0) * 100
            
            # Show summary metrics in nice styled blocks
            st.markdown(f"""
            <div class='kpi-card' style='margin-bottom: 1rem;'>
                <div class='kpi-card-label'>Total Historical Records</div>
                <div class='kpi-card-value'>{total_cust:,}</div>
                <div class='indicator-bar teal'></div>
            </div>
            <div class='kpi-card' style='margin-bottom: 1rem;'>
                <div class='kpi-card-label'>Historical Churn Rate</div>
                <div class='kpi-card-value' style='color: #EF4444;'>{churn_pct:.2f}%</div>
                <div class='indicator-bar red'></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Dataset Dimensions & Columns**")
            st.markdown(f"- **Total Features:** {df.shape[1] - 1} attributes")
            st.markdown(f"- **Numerical columns:** Tenure, Monthly Charges, Total Charges")
            st.markdown(f"- **Categorical columns:** Demographics (Gender, Partners), Services (Internet, Tech Support, Backup, Streaming), Contracts, and Billing options")
            
        except Exception as e:
            st.error(f"Error loading dataset stats: {e}")
    else:
        st.warning("Telco Customer Churn dataset not found in data/ folder. Please run Step 1 to copy the file.")

# How to navigate section
st.markdown("---")
st.subheader("🚀 Navigate the Platform")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    ##### [📊 Executive Dashboard](Dashboard)
    Explore interactive behavioral charts, filter customer groups, and download summary insights.
    """)
    
with nav_col2:
    st.markdown("""
    ##### [🔍 Interactive Data Explorer](Data_Explorer)
    Filter, sort, search, and export the entire dataset for manual analysis.
    """)
    
with nav_col3:
    st.markdown("""
    ##### [🔮 Customer Prediction](Prediction_Engine)
    Input profiles manually or score batches to check churn probability, explain results, and get actions.
    """)

st.markdown("""
<div style='text-align: center; margin-top: 3rem; color: #64748B; font-size: 0.8rem; border-top: 1px solid #E2E8F0; padding-top: 1.5rem;'>
    ChurnRadar Analytics Platform • Built with Python, Streamlit, Scikit-Learn, and Plotly
</div>
""", unsafe_allow_html=True)
