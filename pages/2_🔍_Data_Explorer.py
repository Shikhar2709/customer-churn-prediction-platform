import streamlit as st
import pandas as pd
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import setup_page, DATA_PATH
from utils.data_processing import load_data

setup_page("Data Explorer")

st.markdown("<h1 class='animate-fade-in'>🔍 Interactive Data Explorer</h1>", unsafe_allow_html=True)
st.markdown("Search, sort, filter, and inspect individual customer records. Download custom cohorts directly as CSV.", unsafe_allow_html=True)

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please run the model training step to initialize.")
    st.stop()

# Load clean data
try:
    df_raw = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Make a copy for filtering
df = df_raw.copy()

# Add a text search for Customer ID
st.markdown("### 🔍 Quick Search & Filter Panel")
col_search, col_churn = st.columns([3, 1])

with col_search:
    search_id = st.text_input("Search Customer ID", "", placeholder="e.g. 7590-VHVEG")
    if search_id:
        df = df[df['customerID'].str.contains(search_id.strip(), case=False, na=False)]

with col_churn:
    churn_filter = st.selectbox("Churn Status", ["All", "Active Customers (No)", "Churned Customers (Yes)"])
    if churn_filter == "Active Customers (No)":
        df = df[df['Churn'] == 0]
    elif churn_filter == "Churned Customers (Yes)":
        df = df[df['Churn'] == 1]

# Expandable Advanced Filters Grid
with st.expander("🛠️ Advanced Demographic & Service Filters", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender = st.selectbox("Gender Filter", ["All"] + list(df_raw['gender'].unique()), key="de_gender")
        if gender != "All":
            df = df[df['gender'] == gender]
            
        partner = st.selectbox("Partner", ["All"] + list(df_raw['Partner'].unique()), key="de_partner")
        if partner != "All":
            df = df[df['Partner'] == partner]

    with col2:
        contract = st.selectbox("Contract Type Filter", ["All"] + list(df_raw['Contract'].unique()), key="de_contract")
        if contract != "All":
            df = df[df['Contract'] == contract]
            
        dependents = st.selectbox("Dependents", ["All"] + list(df_raw['Dependents'].unique()), key="de_dependents")
        if dependents != "All":
            df = df[df['Dependents'] == dependents]

    with col3:
        internet = st.selectbox("Internet Service Filter", ["All"] + list(df_raw['InternetService'].unique()), key="de_internet")
        if internet != "All":
            df = df[df['InternetService'] == internet]
            
        tech_support = st.selectbox("Tech Support", ["All"] + list(df_raw['TechSupport'].unique()), key="de_support")
        if tech_support != "All":
            df = df[df['TechSupport'] == tech_support]

    with col4:
        payment = st.selectbox("Payment Method Filter", ["All"] + list(df_raw['PaymentMethod'].unique()), key="de_payment")
        if payment != "All":
            df = df[df['PaymentMethod'] == payment]
            
        paperless = st.selectbox("Paperless Billing", ["All"] + list(df_raw['PaperlessBilling'].unique()), key="de_paperless")
        if paperless != "All":
            df = df[df['PaperlessBilling'] == paperless]

    # Sliders inside expander
    col_slide1, col_slide2 = st.columns(2)
    with col_slide1:
        tenure_vals = st.slider("Tenure Range (Months)", int(df_raw['tenure'].min()), int(df_raw['tenure'].max()), (int(df_raw['tenure'].min()), int(df_raw['tenure'].max())), key="de_tenure")
        df = df[(df['tenure'] >= tenure_vals[0]) & (df['tenure'] <= tenure_vals[1])]
    with col_slide2:
        charge_vals = st.slider("Monthly Charges Range ($)", float(df_raw['MonthlyCharges'].min()), float(df_raw['MonthlyCharges'].max()), (float(df_raw['MonthlyCharges'].min()), float(df_raw['MonthlyCharges'].max())), key="de_charges")
        df = df[(df['MonthlyCharges'] >= charge_vals[0]) & (df['MonthlyCharges'] <= charge_vals[1])]

# Show status
total_records = len(df_raw)
filtered_records = len(df)
st.markdown(f"**Showing {filtered_records:,} of {total_records:,} customer records**")

# Display dataframe with custom height and full width
if not df.empty:
    # Map Churn back to Yes/No for cleaner viewing in the explorer
    view_df = df.copy()
    view_df['Churn'] = view_df['Churn'].map({1: 'Yes', 0: 'No'})
    
    st.dataframe(
        view_df,
        use_container_width=True,
        height=500
    )
    
    # Download filtered CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current Filtered Table as CSV",
        data=csv_data,
        file_name="customer_churn_filtered_explorer.csv",
        mime="text/csv",
        key="explorer-download-csv"
    )
else:
    st.warning("⚠️ No records match the current filters. Please adjust the options above.")
