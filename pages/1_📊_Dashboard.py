import streamlit as st
import pandas as pd
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import setup_page, DATA_PATH, COLORS
from utils.data_processing import load_data
from utils.insights import calculate_kpis, get_segment_insights
from utils.viz import (
    plot_churn_distribution,
    plot_categorical_churn,
    plot_numerical_distribution,
    plot_correlation_heatmap
)

# Setup page config and load CSS
setup_page("Executive Dashboard")

# Page title
st.markdown("<h1 class='animate-fade-in'>📊 Business Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Analyze customer behaviors, filter customer groups, and identify churn factors.", unsafe_allow_html=True)

# Check if data exists
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please run the model training step first to initialize data.")
    st.stop()

# Load clean data
try:
    df_raw = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("🎯 Filters Panel")

# 1. Tenure Slider
min_tenure, max_tenure = int(df_raw['tenure'].min()), int(df_raw['tenure'].max())
tenure_range = st.sidebar.slider("Tenure (Months)", min_tenure, max_tenure, (min_tenure, max_tenure))

# 2. Monthly Charges Slider
min_mc, max_mc = float(df_raw['MonthlyCharges'].min()), float(df_raw['MonthlyCharges'].max())
mc_range = st.sidebar.slider("Monthly Charges ($)", min_mc, max_mc, (min_mc, max_mc))

# 3. Categorical Filters
gender_opt = ["All"] + list(df_raw['gender'].unique())
selected_gender = st.sidebar.selectbox("Gender", gender_opt)

contract_opt = ["All"] + list(df_raw['Contract'].unique())
selected_contract = st.sidebar.selectbox("Contract Type", contract_opt)

internet_opt = ["All"] + list(df_raw['InternetService'].unique())
selected_internet = st.sidebar.selectbox("Internet Service", internet_opt)

payment_opt = ["All"] + list(df_raw['PaymentMethod'].unique())
selected_payment = st.sidebar.selectbox("Payment Method", payment_opt)

# Apply Filters
df = df_raw.copy()
df = df[(df['tenure'] >= tenure_range[0]) & (df['tenure'] <= tenure_range[1])]
df = df[(df['MonthlyCharges'] >= mc_range[0]) & (df['MonthlyCharges'] <= mc_range[1])]

if selected_gender != "All":
    df = df[df['gender'] == selected_gender]
if selected_contract != "All":
    df = df[df['Contract'] == selected_contract]
if selected_internet != "All":
    df = df[df['InternetService'] == selected_internet]
if selected_payment != "All":
    df = df[df['PaymentMethod'] == selected_payment]

# Calculate KPIs (Filtered vs Overall)
kpis = calculate_kpis(df)
kpis_overall = calculate_kpis(df_raw)

# ----------------- KPI CARDS -----------------
st.markdown("<h3 style='margin-top: 1.5rem;'>📈 Executive KPIs</h3>", unsafe_allow_html=True)

# HTML structure for modern KPI cards
st.markdown(f"""
<div class='kpi-container'>
    <div class='kpi-card'>
        <div class='kpi-card-label'>Total Customers</div>
        <div class='kpi-card-value'>{kpis['total_customers']:,}</div>
        <div class='kpi-card-delta'>
            <span>of {kpis_overall['total_customers']:,} total baseline</span>
        </div>
        <div class='indicator-bar teal'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-card-label'>Active Customers</div>
        <div class='kpi-card-value'>{kpis['active_customers']:,}</div>
        <div class='kpi-card-delta'>
            <span>{(kpis['active_customers']/(kpis['total_customers'] if kpis['total_customers'] > 0 else 1))*100:.1f}% active share</span>
        </div>
        <div class='indicator-bar green'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-card-label'>Churned Customers</div>
        <div class='kpi-card-value'>{kpis['churned_customers']:,}</div>
        <div class='kpi-card-delta'>
            <span>{(kpis['churned_customers']/(kpis['total_customers'] if kpis['total_customers'] > 0 else 1))*100:.1f}% churned share</span>
        </div>
        <div class='indicator-bar red'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-card-label'>Churn Rate</div>
        <div class='kpi-card-value' style='color: {"#EF4444" if kpis["churn_rate"] > kpis_overall["churn_rate"] else "#10B981"};'>
            {kpis['churn_rate']:.2f}%
        </div>
        <div class='kpi-card-delta'>
            <span class='{"delta-up" if kpis["churn_rate"] > kpis_overall["churn_rate"] else "delta-down"}'>
                {"+" if kpis["churn_rate"] > kpis_overall["churn_rate"] else ""}{kpis["churn_rate"] - kpis_overall["churn_rate"]:.2f}% vs baseline ({kpis_overall["churn_rate"]:.2f}%)
            </span>
        </div>
        <div class='indicator-bar red'></div>
    </div>
</div>
<div class='kpi-container'>
    <div class='kpi-card'>
        <div class='kpi-card-label'>Avg Monthly Charges</div>
        <div class='kpi-card-value'>${kpis['avg_monthly_charges']:.2f}</div>
        <div class='kpi-card-delta'>
            <span>baseline: ${kpis_overall['avg_monthly_charges']:.2f}</span>
        </div>
        <div class='indicator-bar teal'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-card-label'>Avg Tenure</div>
        <div class='kpi-card-value'>{kpis['avg_tenure']:.1f} mo</div>
        <div class='kpi-card-delta'>
            <span>baseline: {kpis_overall['avg_tenure']:.1f} mo</span>
        </div>
        <div class='indicator-bar green'></div>
    </div>
    <div class='kpi-card'>
        <div class='kpi-card-label'>Estimated Value (LTV)</div>
        <div class='kpi-card-value'>${kpis['total_revenue']:,.2f}</div>
        <div class='kpi-card-delta'>
            <span>Total Charges sum</span>
        </div>
        <div class='indicator-bar amber'></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Check for empty dataframe
if df.empty:
    st.warning("⚠️ No customer records match the selected filters. Please expand your filter selections in the sidebar.")
    st.stop()

# ----------------- INSIGHTS & STRATEGIC RECOMMENDATIONS -----------------
st.markdown("---")
st.subheader("💡 Automated Business Insights & Recommendations")

insights = get_segment_insights(df)

if insights:
    # Render segment insights as warning alerts and action recommendations
    for ins in insights:
        # Determine color block based on severity
        severity_class = "alert-warning" if ins['severity'] in ["HIGH", "CRITICAL"] else "alert-info"
        st.markdown(f"""
        <div class='alert-box {severity_class}'>
            <strong>🚨 High-Risk Segment Detected: {ins['segment']}</strong><br>
            <em>Finding:</em> {ins['finding']}<br>
            <div class='recommendation-card' style='margin-top: 10px; margin-bottom: 0px;'>
                <strong>🎯 Retention Action:</strong> {ins['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.success("🎉 Excellent! Churn indicators in this filtered customer segment are below standard risk thresholds.")

# ----------------- VISUALIZATIONS GRID -----------------
st.markdown("---")
st.subheader("📊 Interactive Visualizations")

tab1, tab2, tab3 = st.tabs(["⚡ Core Churn Drivers", "👥 Customer Demographics", "📈 Financial & Correlation"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_churn_distribution(df), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_categorical_churn(df, "Contract"), use_container_width=True)
        
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_categorical_churn(df, "InternetService"), use_container_width=True)
    with col_d:
        st.plotly_chart(plot_categorical_churn(df, "TechSupport"), use_container_width=True)

with tab2:
    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(plot_categorical_churn(df, "gender"), use_container_width=True)
    with col_f:
        st.plotly_chart(plot_categorical_churn(df, "SeniorCitizen"), use_container_width=True)
        
    col_g, col_h = st.columns(2)
    with col_g:
        st.plotly_chart(plot_categorical_churn(df, "Partner"), use_container_width=True)
    with col_h:
        st.plotly_chart(plot_categorical_churn(df, "PaymentMethod"), use_container_width=True)

with tab3:
    col_i, col_j = st.columns(2)
    with col_i:
        st.plotly_chart(plot_numerical_distribution(df, "tenure"), use_container_width=True)
    with col_j:
        st.plotly_chart(plot_numerical_distribution(df, "MonthlyCharges"), use_container_width=True)
        
    # Heatmap row
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

# ----------------- DATA DOWNLOAD SECTION -----------------
st.markdown("---")
st.subheader("📥 Export Dashboard Data")
st.markdown("Export the filtered customer records currently displayed in the dashboard metrics above.")

csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Records as CSV",
    data=csv_data,
    file_name="filtered_churn_data.csv",
    mime="text/csv",
    key="dashboard-download-csv"
)
