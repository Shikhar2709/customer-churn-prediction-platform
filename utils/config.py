import os
import streamlit as st

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_comparison.json")
IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")
CSS_PATH = os.path.join(BASE_DIR, "assets", "custom.css")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Data Schema Columns
TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Color Palette (Teal and Slate theme)
COLORS = {
    "primary": "#0F766E",       # Dark Teal
    "secondary": "#14B8A6",     # Light Teal
    "accent": "#F59E0B",        # Amber/Yellow
    "churn": "#EF4444",         # Red for Churn
    "active": "#10B981",        # Green for Active
    "bg_dark": "#0F172A",       # Slate 900
    "card_dark": "#1E293B",     # Slate 800
    "text_dark": "#F1F5F9",     # Slate 100
    "bg_light": "#F8FAFC",      # Slate 50
    "card_light": "#FFFFFF",    # White
    "text_light": "#0F172A",    # Slate 900
    "muted": "#64748B",         # Slate 500
}

# Plotly Theme Settings
PLOTLY_TEMPLATE = "plotly_white"

def inject_custom_css():
    """Reads custom.css and injects it into the Streamlit app page."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS file not found. Falling back to default Streamlit styling.")

def setup_page(title: str, layout: str = "wide"):
    """Standardizes page configuration and loads styling."""
    st.set_page_config(
        page_title=f"{title} | ChurnRadar",
        page_icon="🎯",
        layout=layout,
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
