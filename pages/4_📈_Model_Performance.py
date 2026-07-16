import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import plotly.express as px

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import setup_page, METRICS_PATH, IMPORTANCE_PATH, COLORS, PLOTLY_TEMPLATE
from utils.viz import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_feature_importance
)

setup_page("Model Performance")

st.markdown("<h1 class='animate-fade-in'>📈 Machine Learning Model Performance</h1>", unsafe_allow_html=True)
st.markdown("Detailed model evaluation metrics, model-wide comparison, and global Explainable AI analysis.", unsafe_allow_html=True)

# Check if model outputs exist
if not os.path.exists(METRICS_PATH) or not os.path.exists(IMPORTANCE_PATH):
    st.error("🚨 Evaluation metrics not found. Please run the model training step to generate metrics first.")
    st.stop()

# Load evaluation data
try:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_payload = json.load(f)
        
    comparison_data = pd.DataFrame(metrics_payload["comparison"])
    detailed_data = metrics_payload["detailed"]
    best_model_name = metrics_payload["best_model_name"]
    
    with open(IMPORTANCE_PATH, "r", encoding="utf-8") as f:
        feature_importance_list = json.load(f)
    feature_imp_df = pd.DataFrame(feature_importance_list)
    
    perm_imp_path = os.path.join(os.path.dirname(METRICS_PATH), "permutation_importance.json")
    if os.path.exists(perm_imp_path):
        with open(perm_imp_path, "r", encoding="utf-8") as f:
            perm_importance_list = json.load(f)
        perm_imp_df = pd.DataFrame(perm_importance_list)
    else:
        perm_imp_df = None
        
except Exception as e:
    st.error(f"Error loading evaluation artifacts: {e}")
    st.stop()

# ----------------- BEST MODEL CALLOUT -----------------
st.markdown(f"""
<div class='alert-box alert-success animate-fade-in'>
    <strong>🏆 Selected Production Model: {best_model_name}</strong><br>
    The Random Forest classifier was automatically chosen for deployment because it achieved the highest F1-Score on the test set. 
    In customer churn applications, balancing Precision (avoiding false alarms) and Recall (catching all potential churners) is crucial, 
    and F1-Score acts as the optimal selection metric.
</div>
""", unsafe_allow_html=True)

# ----------------- COMPARISON TABLE -----------------
st.markdown("### 📊 Model Benchmark Summary")
st.markdown("Performance benchmarks across the 4 trained classifiers:")

# Format table for cleaner UI
formatted_comparison = comparison_data.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
    formatted_comparison[col] = formatted_comparison[col].apply(lambda x: f"{x*100:.2f}%")
formatted_comparison["Training Time (s)"] = formatted_comparison["Training Time (s)"].apply(lambda x: f"{x:.3f}s")
formatted_comparison["Prediction Time (s)"] = formatted_comparison["Prediction Time (s)"].apply(lambda x: f"{x:.4f}s")

# Display styled table
st.dataframe(formatted_comparison, use_container_width=True)

# ----------------- DETAILED MODEL EVALUATION DETAILS -----------------
st.markdown("---")
st.markdown("### 🔬 Interactive Model Inspector")
st.markdown("Select a classifier to inspect its detailed evaluation metrics, confusion matrix, and ROC/PR curves:")

selected_model = st.selectbox("Select Classifier to Inspect", list(detailed_data.keys()), index=list(detailed_data.keys()).index(best_model_name))

if selected_model in detailed_data:
    model_eval = detailed_data[selected_model]
    
    col_metrics, col_chart = st.columns([1, 2])
    
    with col_metrics:
        # Display key metrics for selected model
        selected_row = comparison_data[comparison_data["Model"] == selected_model].iloc[0]
        
        st.markdown(f"""
        <div class='kpi-card' style='margin-bottom: 0.75rem;'>
            <div class='kpi-card-label'>Accuracy</div>
            <div class='kpi-card-value'>{selected_row['Accuracy']*100:.2f}%</div>
        </div>
        <div class='kpi-card' style='margin-bottom: 0.75rem;'>
            <div class='kpi-card-label'>F1-Score</div>
            <div class='kpi-card-value'>{selected_row['F1-Score']*100:.2f}%</div>
        </div>
        <div class='kpi-card' style='margin-bottom: 0.75rem;'>
            <div class='kpi-card-label'>ROC-AUC</div>
            <div class='kpi-card-value'>{selected_row['ROC-AUC']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_chart:
        # Confusion matrix
        st.plotly_chart(plot_confusion_matrix(model_eval["confusion_matrix"]), use_container_width=True)
        
    # ROC and PR Curves side-by-side
    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.plotly_chart(plot_roc_curve(model_eval["y_test"], model_eval["predictions_proba"]), use_container_width=True)
    with col_pr:
        st.plotly_chart(plot_pr_curve(model_eval["y_test"], model_eval["predictions_proba"]), use_container_width=True)

# ----------------- EXPLAINABLE AI (XAI) GLOBAL -----------------
st.markdown("---")
st.markdown("### 🧠 Explainable AI (XAI) - Global Importance")
st.markdown("Global feature importances show what features drive the predictions on a macro level across the entire customer base.")

tab_imp1, tab_imp2 = st.tabs(["📊 Global Feature Importance (Best Model)", "🎯 Permutation Importance (Test Set)"])

with tab_imp1:
    st.markdown("""
    **Global Feature Importance** is extracted directly from the best model (Random Forest). It shows the relative importance 
    of the preprocessed features (including one-hot encoded categories). High-ranking features represent the most powerful splits in the forest.
    """)
    st.plotly_chart(plot_feature_importance(feature_imp_df), use_container_width=True)

with tab_imp2:
    if perm_imp_df is not None:
        st.markdown("""
        **Permutation Importance** is computed by shuffling individual features on the test set and measuring the drop in F1-Score. 
        It is computed on the **raw, original variables** before encoding, giving a highly intuitive, model-agnostic view of feature value.
        """)
        
        # Sort and plot Permutation Importance
        perm_imp_sorted = perm_imp_df.sort_values(by='Importance_Mean', ascending=True).copy()
        
        fig_perm = px.bar(
            perm_imp_sorted,
            x='Importance_Mean',
            y='Feature',
            orientation='h',
            color='Importance_Mean',
            color_continuous_scale='Mint',
            error_x='Importance_Std'
        )
        
        fig_perm.update_layout(
            title_text="Permutation Importance on Original Features",
            template=PLOTLY_TEMPLATE,
            height=450,
            margin=dict(t=50, b=50, l=150, r=20),
            xaxis_title="Drop in F1-Score Mean",
            yaxis_title="Raw Feature Name",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_perm, use_container_width=True)
    else:
        st.warning("Permutation importance data not found. Please train models again to generate this chart.")
