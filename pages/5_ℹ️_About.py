import streamlit as st
import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import setup_page

setup_page("About the Platform")

st.markdown("<h1 class='animate-fade-in'>ℹ️ About ChurnRadar</h1>", unsafe_allow_html=True)
st.markdown("Details regarding the platform architecture, machine learning pipelines, methodologies, and technologies used.", unsafe_allow_html=True)

# ----------------- SYSTEM OVERVIEW -----------------
st.subheader("⚙️ System Architecture")
st.markdown("""
ChurnRadar is designed using a **modular, layered software architecture** to isolate concerns and ensure maximum scalability, 
maintainability, and code reusability. Below is an overview of the system boundaries and layers:
""")

st.markdown("""
- **1. Data Layer:** Reads, checks, and validates the raw input CSV (`customer_churn.csv`).
- **2. Preprocessing Layer:** Handles cleanups (imputations, maps) and builds an automated Scikit-Learn `ColumnTransformer` (standard scaling for numeric columns, and one-hot encoding for categorical variables).
- **3. Machine Learning Layer:** Handles pipeline definition, training, model evaluation benchmarking, and saves the best model artifacts.
- **4. Prediction Layer (Inference Engine):** Loads the saved model pipeline and scores real-time customer inputs. Generates local explanation drivers.
- **5. Visualizations Layer:** Custom chart generators powered by Plotly for metrics plotting.
- **6. Streamlit UI Layer:** Multi-page dashboard layouts using SaaS-styled components and responsive HTML/CSS structures.
- **7. Business Intelligence Layer:** Extracts dynamic customer insights and maps risk probabilities to CRM retention actions.
""")

# ----------------- PIPELINE WORKFLOW -----------------
st.subheader("🔄 Machine Learning Pipeline Workflow")
st.markdown("""
```mermaid
flowchart LR
    subgraph Preprocessing
        A[Whitespace Imputer] --> B[Standard Scaler]
        A --> C[One-Hot Encoder]
    end
    subgraph Model Training
        B --> D[Pipeline Fitting]
        C --> D
        D --> E[F1-Score Selection]
    end
    subgraph Artifact Storage
        E -->|Saved Pipeline| F[joblib best_model.joblib]
        E -->|Metrics| G[model_comparison.json]
    end
```
""", unsafe_allow_html=True)

# ----------------- TECH STACK -----------------
st.subheader("🛠️ Technology Stack")
col_tech1, col_tech2 = st.columns(2)

with col_tech1:
    st.markdown("""
    **Core Programming & Frameworks:**
    - **Python:** Language of implementation.
    - **Streamlit:** Lightweight web application framework for rapid front-end deployment.
    - **Scikit-Learn:** Core library for model training, preprocessing, cross-validation, and metric evaluation.
    """)
    
with col_tech2:
    st.markdown("""
    **Data Processing & Visualizations:**
    - **Pandas & NumPy:** In-memory dataset manipulation, cleansing, and linear algebra.
    - **Plotly:** Dynamic, interactive, mobile-responsive vector plotting.
    - **Joblib:** Serializing trained ML models and data pipelines for low-latency inference.
    """)

# ----------------- DEVELOPER INFO -----------------
st.subheader("👨‍💻 Portfolio & Showcase Info")
st.markdown("""
This platform is a showcase-ready application designed for college major projects, GitHub repositories, and developer interviews. 
It demonstrates a complete production-quality data science implementation:
- Full **CI/CD ready structure** (separate scripts, pipelines, configuration files).
- **Explainable AI (XAI)** integrations at the global and individual customer level.
- High-fidelity visual standards (Custom CSS card injections, curated color palettes, hover effects).
- **Business Intelligence integration** that converts predictions into dollar amounts and strategic CRM recommendations.
""")

st.markdown("""
<div class='recommendation-card' style='border-left-color: #0F766E;'>
    <h4>Project Showcase Information</h4>
    <p>Feel free to fork this project on GitHub and adapt it to your own resume or portfolio. Below are suggestions for showcase credentials:</p>
    <ul>
        <li><strong>GitHub:</strong> <a href='https://github.com' target='_blank'>github.com/your-username</a></li>
        <li><strong>LinkedIn:</strong> <a href='https://linkedin.com' target='_blank'>linkedin.com/in/your-profile</a></li>
        <li><strong>Email:</strong> developer@example.com</li>
    </ul>
</div>
""", unsafe_allow_html=True)
