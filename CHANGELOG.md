# Changelog

All notable changes to the ChurnRadar project will be documented in this file.

## [1.0.0] - 2026-07-16

### Added
- **Core Architecture:** Set up modular, layered package structure (Data, Preprocessing, ML, UI, Viz, Insights).
- **Data Preprocessing Layer:** Implemented missing value imputation for `TotalCharges`, duplicate removal, and categorical transformations.
- **ML Pipeline:** Integrated training, comparison, and evaluation for 4 classifiers (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting).
- **Explainable AI (XAI):** Added global Feature Importance plots, Permutation Importance analysis, and individual customer risk factor contribution charts.
- **Interactive UI:** Developed multi-page Streamlit dashboard featuring custom CSS KPI cards, sidebar controls, data explorer tables, and live prediction forms.
- **Business Intelligence Layer:** Integrated automatic retention suggestions and high-risk customer segments alerts.
- **Testing:** Implemented automated pipeline verification script (`test_pipeline.py`).
