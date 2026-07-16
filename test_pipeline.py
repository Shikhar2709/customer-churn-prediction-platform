import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import DATA_PATH, MODEL_PATH
from utils.data_processing import load_data, split_data
from utils.ml_pipeline import explain_prediction

def test_pipeline():
    print("=" * 60)
    print("RUNNING END-TO-END PIPELINE VALIDATION TESTS")
    print("=" * 60)
    
    # 1. Test data loading and cleaning
    print("Test 1: Data Loading & Cleanups...")
    assert os.path.exists(DATA_PATH), f"Dataset CSV does not exist at {DATA_PATH}"
    df = load_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame), "Loaded data must be a pandas DataFrame"
    assert df.shape[0] > 0, "Loaded DataFrame is empty"
    assert "customerID" in df.columns, "customerID must be present"
    assert "Churn" in df.columns, "Churn column must be present"
    assert df["Churn"].dtype in [np.int32, np.int64, int], "Churn must be integer encoded"
    assert not df["TotalCharges"].isnull().any(), "TotalCharges must not contain NaN values after imputation"
    assert df["SeniorCitizen"].isin(["Yes", "No"]).all(), "SeniorCitizen must be mapped to Yes/No"
    print("   [PASSED] Data loading, validation, and cleaning tests.")
    
    # 2. Test train-test splitting
    print("Test 2: Train/Test Splitting...")
    X_train, X_test, y_train, y_test = split_data(df)
    assert X_train.shape[0] + X_test.shape[0] == len(df), "Split sizes do not match total dataset size"
    assert "Churn" not in X_train.columns, "Target column must not be in features"
    assert "customerID" not in X_train.columns, "ID column must not be in features"
    print("   [PASSED] Split ratios and feature shapes validated.")
    
    # 3. Test model load and inference
    print("Test 3: Model Pipeline Deserialization & Prediction...")
    assert os.path.exists(MODEL_PATH), f"Model file does not exist at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    assert hasattr(pipeline, "predict"), "Loaded model must have a 'predict' method"
    
    # Create single dummy customer row matching the schema
    dummy_customer = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": ["No"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["Yes"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [45.85],
        "TotalCharges": [550.00]
    })
    
    pred = pipeline.predict(dummy_customer)
    prob = pipeline.predict_proba(dummy_customer)
    
    assert pred[0] in [0, 1], "Prediction must be binary 0 or 1"
    assert prob.shape == (1, 2), "Predictive probability output shape must be (1, 2)"
    assert 0.0 <= prob[0, 1] <= 1.0, "Probability must be between 0 and 1"
    print("   [PASSED] Preprocessing & classifier pipeline inference verified.")
    
    # 4. Test explainability engine
    print("Test 4: Explainability & local contribution check...")
    explanation = explain_prediction(pipeline, dummy_customer)
    assert "churn_probability" in explanation, "Explanation must contain Churn Probability"
    assert "prediction" in explanation, "Explanation must contain prediction label"
    assert "risk_factors" in explanation, "Explanation must list risk factors"
    assert "loyalty_factors" in explanation, "Explanation must list loyalty factors"
    print("   [PASSED] Local explanation logic works.")
    
    print("=" * 60)
    print("ALL PIPELINE VALIDATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
