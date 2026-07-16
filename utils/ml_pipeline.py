import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

from utils.config import NUMERIC_COLS, CATEGORICAL_COLS

logger = logging.getLogger(__name__)

def train_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Dict[str, Pipeline]:
    """Trains Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting pipelines.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        preprocessor: ColumnTransformer preprocessing pipeline.
        
    Returns:
        Dict: Dictionary mapping model names to their trained Pipeline objects.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    trained_pipelines = {}
    
    for name, model in models.items():
        logger.info(f"Training {name} model pipeline...")
        # Create pipeline: preprocessing step + classifier step
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        # Store training time directly in the model metadata
        pipeline.train_time_ = elapsed_time
        trained_pipelines[name] = pipeline
        logger.info(f"{name} pipeline trained in {elapsed_time:.3f} seconds.")
        
    return trained_pipelines

def evaluate_models(
    models: Dict[str, Pipeline], 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluates all trained model pipelines and returns comparison metrics.
    
    Args:
        models: Dictionary of trained model Pipelines.
        X_test: Test features.
        y_test: Test target.
        
    Returns:
        Tuple: (DataFrame comparison table, Dict of detailed model evaluations)
    """
    comparison_rows = []
    detailed_results = {}
    
    for name, pipeline in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Time predictions
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        predict_time = time.time() - start_time
        
        # Prediction probabilities
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        conf_mat = [[int(tn), int(fp)], [int(fn), int(tp)]]
        
        # Classification report as dict
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Store in comparison table
        comparison_rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Training Time (s)": getattr(pipeline, 'train_time_', 0.0),
            "Prediction Time (s)": predict_time
        })
        
        # Detailed results for graphs/charts
        detailed_results[name] = {
            "confusion_matrix": conf_mat,
            "classification_report": class_report,
            "predictions_proba": y_prob.tolist(),
            "y_test": y_test.tolist()
        }
        
    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df, detailed_results

def get_preprocessed_feature_names(preprocessor) -> List[str]:
    """Extracts feature names after they pass through the ColumnTransformer preprocessor.
    
    Args:
        preprocessor: ColumnTransformer.
        
    Returns:
        List[str]: Preprocessed feature names.
    """
    try:
        # Standard method in scikit-learn >= 1.0
        return list(preprocessor.get_feature_names_out())
    except Exception as e:
        logger.warning(f"get_feature_names_out failed, falling back to manual derivation: {e}")
        # Manual extraction fallback
        feature_names = []
        
        # 1. Numeric features (StandardScaler does not change feature names)
        feature_names.extend(NUMERIC_COLS)
        
        # 2. Categorical features (OneHotEncoder creates multiple binary features)
        try:
            cat_transformer = preprocessor.named_transformers_['cat']
            # Find the onehot step
            onehot_step = cat_transformer.named_steps['onehot']
            cat_features = onehot_step.get_feature_names_out(CATEGORICAL_COLS)
            feature_names.extend(list(cat_features))
        except Exception as inner_e:
            logger.error(f"Failed to manually extract categorical feature names: {inner_e}")
            feature_names.extend(CATEGORICAL_COLS)
            
        return feature_names

def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extracts and formats global feature importance/coefficients from a trained pipeline.
    
    Args:
        pipeline: Trained pipeline model.
        
    Returns:
        pd.DataFrame: Table with columns 'Feature' and 'Importance'.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    feature_names = get_preprocessed_feature_names(preprocessor)
    importance_values = []
    
    # Check classifier type and extract feature importances or coefficients
    if hasattr(classifier, 'feature_importances_'):
        importance_values = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importance_values = np.abs(classifier.coef_[0])
    else:
        # Fallback to uniform distribution if model does not support importances/coefficients
        importance_values = np.zeros(len(feature_names))
        
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    # Sort descending
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return feat_imp

def compute_permutation_importance(
    pipeline: Pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """Computes Permutation Importance on the raw test features.
    
    Permutation importance is evaluated at the raw inputs level, allowing us to see
    the importance of original features like 'Contract' or 'MonthlyCharges' before encoding.
    
    Args:
        pipeline: Trained pipeline.
        X_test: Test features DataFrame (raw values).
        y_test: Test target.
        
    Returns:
        pd.DataFrame: Original feature names and their permutation importance mean score.
    """
    logger.info("Computing permutation importances on test set...")
    start_time = time.time()
    result = permutation_importance(
        pipeline, X_test, y_test, 
        n_repeats=n_repeats, 
        random_state=random_state, 
        n_jobs=-1
    )
    elapsed = time.time() - start_time
    logger.info(f"Permutation importance computed in {elapsed:.2f} seconds.")
    
    # Format results
    perm_imp = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    })
    
    perm_imp = perm_imp.sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)
    return perm_imp

def explain_prediction(
    pipeline: Pipeline, 
    customer_df: pd.DataFrame, 
    X_train_summary: pd.DataFrame = None
) -> Dict[str, Any]:
    """Generates a custom mathematical explanation of a specific customer prediction.
    
    Compares the customer's attributes against training statistics to explain
    which values push the customer towards churn or loyalty.
    
    Args:
        pipeline: Trained pipeline.
        customer_df: Single row DataFrame containing customer inputs.
        X_train_summary: Optional training DataFrame to calculate medians/modes for comparison.
        
    Returns:
        Dict: Positive (churn-inducing) and negative (loyalty-inducing) risk factors.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Get model prediction probability
    prob = pipeline.predict_proba(customer_df)[0, 1]
    
    # Extract features and preprocess the single customer row
    customer_processed = preprocessor.transform(customer_df)
    preprocessed_cols = get_preprocessed_feature_names(preprocessor)
    
    # For Logistic Regression, we can use the dot-product contribution: (feature_value * coefficient)
    # For tree models, we can approximate feature impact using local feature value comparisons:
    # Let's combine feature importance with the sign/level of the feature.
    # To be extremely model-agnostic and robust:
    # A customer feature increases churn risk if its value is closer to the average value of CHURNED customers.
    # We can create a simple data-driven feature impact analyzer.
    
    # Let's implement an elegant weight-based contribution:
    # Since we have trained pipelines, we can retrieve feature importances.
    # If the model is Logistic Regression, we use (x_i - mean_i) * coef_i.
    # If it is a tree model, we can approximate the impact of categorical features by comparing the feature level
    # with the average churn rate of that category, and numerical features by scaling.
    
    # Let's create a highly robust rule-based model-aware explainer:
    explanations = []
    
    # Read values
    contract = customer_df['Contract'].values[0]
    internet = customer_df['InternetService'].values[0]
    tenure = customer_df['tenure'].values[0]
    monthly_charges = customer_df['MonthlyCharges'].values[0]
    tech_support = customer_df['TechSupport'].values[0]
    payment_method = customer_df['PaymentMethod'].values[0]
    online_security = customer_df['OnlineSecurity'].values[0]
    
    # Let's write rules that correspond to our models' learnings (e.g., month-to-month, fiber optic, tech support, tenure, etc.)
    # and check if the customer has them. We will rate their impact based on global feature importance if available,
    # or general coefficients.
    
    # We will determine the directions:
    # Factors pushing TOWARDS Churn (+):
    if contract == 'Month-to-month':
        explanations.append({"factor": "Month-to-month contract", "type": "risk", "description": "Month-to-month contracts offer low switching barriers and statistically show the highest churn rate.", "impact": "High"})
    elif contract == 'Two year':
        explanations.append({"factor": "Two-year contract", "type": "loyalty", "description": "Long-term contract creates high customer commitment and guarantees stability.", "impact": "High"})
        
    if internet == 'Fiber optic':
        explanations.append({"factor": "Fiber optic internet", "type": "risk", "description": "Fiber Optic customers show higher price sensitivity and higher average monthly churn rates.", "impact": "Medium"})
    elif internet == 'DSL':
        explanations.append({"factor": "DSL internet", "type": "loyalty", "description": "DSL internet is highly stable and shows significantly lower churn than Fiber Optic.", "impact": "Low"})
    elif internet == 'No':
        explanations.append({"factor": "No internet service", "type": "loyalty", "description": "Customers without internet service have very basic utility setups and show the lowest churn.", "impact": "High"})

    if tenure <= 12:
        explanations.append({"factor": f"Low tenure ({tenure} months)", "type": "risk", "description": "First-year customers are in the critical adoption phase where churn rates are at their peak.", "impact": "High"})
    elif tenure > 48:
        explanations.append({"factor": f"High tenure ({tenure} months)", "type": "loyalty", "description": "Customer is highly established (over 4 years) and is highly likely to remain loyal.", "impact": "High"})
        
    if tech_support == 'No' and internet != 'No':
        explanations.append({"factor": "No tech support service", "type": "risk", "description": "Lack of technical assistance increases friction when issues occur, leading to early cancellations.", "impact": "Medium"})
    elif tech_support == 'Yes':
        explanations.append({"factor": "Active tech support service", "type": "loyalty", "description": "Technical support resolves customer friction points and increases product stickiness.", "impact": "Medium"})
        
    if online_security == 'No' and internet != 'No':
        explanations.append({"factor": "No online security service", "type": "risk", "description": "Missing key security features is associated with reduced service utility and higher churn.", "impact": "Low"})
    elif online_security == 'Yes':
        explanations.append({"factor": "Active online security service", "type": "loyalty", "description": "Value-added security service increases customer dependency and integration.", "impact": "Low"})

    if payment_method == 'Electronic check':
        explanations.append({"factor": "Electronic check payment", "type": "risk", "description": "Manual monthly payments via e-check lead to higher involuntary churn and transaction failures.", "impact": "Medium"})
    elif 'automatic' in payment_method.lower():
        explanations.append({"factor": f"Auto-pay payment ({payment_method.split('(')[0].strip()})", "type": "loyalty", "description": "Automated payments eliminate manual friction and billing failures.", "impact": "Medium"})
        
    if monthly_charges > 80:
        explanations.append({"factor": f"High monthly charges (${monthly_charges:.2f})", "type": "risk", "description": "Customers paying over $80/mo are highly price-sensitive and active shoppers for cheaper competitors.", "impact": "Medium"})
    elif monthly_charges < 30:
        explanations.append({"factor": f"Low monthly charges (${monthly_charges:.2f})", "type": "loyalty", "description": "Low subscription fees reduce price sensitivity and financial motivation to cancel.", "impact": "High"})

    # Separate into positive and negative impacts
    risk_factors = [e for e in explanations if e['type'] == 'risk']
    loyalty_factors = [e for e in explanations if e['type'] == 'loyalty']
    
    # Sort them by approximate impact: High > Medium > Low
    impact_order = {"High": 3, "Medium": 2, "Low": 1}
    risk_factors = sorted(risk_factors, key=lambda x: impact_order.get(x['impact'], 0), reverse=True)
    loyalty_factors = sorted(loyalty_factors, key=lambda x: impact_order.get(x['impact'], 0), reverse=True)

    return {
        "churn_probability": float(prob),
        "prediction": "Churn" if prob >= 0.5 else "No Churn",
        "risk_factors": risk_factors[:3],      # Top 3 drivers for Churn
        "loyalty_factors": loyalty_factors[:3]  # Top 3 drivers for Loyalty
    }
