import os
import json
import logging
import joblib
import pandas as pd
from utils.config import DATA_PATH, MODEL_PATH, METRICS_PATH, IMPORTANCE_PATH, MODEL_DIR
from utils.data_processing import load_data, split_data, get_preprocessor
from utils.ml_pipeline import (
    train_models, 
    evaluate_models, 
    get_feature_importance, 
    compute_permutation_importance
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=" * 60)
        logger.info("STARTING CUSTOMER CHURN PLATFORM ML TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # 1. Load data
        df = load_data(DATA_PATH)
        
        # 2. Split data
        X_train, X_test, y_train, y_test = split_data(df)
        
        # 3. Create preprocessor
        preprocessor = get_preprocessor()
        
        # 4. Train models
        trained_pipelines = train_models(X_train, y_train, preprocessor)
        
        # 5. Evaluate models
        comparison_df, detailed_results = evaluate_models(trained_pipelines, X_test, y_test)
        
        # Log comparison table
        logger.info("\nModel Comparison Table:")
        print(comparison_df.to_string(index=False))
        
        # 6. Select the best model based on F1-Score
        # We use F1-Score because churn datasets are imbalanced and we want to balance Precision & Recall
        best_model_name = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
        best_pipeline = trained_pipelines[best_model_name]
        logger.info(f"\n---> Best Model Selected: {best_model_name} (based on F1-Score)")
        
        # 7. Save best model pipeline
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_pipeline, MODEL_PATH)
        logger.info(f"Saved best model pipeline to: {MODEL_PATH}")
        
        # 8. Save comparison metrics and evaluation data
        # Merge comparison metrics and detailed results into a single file for dashboard loading
        comparison_dict = comparison_df.to_dict(orient="records")
        metrics_payload = {
            "comparison": comparison_dict,
            "detailed": detailed_results,
            "best_model_name": best_model_name
        }
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=4)
        logger.info(f"Saved evaluation metrics to: {METRICS_PATH}")
        
        # 9. Extract and save global feature importances for the best model
        feature_imp_df = get_feature_importance(best_pipeline)
        feature_imp_dict = feature_imp_df.to_dict(orient="records")
        with open(IMPORTANCE_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_imp_dict, f, indent=4)
        logger.info(f"Saved global feature importances to: {IMPORTANCE_PATH}")
        
        # 10. Compute and save permutation importance for the best model
        perm_imp_df = compute_permutation_importance(best_pipeline, X_test, y_test)
        perm_imp_path = os.path.join(MODEL_DIR, "permutation_importance.json")
        with open(perm_imp_path, "w", encoding="utf-8") as f:
            json.dump(perm_imp_df.to_dict(orient="records"), f, indent=4)
        logger.info(f"Saved permutation importances to: {perm_imp_path}")
        
        logger.info("=" * 60)
        logger.info("ML TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to execute training pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
