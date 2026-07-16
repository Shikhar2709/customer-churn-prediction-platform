import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from utils.config import DATA_PATH, TARGET_COL, ID_COL, NUMERIC_COLS, CATEGORICAL_COLS

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads the customer churn dataset, cleans and validates it.
    
    Args:
        file_path: Path to the CSV dataset.
        
    Returns:
        pd.DataFrame: Cleaned customer churn data.
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # 1. Check duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows.")
            df.drop_duplicates(inplace=True)
            
        # 2. Preprocess TotalCharges (contains empty space characters)
        if 'TotalCharges' in df.columns:
            # Replace whitespace characters with NaN
            df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
            # Convert to numeric
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            # Impute NaN values with 0.0 (since these represent new customers with tenure = 0)
            df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
            
        # 3. Map SeniorCitizen (0/1) to (No/Yes) for better UI and consistency
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).fillna('No')
            
        # 4. Standardize Target Column (Churn: Yes -> 1, No -> 0)
        if TARGET_COL in df.columns:
            # Map Yes/No to 1/0
            df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            
        # Validate schema
        validate_data(df)
        
        logger.info(f"Dataset successfully loaded. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise e

def validate_data(df: pd.DataFrame) -> bool:
    """Validates the input DataFrame against the expected schema and types.
    
    Args:
        df: Pandas DataFrame.
        
    Returns:
        bool: True if data is valid.
    """
    required_cols = [ID_COL] + NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]
    
    # Check for missing required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Data validation failed: Missing columns {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Check for invalid values in tenure
    if (df['tenure'] < 0).any():
        logger.warning("Found negative values in 'tenure'. Clamping to 0.")
        df['tenure'] = df['tenure'].clip(lower=0)
        
    # Check for invalid values in MonthlyCharges
    if (df['MonthlyCharges'] < 0).any():
        logger.warning("Found negative values in 'MonthlyCharges'. Clamping to 0.")
        df['MonthlyCharges'] = df['MonthlyCharges'].clip(lower=0)
        
    logger.info("Data validation passed successfully.")
    return True

def get_preprocessor() -> ColumnTransformer:
    """Builds a Scikit-Learn preprocessor ColumnTransformer for scaling and encoding.
    
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    # Numerical pipeline: Imputation + Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Imputation + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])
    
    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ]
    )
    
    return preprocessor

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into features and target, and train/test sets.
    
    Args:
        df: Pandas DataFrame.
        test_size: Ratio of test dataset split.
        random_state: Random state seed.
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    # Features and Target
    X = df.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
    y = df[TARGET_COL]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data split completed: Train shape {X_train.shape}, Test shape {X_test.shape}")
    return X_train, X_test, y_train, y_test
