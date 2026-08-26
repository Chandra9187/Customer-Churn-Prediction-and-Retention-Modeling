import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder
import yaml
import joblib
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_preprocessor(config):
    """Build the sklearn preprocessing pipeline according to config specifications."""
    num_cols = config['features']['numerical']
    cat_cols = config['features']['categorical']
    # Binary columns don't need transformation generally if they are strictly 0/1 without nulls
    
    # Needs Yeo-Johnson transform for skewed distributions as per requirements
    cols_to_transform = ['balance', 'estimated_salary', 'credit_score']
    # Other numerics that just need scaling
    other_num_cols = [c for c in num_cols if c not in cols_to_transform]

    # Pipeline for highly skewed numericals (Impute -> Yeo-Johnson -> Scaler)
    skewed_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('yeo_johnson', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler())
    ])

    # Pipeline for other numericals (Impute -> Scaler)
    standard_num_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])

    # Pipeline for categoricals (Impute -> OneHot)
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('skewed', skewed_pipeline, cols_to_transform),
            ('num', standard_num_pipeline, other_num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='passthrough' # Keep binary columns and target as-is for now
    )
    
    return preprocessor

def preprocess_data(config):
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    model_dir = os.path.dirname(config['paths']['preprocessor'])
    
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Separate features and target
    target_col = config['data']['target_col']
    # Customer ID is not a predictive feature
    X = df.drop(columns=[target_col, 'customer_id'])
    y = df[target_col]
    
    # Ensure binary columns are properly typed before passing through
    bin_cols = config['features']['binary']
    for col in bin_cols:
        X[col] = X[col].astype(int)
    
    preprocessor = build_preprocessor(config)
    
    print("Fitting and transforming data (this may take a minute with KNNImputer on 100K rows)...")
    # This will return a numpy array if we don't specify output
    preprocessor.set_output(transform='pandas')
    X_processed = preprocessor.fit_transform(X)
    
    # Save the pipeline
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(preprocessor, config['paths']['preprocessor'])
    print(f"Saved preprocessor to {config['paths']['preprocessor']}")
    
    # Combine back with target and id for saving the processed dataset
    df_processed = X_processed.copy()
    df_processed['customer_id'] = df['customer_id']
    df_processed[target_col] = y
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path}")
    
    return X_processed, y

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)
