import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_model(model, X_test, y_test, model_name):
    print(f"\\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    return metrics

def train_models(config):
    processed_path = config['data']['processed_path']
    target_col = config['data']['target_col']
    
    print(f"Loading processed data from {processed_path}...")
    df = pd.read_csv(processed_path)
    
    X = df.drop(columns=[target_col, 'customer_id'])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'], stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    results = {}
    
    # 1. Logistic Regression (Baseline)
    print("\\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    results["Logistic Regression"] = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 2. Random Forest
    print("\\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    results["Random Forest"] = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # 3. XGBoost (Main Model)
    print("\\n--- Training XGBoost ---")
    
    # We will do a small GridSearchCV to find the best XGBoost params
    # Using the params defined in config
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Fast parameters for the demo to not take forever
    param_grid = {
        'learning_rate': [0.1],
        'max_depth': [5, 7],
        'n_estimators': [100],
        'scale_pos_weight': [sum(y_train==0)/sum(y_train==1)] # Handle class imbalance for better recall
    }
    
    print("Running GridSearchCV for XGBoost to optimize ROC-AUC...")
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        cv=3, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    
    print(f"Best XGBoost params: {grid_search.best_params_}")
    results["XGBoost (Optimized)"] = evaluate_model(best_xgb, X_test, y_test, "XGBoost")
    
    # Save the best model
    model_path = config['paths']['model']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_xgb, model_path)
    print(f"\\nSaved best XGBoost model to {model_path}")
    
    # Save metrics
    metrics_path = config['paths']['metrics']
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved evaluation metrics to {metrics_path}")

if __name__ == "__main__":
    config = load_config()
    train_models(config)
