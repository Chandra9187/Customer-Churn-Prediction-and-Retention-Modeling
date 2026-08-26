import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_expected_loss_analysis(config):
    """Calculate expected financial loss and top 10% contribution."""
    print("Running Expected Financial Loss Analysis...")
    
    # We will use the raw generated data for balances to keep amounts interpretable
    # and use the trained model on processed data
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    model_path = config['paths']['model']
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run model_training.py first.")
        return
        
    df_raw = pd.read_csv(raw_path)
    df_proc = pd.read_csv(processed_path)
    model = joblib.load(model_path)
    
    # We evaluate on the entire dataset for this business analysis
    X_proc = df_proc.drop(columns=[config['data']['target_col'], 'customer_id'])
    
    # Predict churn probabilities
    print("Predicting churn probabilities for all customers...")
    churn_probs = model.predict_proba(X_proc)[:, 1]
    
    # Calculate Expected Loss
    # Expected Loss = P(Churn) * Balance
    # (Simplified assumption: loss is the entire balance leaving the bank)
    
    analysis_df = pd.DataFrame({
        'customer_id': df_raw['customer_id'],
        'balance': df_raw['balance'],
        'churn_probability': churn_probs,
        'exited_actual': df_raw[config['data']['target_col']]
    })
    
    analysis_df['expected_loss'] = analysis_df['churn_probability'] * analysis_df['balance']
    
    # Sort by expected loss descending
    analysis_df = analysis_df.sort_values(by='expected_loss', ascending=False).reset_index(drop=True)
    
    total_expected_loss = analysis_df['expected_loss'].sum()
    
    # Calculate top 10% metrics
    top_10_percent_idx = int(len(analysis_df) * 0.1)
    top_10_percent_loss = analysis_df.head(top_10_percent_idx)['expected_loss'].sum()
    
    top_10_contribution = (top_10_percent_loss / total_expected_loss) * 100
    
    print(f"\\n--- Expected Loss Results ---")
    print(f"Total Expected Loss across all customers: ${total_expected_loss:,.2f}")
    print(f"Top 10% risky customers Expected Loss: ${top_10_percent_loss:,.2f}")
    print(f"\\nTop 10% of customers contribute to {top_10_contribution:.1f}% of total expected loss.")
    
    # Create retention priority tiers
    # High: Top 10%, Medium: 10-30%, Low: Bottom 70%
    analysis_df['retention_priority'] = 'Low'
    p30_idx = int(len(analysis_df) * 0.3)
    
    analysis_df.loc[:p30_idx, 'retention_priority'] = 'Medium'
    analysis_df.loc[:top_10_percent_idx, 'retention_priority'] = 'High'
    
    # Save the analysis summary
    summary = {
        "total_expected_loss": float(total_expected_loss),
        "top_10_percent_loss": float(top_10_percent_loss),
        "top_10_contribution_pct": float(top_10_contribution)
    }
    
    summary_path = "results/expected_loss_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    config = load_config()
    run_expected_loss_analysis(config)
