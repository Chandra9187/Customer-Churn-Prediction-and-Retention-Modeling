import pandas as pd
import numpy as np
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_synthetic_data(num_samples: int) -> pd.DataFrame:
    """Generate a realistic banking customer churn dataset."""
    print(f"Generating {num_samples} synthetic customer records...")
    
    np.random.seed(42)
    
    # Base features
    customer_id = np.arange(10000000, 10000000 + num_samples)
    
    # Skewed left credit score (higher is more common)
    credit_score = np.random.normal(loc=650, scale=100, size=num_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    geography = np.random.choice(["France", "Germany", "Spain"], size=num_samples, p=[0.5, 0.25, 0.25])
    gender = np.random.choice(["Male", "Female"], size=num_samples, p=[0.55, 0.45])
    
    # Age normally distributed around 39, right skewed
    age = np.random.lognormal(mean=3.6, sigma=0.3, size=num_samples)
    age = np.clip(age, 18, 92).astype(int)
    
    tenure = np.random.randint(0, 11, size=num_samples)
    
    # Balance bimodal: many 0s, rest lognormal
    balance_zero_mask = np.random.choice([True, False], size=num_samples, p=[0.36, 0.64])
    balance = np.random.lognormal(mean=11.5, sigma=0.5, size=num_samples)
    balance[balance_zero_mask] = 0.0
    balance = np.round(balance, 2)
    
    num_of_products = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.5, 0.46, 0.03, 0.01])
    has_cr_card = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=num_samples, p=[0.48, 0.52])
    
    estimated_salary = np.random.uniform(low=10000, high=200000, size=num_samples)
    estimated_salary = np.round(estimated_salary, 2)
    
    satisfaction_score = np.random.randint(1, 6, size=num_samples)
    card_type = np.random.choice(["Silver", "Gold", "Platinum", "Diamond"], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])
    points_earned = np.random.randint(100, 1000, size=num_samples)
    
    # --- Target variable generation (Exited) ---
    # Create realistic correlations
    churn_prob = np.full(num_samples, 0.05)
    
    # Higher age -> higher churn
    churn_prob[age > 50] += 0.2
    churn_prob[age > 60] += 0.1
    
    # Lower balance -> lower churn, High balance -> moderate churn
    churn_prob[(balance > 0) & (balance < 50000)] += 0.1
    churn_prob[balance > 150000] += 0.15
    
    # Germany has higher churn in the real dataset typically
    churn_prob[geography == 'Germany'] += 0.15
    
    # Active members less likely to churn
    churn_prob[is_active_member == 1] -= 0.15
    
    # 3+ products strongly correlates with churn
    churn_prob[num_of_products >= 3] += 0.6
    
    # Generate complain indicator heavily correlated with churn
    complain = np.random.binomial(n=1, p=np.clip(churn_prob + 0.1, 0, 1))
    churn_prob[complain == 1] += 0.5
    churn_prob[complain == 0] -= 0.2

    # Final churn determination
    churn_prob = np.clip(churn_prob, 0, 1)
    exited = np.random.binomial(n=1, p=churn_prob)
    
    # Construct DataFrame
    df = pd.DataFrame({
        "customer_id": customer_id,
        "credit_score": credit_score,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_of_products": num_of_products,
        "has_cr_card": has_cr_card,
        "is_active_member": is_active_member,
        "estimated_salary": estimated_salary,
        "complain": complain,
        "satisfaction_score": satisfaction_score,
        "card_type": card_type,
        "points_earned": points_earned,
        "exited": exited
    })
    
    # Introduce ~5% missing values in selected columns for KNN imputation logic
    print("Introducing ~5% missing values in numeric columns...")
    cols_to_null = ['credit_score', 'age', 'balance', 'estimated_salary']
    for col in cols_to_null:
        null_mask = np.random.choice([True, False], size=num_samples, p=[0.05, 0.95])
        df.loc[null_mask, col] = np.nan
        
    return df

if __name__ == "__main__":
    config = load_config()
    num_samples = config['data']['num_samples']
    output_path = config['data']['raw_path']
    
    df = generate_synthetic_data(num_samples)
    
    print(f"Dataset generated with shape: {df.shape}")
    print(f"Overall churn rate: {df['exited'].mean():.2%}")
    print(f"Nulls summary:\\n{df.isnull().sum()}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")
