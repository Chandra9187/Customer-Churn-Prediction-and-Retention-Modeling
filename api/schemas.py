from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerData(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, example=650)
    geography: str = Field(..., example="France")
    gender: str = Field(..., example="Female")
    age: int = Field(..., ge=18, le=100, example=40)
    tenure: int = Field(..., ge=0, le=10, example=5)
    balance: float = Field(..., ge=0, example=85000.50)
    num_of_products: int = Field(..., ge=1, le=4, example=2)
    has_cr_card: int = Field(..., ge=0, le=1, example=1)
    is_active_member: int = Field(..., ge=0, le=1, example=1)
    estimated_salary: float = Field(..., ge=0, example=120000.00)
    complain: int = Field(..., ge=0, le=1, example=0)
    satisfaction_score: int = Field(..., ge=1, le=5, example=4)
    card_type: str = Field(..., example="Gold")
    points_earned: int = Field(..., ge=0, example=500)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    expected_loss: float

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]

class MetricsResponse(BaseModel):
    models: dict
    
class SummaryResponse(BaseModel):
    total_expected_loss: float
    top_10_percent_loss: float
    top_10_contribution_pct: float
