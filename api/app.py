import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.schemas import CustomerData, PredictionResponse, BatchPredictionRequest, MetricsResponse, SummaryResponse

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-End ML Pipeline for predicting bank customer churn and expected loss.",
    version="1.0.0"
)

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = "models/xgboost_final.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
METRICS_PATH = "results/metrics.json"
SUMMARY_PATH = "results/expected_loss_summary.json"

# Global variables for models
model = None
preprocessor = None
metrics_data = None
summary_data = None

@app.on_event("startup")
def load_assets():
    global model, preprocessor, metrics_data, summary_data
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        if os.path.exists(SUMMARY_PATH):
            with open(SUMMARY_PATH, "r") as f:
                summary_data = json.load(f)
                
        print("Successfully loaded ML assets.")
    except Exception as e:
        print(f"Error loading assets: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([customer.dict()])
    
    try:
        # Preprocess
        X_proc = preprocessor.transform(df)
        
        # Predict
        prob = float(model.predict_proba(X_proc)[0, 1])
        pred = int(model.predict(X_proc)[0])
        
        # Risk assessment
        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        
        # Expected Loss
        expected_loss = float(prob * customer.balance)
        
        return PredictionResponse(
            churn_probability=prob,
            churn_prediction=pred,
            risk_level=risk_level,
            expected_loss=expected_loss
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    if not metrics_data:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return {"models": metrics_data}

@app.get("/eda/summary", response_model=SummaryResponse)
def get_summary():
    if not summary_data:
        raise HTTPException(status_code=404, detail="Summary data not found")
    return summary_data

# Mount frontend static files last so API routes take precedence
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
else:
    print("Warning: frontend directory not found. Static files will not be served.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
