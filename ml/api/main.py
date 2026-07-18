from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime

from api.schemas import PredictionRequest, PredictionResponse, ModelInfo
from api.model_loader import ChurnModel

app = FastAPI(
    title="Churn Prediction API",
    description="FastAPI service for predicting customer churn.",
    version="1.0.0"
)

# CORS configuration for connecting with the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ChurnModel()
startup_time = datetime.now()

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "uptime": str(datetime.now() - startup_time),
        "model_loaded": model.model is not None
    }

@app.get("/metrics")
def get_metrics():
    """Returns evaluation metrics for the current model."""
    # In a real-world scenario, these should be dynamically fetched from MLflow
    return {
        "accuracy": 0.85,
        "precision": 0.79,
        "recall": 0.81,
        "f1_score": 0.80,
        "roc_auc": 0.88
    }

@app.get("/model-info", response_model=ModelInfo)
def model_info():
    """Returns information about the currently loaded model."""
    return ModelInfo(
        version="1.0.0",
        algorithm="XGBoost",
        training_date="2023-10-25T14:30:00Z", # This could be read from MLflow artifacts
        features_required=[
            "gender", "senior_citizen", "partner", "dependents",
            "tenure", "phone_service", "internet_service", 
            "contract", "payment_method", "monthly_charges", "total_charges"
        ]
    )

def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def get_recommendation(risk_level: str) -> str:
    if risk_level == "Low":
        return "Standard retention efforts."
    elif risk_level == "Medium":
        return "Offer a small discount or follow-up call."
    else:
        return "Immediate intervention required. Offer premium retention package."

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(request: PredictionRequest):
    """Predicts churn for a single customer."""
    try:
        data_dict = request.model_dump(exclude={"customer_id"})
        df = pd.DataFrame([data_dict])
        
        # Here you would typically apply the exact same preprocessing pipeline (scaling, encoding) 
        # that was fit during training. For example:
        # df_processed = preprocessor.transform(df)
        
        prediction, probability = model.predict(df)
        
        pred_label = "Yes" if prediction == 1 else "No"
        risk_level = get_risk_level(probability)
        recommendation = get_recommendation(risk_level)
        
        return PredictionResponse(
            prediction=pred_label,
            probability=round(probability, 2),
            risk_level=risk_level,
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
