from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime

from api.schemas import PredictionRequest, PredictionResponse, ModelInfo
from api.model_loader import ChurnModel

app = FastAPI(
    title="Churn Prediction API",
    description="FastAPI service for predicting customer churn.",
    version="1.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ChurnModel()
startup_time = datetime.now()

@app.get("/")
def read_root():
    return {"message": "Welcome to the ChurnOps AI Prediction API (21-Column Edition)"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "uptime": str(datetime.now() - startup_time),
        "model_loaded": model.pipeline is not None
    }

@app.get("/model-info", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        version="1.1.0",
        algorithm="XGBoost Pipeline",
        training_date=str(datetime.now().date()),
        features_required=[
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService", 
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
            "TechSupport", "StreamingTV", "StreamingMovies", 
            "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
    )

def get_risk_level(probability: float) -> str:
    if probability < 0.4:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def get_recommendation(data: dict, probability: float) -> str:
    monthly_charges = float(data.get("MonthlyCharges", 0))
    internet_service = data.get("InternetService")
    contract = data.get("Contract")
    tech_support = data.get("TechSupport")
    online_security = data.get("OnlineSecurity")
    tenure = int(data.get("tenure", 0))

    # Rule 1: Discount / Loyalty Reward
    if monthly_charges > 80 and internet_service == "Fiber optic" and probability > 0.6:
        return "Discount / Loyalty Reward: High-risk fiber customer. Offer a $10-$15 monthly discount to retain."
    
    # Rule 2: Premium Customer Support / Tech Upgrade
    if contract == "Month-to-month" and (tech_support == "No" or online_security == "No"):
        return "Premium Customer Support / Tech Upgrade: Vulnerable month-to-month user. Bundle free tech support or online security."
    
    # Rule 3: Contract Lock-In Incentive (Service Upgrade)
    if contract == "Month-to-month" and 12 <= tenure <= 24:
        return "Contract Lock-In Incentive: Offer a free streaming add-on (StreamingTV or StreamingMovies) in exchange for switching to a One-year contract."

    # Fallbacks
    risk = get_risk_level(probability)
    if risk == "High":
        return "Immediate Intervention: Recommend manager outreach and personalized retention package."
    elif risk == "Medium":
        return "Standard Check-in: Send an automated follow-up survey or offer a small loyalty perk."
    else:
        return "No action required. Standard retention policies apply."

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(request: PredictionRequest):
    try:
        # Use by_alias=True to match exactly what the scikit-learn pipeline expects
        data_dict = request.model_dump(by_alias=True, exclude={"customer_id"})
        df = pd.DataFrame([data_dict])
        
        prediction, probability = model.predict(df)
        
        pred_label = "Yes" if prediction == 1 else "No"
        risk_level = get_risk_level(probability)
        recommendation = get_recommendation(data_dict, probability)
        
        # Calculate revenue at risk: 12 * MonthlyCharges * Probability
        monthly_charges = float(data_dict.get("MonthlyCharges", 0))
        revenue_at_risk = round(12.0 * monthly_charges * probability, 2)
        
        return PredictionResponse(
            prediction=pred_label,
            probability=round(probability, 2),
            risk_level=risk_level,
            recommendation=recommendation,
            revenue_at_risk=revenue_at_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
