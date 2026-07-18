from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class PredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(None, description="Optional Customer ID")
    gender: str = Field(..., description="Male or Female")
    senior_citizen: int = Field(..., alias="SeniorCitizen", description="1 for yes, 0 for no")
    partner: str = Field(..., alias="Partner", description="Yes or No")
    dependents: str = Field(..., alias="Dependents", description="Yes or No")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    phone_service: str = Field(..., alias="PhoneService", description="Yes or No")
    multiple_lines: str = Field(..., alias="MultipleLines", description="Yes, No, No phone service")
    internet_service: str = Field(..., alias="InternetService", description="DSL, Fiber optic, No")
    online_security: str = Field(..., alias="OnlineSecurity", description="Yes, No, No internet service")
    online_backup: str = Field(..., alias="OnlineBackup", description="Yes, No, No internet service")
    device_protection: str = Field(..., alias="DeviceProtection", description="Yes, No, No internet service")
    tech_support: str = Field(..., alias="TechSupport", description="Yes, No, No internet service")
    streaming_tv: str = Field(..., alias="StreamingTV", description="Yes, No, No internet service")
    streaming_movies: str = Field(..., alias="StreamingMovies", description="Yes, No, No internet service")
    contract: str = Field(..., alias="Contract", description="Month-to-month, One year, Two year")
    paperless_billing: str = Field(..., alias="PaperlessBilling", description="Yes or No")
    payment_method: str = Field(..., alias="PaymentMethod", description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    monthly_charges: float = Field(..., alias="MonthlyCharges", description="The amount charged to the customer monthly")
    total_charges: float = Field(..., alias="TotalCharges", description="The total amount charged to the customer")

    model_config = ConfigDict(populate_by_name=True)

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    recommendation: str
    revenue_at_risk: float

class ModelInfo(BaseModel):
    version: str
    algorithm: str
    training_date: str
    features_required: list[str]
