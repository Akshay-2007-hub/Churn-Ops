from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(None, description="Optional Customer ID")
    gender: str = Field(..., description="Male or Female")
    senior_citizen: int = Field(..., description="1 for yes, 0 for no")
    partner: str = Field(..., description="Yes or No")
    dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    phone_service: str = Field(..., description="Yes or No")
    internet_service: str = Field(..., description="DSL, Fiber optic, No")
    contract: str = Field(..., description="Month-to-month, One year, Two year")
    payment_method: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    monthly_charges: float = Field(..., description="The amount charged to the customer monthly")
    total_charges: float = Field(..., description="The total amount charged to the customer")

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    recommendation: str

class ModelInfo(BaseModel):
    version: str
    algorithm: str
    training_date: str
    features_required: list[str]
