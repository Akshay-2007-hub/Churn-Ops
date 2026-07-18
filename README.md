# ChurnOps AI - Backend

This repository contains the Machine Learning pipeline and FastAPI backend for the ChurnOps AI project. It uses the official 21-column IBM Telco Customer Churn schema.

## Features
- **Synthetic Data Generator**: Generates 7,043 rows of realistic telco customer data.
- **XGBoost Pipeline**: A Scikit-Learn `Pipeline` with `StandardScaler` and `OneHotEncoder` trained on the dataset.
- **MLflow Tracking**: Model parameters and metrics are tracked locally.
- **FastAPI Inference Server**: Exposes a `/predict` endpoint that returns probability, risk levels, AI recommendations based on business rules, and the `revenue_at_risk`.

## Setup & Running Locally

1. **Create and Activate Virtual Environment:**
   ```bash
   cd ml
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model:**
   ```bash
   python training/train.py
   ```
   *This will generate the synthetic data, train the model, log to MLflow, and save `models/best_model.pkl`.*

4. **Run the FastAPI Server:**
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Test the API:**
   Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to interact with the endpoints.

## Using Docker

```bash
cd ml
docker build -t churn-ops-backend .
docker run -p 8000:8000 churn-ops-backend
```