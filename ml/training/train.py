import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Telco_Customer_Churn")

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "senior_citizen": np.random.choice([0, 1], n_samples),
        "partner": np.random.choice(["Yes", "No"], n_samples),
        "dependents": np.random.choice(["Yes", "No"], n_samples),
        "tenure": np.random.randint(1, 72, n_samples),
        "phone_service": np.random.choice(["Yes", "No"], n_samples),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n_samples),
        "monthly_charges": np.random.uniform(18.0, 118.0, n_samples),
        "total_charges": np.random.uniform(18.0, 8000.0, n_samples),
    }
    
    # Introduce some logic for churn based on standard Telco features
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(data["contract"] == "Month-to-month", 0.3, 0)
    churn_prob += np.where(data["internet_service"] == "Fiber optic", 0.15, 0)
    churn_prob -= np.where(data["tenure"] > 24, 0.2, 0)
    churn_prob -= np.where(data["partner"] == "Yes", 0.1, 0)
    churn_prob -= np.where(data["dependents"] == "Yes", 0.1, 0)
    
    # Clip probability to [0, 1] range and generate label
    churn_prob = np.clip(churn_prob + 0.15, 0.05, 0.95)
    data["churn"] = np.random.binomial(1, churn_prob)
    
    return pd.DataFrame(data)

def train_model():
    print("Generating synthetic Telco data...")
    df = generate_synthetic_data(2500)
    
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing steps
    numeric_features = ["tenure", "monthly_charges", "total_charges"]
    categorical_features = ["gender", "senior_citizen", "partner", "dependents", 
                            "phone_service", "internet_service", "contract", "payment_method"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])
    
    # Define scikit-learn pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss"))
    ])
    
    with mlflow.start_run():
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        print("Evaluating model...")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        print(f"Validation Metrics: {metrics}")
        
        # Log parameters, metrics, and model to MLflow
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier", "numpy.dtype"])
        
        # Save pipeline for FastAPI
        model_path = "models/best_model.pkl"
        joblib.dump(pipeline, model_path)
        print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()
