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

def generate_telco_churn_data(n_samples=7043):
    np.random.seed(42)
    # 1. Demographic Features
    customer_ids = [f"{np.random.randint(1000, 9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n_samples)]
    gender = np.random.choice(['Male', 'Female'], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.30, 0.70])
    
    # 2. Account Features (Tenure in months)
    tenure = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenure = np.clip(tenure, 1, 72)  # Cap between 1 and 72 months
    
    # Contract type strongly correlated with tenure
    contract = []
    for t in tenure:
        if t < 12:
            contract.append(np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.85, 0.10, 0.05]))
        elif t < 36:
            contract.append(np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.40, 0.45, 0.15]))
        else:
            contract.append(np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.15, 0.35, 0.50]))
            
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
        size=n_samples, p=[0.33, 0.23, 0.22, 0.22]
    )
    
    # 3. Service Subscriptions
    phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.90, 0.10])
    multiple_lines = [np.random.choice(['Yes', 'No']) if ps == 'Yes' else 'No phone service' for ps in phone_service]
    
    internet_service = np.random.choice(['Fiber optic', 'DSL', 'No'], size=n_samples, p=[0.44, 0.34, 0.22])
    
    def get_addon(net_service, p_yes=0.3):
        return [np.random.choice(['Yes', 'No'], p=[p_yes, 1-p_yes]) if inet != 'No' else 'No internet service' for inet in net_service]
    
    online_security = get_addon(internet_service, p_yes=0.28)
    online_backup = get_addon(internet_service, p_yes=0.34)
    device_protection = get_addon(internet_service, p_yes=0.34)
    tech_support = get_addon(internet_service, p_yes=0.29)
    streaming_tv = get_addon(internet_service, p_yes=0.38)
    streaming_movies = get_addon(internet_service, p_yes=0.39)
    
    # 4. Financial Metrics (Monthly and Total Charges)
    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if phone_service[i] == 'Yes': base += 15.0
        if multiple_lines[i] == 'Yes': base += 10.0
        if internet_service[i] == 'DSL': base += 25.0
        elif internet_service[i] == 'Fiber optic': base += 50.0
        for addon in [online_security[i], online_backup[i], device_protection[i], tech_support[i], streaming_tv[i], streaming_movies[i]]:
            if addon == 'Yes': base += 8.0
        monthly_charges.append(round(base + np.random.uniform(-3, 3), 2))
        
    total_charges = [round(mc * t + np.random.uniform(-10, 10), 2) for mc, t in zip(monthly_charges, tenure)]
    total_charges = [max(19.85, tc) for tc in total_charges] # Ensure positive realistic minimum
    
    # 5. Target Variable: Churn Probability Calculation (Grounding rules for ML models)
    churn_prob = np.zeros(n_samples) + 0.25
    for i in range(n_samples):
        if contract[i] == 'Month-to-month': churn_prob[i] += 0.25
        if contract[i] == 'Two year': churn_prob[i] -= 0.20
        if internet_service[i] == 'Fiber optic': churn_prob[i] += 0.15
        if tech_support[i] == 'No': churn_prob[i] += 0.10
        if online_security[i] == 'No': churn_prob[i] += 0.05
        if payment_method[i] == 'Electronic check': churn_prob[i] += 0.10
        if tenure[i] > 36: churn_prob[i] -= 0.15
        if senior_citizen[i] == 1: churn_prob[i] += 0.08
        
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = [np.random.choice([1, 0], p=[p, 1-p]) for p in churn_prob]
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids, 'gender': gender, 'SeniorCitizen': senior_citizen,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges, 'Churn': churn
    })
    return df

def train_model():
    print("Generating synthetic 21-column Telco data (N=7043)...")
    df = generate_telco_churn_data(7043)
    
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing steps
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", 
        "PhoneService", "MultipleLines", "InternetService", 
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
        "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])
    
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
        
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier", "numpy.dtype"])
        
        model_path = "models/best_model.pkl"
        joblib.dump(pipeline, model_path)
        print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()
