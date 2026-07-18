import joblib
import os
import pandas as pd
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")

class ChurnModel:
    def __init__(self):
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Loads the trained pipeline from the file system."""
        try:
            if os.path.exists(MODEL_PATH):
                self.pipeline = joblib.load(MODEL_PATH)
                print(f"Model pipeline successfully loaded from {MODEL_PATH}")
            else:
                print(f"Warning: Model pipeline not found at {MODEL_PATH}. Using mock predictions for development.")
                self.pipeline = None
        except Exception as e:
            print(f"Error loading model pipeline: {e}")
            self.pipeline = None

    def predict(self, data: pd.DataFrame):
        """Predicts churn probability and class using the trained pipeline."""
        if self.pipeline:
            # The pipeline automatically handles preprocessing (StandardScaler, OneHotEncoder)
            probability = self.pipeline.predict_proba(data)[0][1]
            prediction = int(self.pipeline.predict(data)[0])
        else:
            # Mock behavior if model doesn't exist yet
            probability = float(np.random.rand())
            prediction = 1 if probability > 0.5 else 0
        
        return prediction, float(probability)
