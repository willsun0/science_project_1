import joblib
import numpy as np

# Load the pre-trained XGBoost model
model = joblib.load("xgb_model.pkl")

def predict_wqi(latest_features: np.ndarray) -> float:
    prediction = model.predict(latest_features)
    return float(prediction[0])
