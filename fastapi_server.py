from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# === Load models and resources ===
ensemble_model = joblib.load("voting_ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("feature_columns.pkl")

# Try to load encoder (if categoricals were used during training)
try:
    ordinal_encoder = joblib.load("ordinal_encoder.pkl")
    use_encoder = True
except FileNotFoundError:
    ordinal_encoder = None
    use_encoder = False

# === Define input schema ===
class PredictionInput(BaseModel):
    user_id: str
    total_orders: int
    total_returns: int
    days_to_return_avg: float
    high_value_returns: int
    category_return_ratio: float
    exchange_ratio: float
    damaged_returns: int
    location: str = None
    device_fingerprint: str = None

# === FastAPI app ===
app = FastAPI(title="Walmart Hack Voting Ensemble API")

@app.get("/")
def root():
    return {"message": "Voting Ensemble API is running!"}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Feature Engineering
    df['return_rate'] = df['total_returns'] / (df['total_orders'] + 1e-5)
    df['fast_return_flag'] = (df['days_to_return_avg'] < 3).astype(int)

    # Drop unused / leaky
    df.drop(columns=['user_id', 'high_value_returns', 'damaged_returns'], inplace=True, errors='ignore')

    expected_cols = joblib.load("feature_columns.pkl")
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Encode categoricals if applicable
    cat_cols = ['location', 'device_fingerprint']
    cat_cols = [col for col in cat_cols if col in df.columns and use_encoder]
    if cat_cols:
        try:
            df[cat_cols] = ordinal_encoder.transform(df[cat_cols])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Encoding failed: {str(e)}")

    # Scale
    try:
        df_scaled = scaler.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling failed: {str(e)}")

    # Predict class and probability
    try:
        prediction = int(ensemble_model.predict(df_scaled)[0])
        proba = ensemble_model.predict_proba(df_scaled)[0][prediction]  # Probability of predicted class
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Custom ReturnShield AI messaging
    if prediction == 0:
        message = (
            "Low Return Risk: This customer is unlikely to abuse return policies. "
            "They are expected to be a genuine buyer. Continue with confidence."
        )
    else:
        message = (
            "High Return Risk: ReturnShield AI flags this customer as likely to abuse return policies. "
            "Recommend monitoring their activity or flagging for review to save costs."
        )

    return {
        "prediction": prediction,
        "probability": round(proba, 4),
        "message": message,
        "note": "Prediction labels: 0 = Low Risk, 1 = High Risk"
    }
