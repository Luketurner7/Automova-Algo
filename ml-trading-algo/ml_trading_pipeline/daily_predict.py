# ml_trading_pipeline/daily_predict.py

import pandas as pd
from datetime import datetime
import joblib

# Load saved model, scaler, features
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features_list.pkl")  # You need to save features list after training

# Load latest preprocessed data (for today's date or last market date)
latest_data = pd.read_csv("data/latest_data.csv")

# Feature scaling
latest_X = scaler.transform(latest_data[features])

# Predict probabilities
latest_data["pred_proba_up"] = model.predict_proba(latest_X)[:, 1]

# Calculate ATR% and filter high volatility
latest_data["ATR%"] = latest_data["volatility_atr"] / latest_data["Close"]
latest_data = latest_data[latest_data["ATR%"] > 0.02]

# Filter by confidence threshold > 0.7
top_preds = latest_data[latest_data["pred_proba_up"] > 0.7]

# Save predictions with date stamp
out_filename = f"data/predictions_{datetime.today().date()}.csv"
top_preds[["ticker", "pred_proba_up", "Close"]].to_csv(out_filename, index=False)

print(f"\n✅ Saved today's high-confidence predictions to {out_filename}")
