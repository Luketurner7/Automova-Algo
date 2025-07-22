# ml_trading_pipeline/backtest.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load preprocessed data
data = pd.read_csv("data/processed_stock_data.csv")

features = [col for col in data.columns if col.startswith("volume_") or 
            col.startswith("trend_") or col.startswith("momentum_") or 
            col.startswith("volatility_") or col.startswith("others_")]

initial_balance = 100000
balance = initial_balance
trade_log = []

# Sort dates and leave initial training window (e.g., 100 days)
dates = sorted(data["date"].unique())[100:]

for current_date in dates:
    train_data = data[data["date"] < current_date]
    test_data = data[data["date"] == current_date]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[features])
    y_train = train_data["target"]
    X_test = scaler.transform(test_data[features])

    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities
    pred_proba = model.predict_proba(X_test)[:, 1]
    test_data = test_data.copy()
    test_data["pred_proba_up"] = pred_proba

    # Calculate ATR% for volatility filtering
    test_data["ATR%"] = test_data["volatility_atr"] / test_data["Close"]

    # Filter high volatility stocks
    test_data = test_data[test_data["ATR%"] > 0.02]

    # Simulate trades for high confidence predictions (> 0.7)
    for _, row in test_data.iterrows():
        if row["pred_proba_up"] > 0.7 and not pd.isna(row["future_return_5d"]):
            position_size = balance * 0.01  # invest 1% per trade
            profit = position_size * row["future_return_5d"]
            balance += profit
            trade_log.append({
                "date": current_date,
                "ticker": row["ticker"],
                "pred_proba_up": row["pred_proba_up"],
                "future_return_5d": row["future_return_5d"],
                "profit": profit,
                "balance": balance
            })

print(f"\n🔎 Backtesting complete. Final balance: ${balance:.2f}")

# Save trade log for analysis
trade_log_df = pd.DataFrame(trade_log)
trade_log_df.to_csv("data/backtest_trade_log.csv", index=False)
print("Trade log saved to data/backtest_trade_log.csv")
