import yfinance as yf
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from datetime import datetime, timedelta
import os

# Step 1: Get S&P 500 Tickers
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tickers = pd.read_html(sp500_url)[0]['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers]  # Adjust for yfinance

# Step 2: Limit to first 50 for now (can expand later)
tickers = tickers[:50]

# Step 3: Download daily data (last 2 years)
start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
data_dict = {}

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if df.empty:
            print(f"Warning: No data for {ticker} after download")
            continue
        df = dropna(df)
        if df.empty:
            print(f"Warning: No data for {ticker} after dropna")
            continue
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )
        df["future_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1
        df["target"] = (df["future_return_5d"] > 0).astype(int)
        df["ticker"] = ticker
        df["date"] = df.index
        data_dict[ticker] = df
        print(f"Downloaded and processed {ticker} ({len(df)} rows)")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

if data_dict:
    df_all = pd.concat(data_dict.values())
    df_all.reset_index(drop=True, inplace=True)
    os.makedirs("data", exist_ok=True)
    df_all.to_csv("data/processed_stock_data.csv", index=False)
    print("✅ Data download and feature generation complete. Saved to data/processed_stock_data.csv")
else:
    print("No data downloaded; nothing to save.")
