# ml_trading_pipeline/data_download_and_features.py

import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from datetime import datetime, timedelta

# Optional: Fix SSL certificate issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Step 1: Get S&P 500 Tickers
try:
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(sp500_url)
    tickers = tables[0]['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]  # Adjust for yfinance
except Exception as e:
    print(f"❌ Failed to fetch S&P 500 tickers: {e}")
    tickers = []

# Step 2: Limit to first 50 for now
tickers = tickers[:50]

# Step 3: Download daily data (last 2 years)
start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
data_dict = {}

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
        if df.empty or len(df) < 100:
            print(f"⚠️ Skipping {ticker}: not enough data")
            continue

        df = dropna(df)
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        df["future_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1
        df["target"] = (df["future_return_5d"] > 0).astype(int)
        df["ticker"] = ticker
        df["date"] = df.index

        data_dict[ticker] = df

        print(f"✅ {ticker} data processed")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Step 4: Combine and save
if data_dict:
    df_all = pd.concat(data_dict.values())
    df_all.reset_index(drop=True, inplace=True)

    # Save to CSV
    output_path = "data/processed_stock_data.csv"
    df_all.to_csv(output_path, index=False)
    print(f"\n✅ Data saved to {output_path} with {len(df_all)} rows")
else:
    print("❌ No data downloaded. Check your internet or ticker list.")
