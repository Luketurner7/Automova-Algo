import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from datetime import datetime, timedelta
import ssl

# Fix SSL issues (on macOS)
ssl._create_default_https_context = ssl._create_unverified_context

# Step 1: Get S&P 500 tickers
try:
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tickers = pd.read_html(sp500_url)[0]['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]
except Exception as e:
    print(f"❌ Error loading ticker list: {e}")
    tickers = []

# Limit to first 50
tickers = tickers[:50]

# Step 2: Download data
start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
data_dict = {}

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)

        if df.empty or len(df) < 100:
            print(f"⚠️ Skipping {ticker}: insufficient data")
            continue

        df = dropna(df)

        # Ensure no multi-dimensional columns are created
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        df["future_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1
        df["target"] = (df["future_return_5d"] > 0).astype(int)
        df["ticker"] = ticker
        df["date"] = df.index

        data_dict[ticker] = df

        print(f"✅ Processed: {ticker}")

    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")

# Step 3: Save
if data_dict:
    df_all = pd.concat(data_dict.values())
    df_all.reset_index(drop=True, inplace=True)

    output_path = "data/processed_stock_data.csv"
    df_all.to_csv(output_path, index=False)
    print(f"\n✅ All data saved to {output_path} — {len(df_all)} rows.")
else:
    print("❌ No data was downloaded successfully.")
