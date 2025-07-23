import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ssl

from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

ssl._create_default_https_context = ssl._create_unverified_context

# Get S&P 500 tickers
try:
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tickers = pd.read_html(sp500_url)[0]['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]
except Exception as e:
    print(f"❌ Error loading ticker list: {e}")
    tickers = []

tickers = tickers[:50]

start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
data_dict = {}

for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)

        if df.empty or len(df) < 100:
            print(f"⚠️ Skipping {ticker}: insufficient data")
            continue

        df.dropna(inplace=True)

        # Technical indicators (force flattening to 1D Series)
        df['sma_20'] = SMAIndicator(close=df["Close"], window=20).sma_indicator().astype(float)
        df['rsi'] = RSIIndicator(close=df["Close"], window=14).rsi().astype(float)
        macd = MACD(close=df["Close"])
        df['macd'] = macd.macd().astype(float)
        df['macd_signal'] = macd.macd_signal().astype(float)
        bb = BollingerBands(close=df["Close"])
        df['bb_upper'] = bb.bollinger_hband().astype(float)
        df['bb_lower'] = bb.bollinger_lband().astype(float)

        # Targets
        df["future_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1
        df["target"] = (df["future_return_5d"] > 0).astype(int)
        df["ticker"] = ticker
        df["date"] = df.index

        data_dict[ticker] = df

        print(f"✅ Processed: {ticker}")

    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")

# Save
if data_dict:
    df_all = pd.concat(data_dict.values())
    df_all.reset_index(drop=True, inplace=True)
    output_path = "data/processed_stock_data.csv"
    df_all.to_csv(output_path, index=False)
    print(f"\n✅ All data saved to {output_path} — {len(df_all)} rows.")
else:
    print("❌ No data was downloaded successfully.")
