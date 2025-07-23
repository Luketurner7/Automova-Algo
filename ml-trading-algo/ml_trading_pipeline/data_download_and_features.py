import yfinance as yf
import pandas as pd
from tqdm import tqdm

tickers = ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "AAPL"]

def download_and_process(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        close = df['Close'].squeeze()  # <-- force 1D shape

        # Example: Calculate a simple moving average as a feature
        features = pd.DataFrame()
        features['Close'] = close
        features['SMA_5'] = close.rolling(window=5).mean()
        features['Ticker'] = ticker

        return features.dropna()

    except Exception as e:
        print(f"Error with {ticker}: {e}")
        return None

all_data = []

for ticker in tqdm(tickers, desc="Downloading and processing"):
    result = download_and_process(ticker)
    if result is not None:
        all_data.append(result)

if all_data:
    combined_df = pd.concat(all_data)
    combined_df.to_csv("features.csv", index=False)
    print("✅ Feature extraction complete.")
else:
    print("❌ No data processed.")
