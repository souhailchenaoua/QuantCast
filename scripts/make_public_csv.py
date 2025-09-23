import os, pandas as pd

SRC = r"data/processed/prices_features.csv"
DST = r"data/processed/prices_features_public.csv"

df = pd.read_csv(SRC, parse_dates=["Date"])
keep = ["AAPL","MSFT","BTC-USD"]   # pick the tickers you’re OK publishing
df = df[df["Ticker"].isin(keep)].sort_values(["Ticker","Date"])
df = df.groupby("Ticker").tail(200)   # last N rows per ticker

os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_csv(DST, index=False)
print(f"✅ Wrote {DST} rows: {len(df)}")
