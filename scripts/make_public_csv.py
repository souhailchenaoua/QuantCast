import os, pandas as pd

SRC = r"data/processed/prices_features.csv"
DST = r"data/processed/prices_features_public.csv"

df = pd.read_csv(SRC, parse_dates=["Date"])
keep = ["AAPL","MSFT","INTC","BTC-USD"]  # add any others you want visible
df = df[df["Ticker"].isin(keep)].sort_values(["Ticker","Date"])
df = df.groupby("Ticker").tail(1250)      # >=300 so training path runs

os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_csv(DST, index=False)
print(f'Wrote {DST} rows: {len(df)}')
