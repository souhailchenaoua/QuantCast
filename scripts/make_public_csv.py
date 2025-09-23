import pandas as pd, os

src = r"data/processed/prices_features.csv"
dst = r"data/processed/prices_features_public.csv"

df = pd.read_csv(src, parse_dates=["Date"])
keep = ["AAPL","MSFT","BTC-USD","INTC"]   # 👈 add INTC here
df = df[df["Ticker"].isin(keep)].sort_values(["Ticker","Date"])
df = df.groupby("Ticker").tail(500)       # last 500 rows per ticker

os.makedirs(os.path.dirname(dst), exist_ok=True)
df.to_csv(dst, index=False)
print("✅ Wrote", dst, "rows:", len(df))