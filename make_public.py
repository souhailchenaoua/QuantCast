import pandas as pd, os

src = r"data/processed/prices_features.csv"
dst = r"data/processed/prices_features_public.csv"

df = pd.read_csv(src, parse_dates=["Date"])
keep = ["AAPL","MSFT","INTC","BTC-USD"]
df = df[df["Ticker"].isin(keep)].sort_values(["Ticker","Date"])
df = df.groupby("Ticker").tail(1000)  # ensure >=300 rows for training
df.to_csv(dst, index=False)
print("✅ wrote", dst, "rows:", len(df))

# also copy ARIMA results if needed
arima_src = r"data/processed/arima_results.csv"
arima_dst = r"data/processed/arima_results_public.csv"
if os.path.exists(arima_src):
    pd.read_csv(arima_src).to_csv(arima_dst, index=False)
    print("✅ wrote", arima_dst)
