import pandas as pd, os

# features -> public
feat_src = r"data/processed/prices_features.csv"
feat_dst = r"data/processed/prices_features_public.csv"
df = pd.read_csv(feat_src, parse_dates=["Date"])
keep = ["AAPL", "INTC", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "BTC-USD", "ETH-USD"] # edit as you like
df = df[df["Ticker"].isin(keep)].sort_values(["Ticker","Date"]).groupby("Ticker").tail(1000)
df.to_csv(feat_dst, index=False)
print(" wrote", feat_dst, "rows:", len(df))

# ARIMA -> public (accept a few column name variants)
ar_src = r"data/processed/arima_results.csv"
ar_dst = r"data/processed/arima_results_public.csv"
if os.path.exists(ar_src):
    ar = pd.read_csv(ar_src)
    ar = ar.rename(columns={"Date":"ForecastDate", "PredClose":"PredictedClose"})
    ar.to_csv(ar_dst, index=False)
    print(" wrote", ar_dst, "rows:", len(ar))
else:
    print(" missing", ar_src)
