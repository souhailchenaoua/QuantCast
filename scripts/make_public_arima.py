import os, pandas as pd

SRC = r"data/processed/arima_results.csv"          # adjust if your ARIMA outputs elsewhere
DST = r"data/processed/arima_results_public.csv"

# expected columns: Ticker, ForecastDate, PredictedClose (or whatever you wrote)
df = pd.read_csv(SRC, parse_dates=["ForecastDate","Date"], infer_datetime_format=True) \
        if os.path.exists(SRC) else pd.DataFrame()

if df.empty:
    # Fallback: synthesize 1-step-ahead from features file if needed
    fpath = r"data/processed/prices_features.csv"
    dff = pd.read_csv(fpath, parse_dates=["Date"])
    keep = ["AAPL","INTC"]            # choose your public tickers
    dff = dff[dff["Ticker"].isin(keep)].sort_values(["Ticker","Date"])
    df = dff.groupby("Ticker").tail(1).copy()
    df["ForecastDate"] = df["Date"] + pd.tseries.offsets.BDay(1)
    df.rename(columns={"AdjClose":"PredictedClose"}, inplace=True)
    df = df[["Ticker","ForecastDate","PredictedClose"]]

# keep last row per ticker (next forecast)
df = df.sort_values(["Ticker","ForecastDate"]).groupby("Ticker").tail(1)
os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_csv(DST, index=False)
print(f"✅ Wrote {DST} rows: {len(df)}")
