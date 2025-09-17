import os
import pandas as pd
import numpy as np

RAW_FILE = os.path.join("data", "raw", "prices_1d_2023-01-01_latest.csv")
OUT_DIR = os.path.join("data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "prices_features.csv")

def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    # Wilder's smoothing (EMA-like)
    roll_up = pd.Series(gain, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_dn = pd.Series(loss, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-9)
    return 100 - (100 / (1 + rs))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_FILE, parse_dates=["Date"])

    # 1) sort & keep Adj Close
    df = df.sort_values(["Ticker","Date"]).rename(columns={"Adj Close":"AdjClose"})

    # 2) log return per ticker
    df["LogClose"] = np.log(df["AdjClose"])
    df["Return"]   = df.groupby("Ticker")["LogClose"].diff()
    # 3) features per ticker
    df["MA7"]   = df.groupby("Ticker")["AdjClose"].transform(lambda x: x.rolling(7).mean())
    df["MA30"]  = df.groupby("Ticker")["AdjClose"].transform(lambda x: x.rolling(30).mean())
    df["Vol20"] = df.groupby("Ticker")["Return"].transform(lambda x: x.rolling(20).std())
    df["RSI14"] = df.groupby("Ticker")["AdjClose"].transform(compute_rsi)

    # 4) clean
    keep_cols = ["Date","Ticker","Open","High","Low","Close","AdjClose","Volume",
                 "Return","MA7","MA30","Vol20","RSI14"]
    df = df[keep_cols].dropna()

    # 5) write to CSV
    df.to_csv(OUT_FILE, index=False)
    print(f"[feature_engineering] Saved {len(df):,} rows → {OUT_FILE}")


if __name__ == "__main__":
    main()
