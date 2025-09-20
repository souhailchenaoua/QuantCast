# feature_engineering.py
import os
import pandas as pd
import numpy as np

# Optional: you imported ta before, but we don't strictly need it here.
# import ta

RAW_FILE = os.path.join("data", "raw", "prices_1d_2020-01-01_latest.csv")
OUT_DIR = os.path.join("data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "prices_features.csv")

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing."""
    delta = series.diff()
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    roll_dn = pd.Series(loss, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-9)
    return 100 - (100 / (1 + rs))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_FILE, parse_dates=["Date"])

    # 1) Sort & keep necessary columns
    df = df.sort_values(["Ticker", "Date"]).rename(columns={"Adj Close": "AdjClose"})
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]]

    # 2) Compute features per ticker
    processed_groups = []
    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.copy()

        # --- Core returns & trend/vol features ---
        g["LogClose"] = np.log(g["AdjClose"])
        g["Return"]   = g["LogClose"].diff()
        g["MA7"]      = g["AdjClose"].rolling(7).mean()
        g["MA30"]     = g["AdjClose"].rolling(30).mean()
        g["Vol20"]    = g["Return"].rolling(20).std()
        g["RSI14"]    = compute_rsi(g["AdjClose"])

        # --- Bollinger Bands ---
        g["BB_Mid"]   = g["AdjClose"].rolling(20).mean()
        g["BB_Std"]   = g["AdjClose"].rolling(20).std()
        g["BB_Upper"] = g["BB_Mid"] + (2 * g["BB_Std"])
        g["BB_Lower"] = g["BB_Mid"] - (2 * g["BB_Std"])

        # --- MACD (12,26,9) ---
        ema12         = g["AdjClose"].ewm(span=12, adjust=False).mean()
        ema26         = g["AdjClose"].ewm(span=26, adjust=False).mean()
        g["MACD"]     = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["MACD_Hist"]   = g["MACD"] - g["MACD_Signal"]

        # --- Gaps / intraday structure ---
        g["PrevClose"]    = g["AdjClose"].shift(1)
        g["OvernightGap"] = (g["Open"] / g["PrevClose"]) - 1
        g["Range"]        = (g["High"] - g["Low"]) / g["AdjClose"]  # ATR-lite

        # --- Extra return memory ---
        for k in [2, 3, 4, 5]:
            g[f"Return_lag{k}"] = g["Return"].shift(k)

        # --- Day-of-week one-hots (0=Mon..6=Sun); we keep 0..4 to cover weekdays ---
        g["DOW"] = g["Date"].dt.dayofweek
        for d in range(5):
            g[f"DOW_{d}"] = (g["DOW"] == d).astype(int)

        # --- Target: next day's return ---
        g["Target_Return"] = g["Return"].shift(-1)

        # --- Lag all model features by 1 day (to predict next day) ---
        feature_cols = [
            # Originals
            "Return", "MA7", "MA30", "Vol20", "RSI14",
            "BB_Upper", "BB_Lower", "MACD", "MACD_Signal",
            # New signals
            "OvernightGap", "Range", "MACD_Hist",
            "Return_lag2", "Return_lag3", "Return_lag4", "Return_lag5",
            "DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4"
        ]
        for col in feature_cols:
            g[f"{col}_lag1"] = g[col].shift(1)

        processed_groups.append(g)

    # 3) Combine and select final columns
    df_featured = pd.concat(processed_groups, axis=0)

    final_cols = [
        "Date", "Ticker", "AdjClose", "Target_Return"
    ] + [f"{col}_lag1" for col in [
        "Return", "MA7", "MA30", "Vol20", "RSI14",
        "BB_Upper", "BB_Lower", "MACD", "MACD_Signal",
        "OvernightGap", "Range", "MACD_Hist",
        "Return_lag2", "Return_lag3", "Return_lag4", "Return_lag5",
        "DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4"
    ]]

    # Drop rows with NA introduced by rolling/lagging
    df_final = df_featured[final_cols].dropna().reset_index(drop=True)

    # 4) Write to CSV
    df_final.to_csv(OUT_FILE, index=False)
    print(f"[feature_engineering] Saved {len(df_final):,} rows → {OUT_FILE}")

if __name__ == "__main__":
    main()
