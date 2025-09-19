import os
import pandas as pd
import numpy as np

RAW_FILE = os.path.join("data", "raw", "prices_1d_2020-01-01_latest.csv")
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
    df = df.sort_values(["Ticker", "Date"]).rename(columns={"Adj Close": "AdjClose"})
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]]

    # 2) compute features per ticker
    # Use a container to store processed groups
    processed_groups = []
    for ticker, group in df.groupby("Ticker"):
        g = group.copy()

        # Original Features
        g["LogClose"] = np.log(g["AdjClose"])
        g["Return"] = g["LogClose"].diff()
        g["MA7"] = g["AdjClose"].rolling(7).mean()
        g["MA30"] = g["AdjClose"].rolling(30).mean()
        g["Vol20"] = g["Return"].rolling(20).std()
        g["RSI14"] = compute_rsi(g["AdjClose"])

        # --- New Features ---
        # Bollinger Bands
        g["BB_Mid"] = g["AdjClose"].rolling(20).mean()
        g["BB_Std"] = g["AdjClose"].rolling(20).std()
        g["BB_Upper"] = g["BB_Mid"] + (2 * g["BB_Std"])
        g["BB_Lower"] = g["BB_Mid"] - (2 * g["BB_Std"])

        # MACD
        ema12 = g["AdjClose"].ewm(span=12, adjust=False).mean()
        ema26 = g["AdjClose"].ewm(span=26, adjust=False).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()

        # --- Target and Lagged Features ---
        # Define target: next day's return
        g["Target_Return"] = g["Return"].shift(-1)

        # Lag all features by 1 day so we use current data to predict the next day
        feature_cols = [
            "Return", "MA7", "MA30", "Vol20", "RSI14",
            "BB_Upper", "BB_Lower", "MACD", "MACD_Signal"
        ]
        for col in feature_cols:
            g[f"{col}_lag1"] = g[col].shift(1)

        processed_groups.append(g)

    # 3) combine and clean
    df_featured = pd.concat(processed_groups)
    
    # Select final columns (lagged features + target)
    final_cols = [
        "Date", "Ticker", "AdjClose", "Target_Return"
    ] + [f"{col}_lag1" for col in [
            "Return", "MA7", "MA30", "Vol20", "RSI14",
            "BB_Upper", "BB_Lower", "MACD", "MACD_Signal"
    ]]
    
    df_final = df_featured[final_cols].dropna()

    # 4) write to CSV
    df_final.to_csv(OUT_FILE, index=False)
    print(f"[feature_engineering] Saved {len(df_final):,} rows → {OUT_FILE}")


if __name__ == "__main__":
    main()
