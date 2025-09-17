# -*- coding: utf-8 -*-
"""
Day 1 - Data Collection
Pull historical OHLCV for 1..N tickers from Yahoo Finance and save to CSV.
Requires: pip install yfinance pandas
"""
import os
import sys
import time
from typing import List, Optional
import pandas as pd
import yfinance as yf

# -------- Settings --------
TICKERS: List[str] = ["AAPL", "BTC-USD"]  # edit as you like
START_DATE: str = "2023-01-01"
END_DATE: Optional[str] = None      # None => latest
INTERVAL: str = "1d"                # "1d", "1h", "30m", etc.
OUT_DIR: str = os.path.join("data", "raw")
COMBINED_FILENAME: str = f"prices_{INTERVAL}_{(START_DATE or 'min')}_{(END_DATE or 'latest')}.csv"
MAX_RETRIES: int = 3
SLEEP_BETWEEN: float = 1.0  # seconds

def log(msg: str) -> None:
    print(f"[data_collection] {msg}", flush=True)

def safe_download(ticker: str) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",   # 🔑 ask for flat columns
            )

            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df = df.copy()
                df.index.name = "Date"

                # 🔧 If still MultiIndex, flatten to first level
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Ensure string column names
                df.columns = [str(c) for c in df.columns]

                # Now safely add the ticker column
                df["Ticker"] = ticker

                # Keep only rows where not all OHLCV are NA
                ohclv = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
                subset = ohclv if ohclv else [c for c in df.columns if c.lower() != "ticker"]
                df = df.dropna(how="all", subset=subset)
                return df

            raise ValueError("Empty DataFrame returned.")
        except Exception as e:
            last_exc = e
            log(f"{ticker}: attempt {attempt}/{MAX_RETRIES} failed -> {e}")
            time.sleep(SLEEP_BETWEEN)

    raise RuntimeError(f"Failed to download {ticker} after {MAX_RETRIES} retries") from last_exc

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for t in TICKERS:
        log(f"Downloading {t} ({START_DATE} to {END_DATE or 'latest'}, {INTERVAL})...")
        df = safe_download(t)
        path = os.path.join(OUT_DIR, f"{t}_{INTERVAL}_{START_DATE}_{(END_DATE or 'latest')}.csv")
        df.to_csv(path)
        log(f"Saved -> {path} ({len(df):,} rows)")
        frames.append(df)

    combined = pd.concat(frames, axis=0).reset_index()
    # Nice column order if present
    preferred = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = [c for c in preferred if c in combined.columns] + [c for c in combined.columns if c not in preferred]
    combined = combined[cols]

    combined_path = os.path.join(OUT_DIR, COMBINED_FILENAME)
    combined.to_csv(combined_path, index=False)
    log(f"Saved combined -> {combined_path} ({len(combined):,} rows)")

    # Quick checks
    log("Quick checks:")
    log(f"- Date range: {combined['Date'].min()} -> {combined['Date'].max()}")
    log(f"- Tickers: {sorted(combined['Ticker'].unique())}")
    log(f"- Missing values per column:\n{combined.isna().sum().to_string()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)
