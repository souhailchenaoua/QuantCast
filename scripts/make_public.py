from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"

# ---- Settings ----
TICKERS = [
    "AAPL", "INTC", "MSFT", "GOOGL", "TSLA",
    "AMZN", "NVDA", "META", "BTC-USD", "ETH-USD",
]

FEATURES_SRC = DATA_PROCESSED / "prices_features.csv"
FEATURES_DST = DATA_PROCESSED / "prices_features_public.csv"

ARIMA_SRC = DATA_PROCESSED / "arima_results.csv"
ARIMA_DST = DATA_PROCESSED / "arima_results_public.csv"


def build_public_features() -> None:
    if not FEATURES_SRC.exists():
        print(f"[features] missing {FEATURES_SRC}, skipping")
        return

    df = pd.read_csv(FEATURES_SRC, parse_dates=["Date"])

    # Basic sanity checks
    required_cols = {"Ticker", "Date"}
    if not required_cols.issubset(df.columns):
        print(f"[features] missing required columns {required_cols}, got {df.columns.tolist()}")
        return

    df = (
        df[df["Ticker"].isin(TICKERS)]
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker")
        .tail(1000)
        .reset_index(drop=True)
    )

    FEATURES_DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_DST, index=False)
    print(f"[features] wrote {FEATURES_DST} rows: {len(df)}")


def build_public_arima() -> None:
    if not ARIMA_SRC.exists():
        print(f"[arima] missing {ARIMA_SRC}, skipping")
        return

    ar = pd.read_csv(ARIMA_SRC)

    # Rename with ignore so it doesn't blow up if names already changed
    ar = ar.rename(
        columns={
            "Date": "ForecastDate",
            "PredClose": "PredictedClose",
        }
    )

    ARIMA_DST.parent.mkdir(parents=True, exist_ok=True)
    ar.to_csv(ARIMA_DST, index=False)
    print(f"[arima] wrote {ARIMA_DST} rows: {len(ar)}")


def main() -> None:
    build_public_features()
    build_public_arima()


if __name__ == "__main__":
    main()
