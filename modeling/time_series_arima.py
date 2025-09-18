# modeling/time_series_arima.py
import os
import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_FILE = os.path.join("data", "raw", "prices_1d_2023-01-01_latest.csv")
TEST_FRAC = 0.2
AIC_GRID = {
    "p": [0, 1, 2, 3],
    "d": [0, 1],
    "q": [0, 1, 2, 3],
}

warnings.filterwarnings("ignore")


def pick_order(series: pd.Series) -> Tuple[int, int, int]:
    n = len(series)
    split = int(n * (1 - TEST_FRAC))
    train = series.iloc[:split]
    best = (None, np.inf)
    for p in AIC_GRID["p"]:
        for d in AIC_GRID["d"]:
            for q in AIC_GRID["q"]:
                try:
                    m = ARIMA(
                        train,
                        order=(p, d, q),
                        trend="n",  # returns usually mean ~ 0
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    r = m.fit(method_kwargs={"warn_convergence": False})
                    if r.aic < best[1]:
                        best = ((p, d, q), r.aic)
                except Exception:
                    continue
    return best[0] or (1, 0, 0)


def fit_arima(series: pd.Series, order: Tuple[int, int, int]):
    # try without constant first (better for returns), then with constant
    for trend in ("n", "c"):
        try:
            model = ARIMA(
                series,
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(method_kwargs={"warn_convergence": False})
            return res
        except Exception:
            continue
    # fallback to AR(1) no-constant
    model = ARIMA(
        series,
        order=(1, 0, 0),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(method_kwargs={"warn_convergence": False})


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def evaluate_arima(series: pd.Series, order: Tuple[int, int, int]) -> dict:
    n = len(series)
    split = int(n * (1 - TEST_FRAC))
    train, test = series.iloc[:split], series.iloc[split:]

    res = fit_arima(train, order)
    fc = pd.Series(res.forecast(steps=len(test)), index=test.index)

    comp = pd.DataFrame({"y": test, "yhat": fc}).dropna()
    if comp.empty:
        # final fallback: AR(1) no-constant
        res2 = fit_arima(train, (1, 0, 0))
        fc2 = pd.Series(res2.forecast(steps=len(test)), index=test.index)
        comp = pd.DataFrame({"y": test, "yhat": fc2}).dropna()
        if comp.empty:
            return {
                "order": order,
                "mae": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "direction_acc": np.nan,
                "n_train": len(train),
                "n_test": len(test),
            }

    mae = mean_absolute_error(comp["y"], comp["yhat"])
    rmse = np.sqrt(mean_squared_error(comp["y"], comp["yhat"]))
    denom = comp["y"].replace(0, np.nan)
    mape = (np.abs((comp["y"] - comp["yhat"]) / denom)).dropna().mean() * 100.0
    dir_acc = direction_accuracy(comp["y"].values, comp["yhat"].values)

    return {
        "order": order,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "direction_acc": dir_acc,
        "n_train": len(train),
        "n_test": len(test),
    }


def main():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Raw data file not found at {RAW_FILE}. Run data_collection.py first.")

    df = pd.read_csv(RAW_FILE, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # choose price column
    price_col = "Adj Close" if "Adj Close" in df.columns else ("AdjClose" if "AdjClose" in df.columns else "Close")
    if price_col not in df.columns:
        raise ValueError("Could not find a price column among ['Adj Close','AdjClose','Close'].")

    print(f"[ARIMA] Using price column: {price_col}")
    tickers = df["Ticker"].unique()
    results = []

    for t in tickers:
        g = df[df["Ticker"] == t].copy()
        g["LogClose"] = np.log(g[price_col])
        g["Ret"] = g["LogClose"].diff()
        g = g.dropna(subset=["Ret"])

        if len(g) < 60:
            print(f"[ARIMA][{t}] Not enough data ({len(g)} rows). Skipping.")
            continue

        y = g["Ret"]

        order = pick_order(y)
        print(f"[ARIMA][{t}] Selected order (p,d,q) = {order}")

        metrics = evaluate_arima(y, order)
        metrics["ticker"] = t
        results.append(metrics)

        print(
            f"[ARIMA][{t}] n_train={metrics['n_train']}, n_test={metrics['n_test']}\n"
            f"[ARIMA][{t}] MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
            f"MAPE={metrics['mape']:.2f}%, DirAcc={metrics['direction_acc']:.3f}"
        )

    if results:
        out = pd.DataFrame(results).sort_values(["ticker"])
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        out_path = os.path.join("data", "processed", "arima_results.csv")
        out.to_csv(out_path, index=False)
        print(f"\n[ARIMA] Saved results → {out_path}")
        print(out)
    else:
        print("[ARIMA] No results computed. Check data availability.")


if __name__ == "__main__":
    main()
