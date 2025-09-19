# modeling/time_series_arima.py
import os
import random
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# Config
# -----------------------------
RAW_FILE = os.path.join("data", "raw", "prices_1d_2020-01-01_latest.csv")

# Final test split for metrics (order selection uses an internal val split on TRAIN ONLY)
TEST_FRAC = 0.2

# Candidate orders for selection
AIC_GRID = {
    "p": [0, 1, 2],
    "d": [0],
    "q": [0, 1, 2],
}

# Save the per-ticker pdq grid scoring table?
SAVE_GRID_SCORES = True

# Also compute price-level metrics by re-integrating predicted returns to prices?
COMPUTE_PRICE_METRICS = True

# Reproducibility
warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)


# -----------------------------
# Utils
# -----------------------------
def _clean_series(s: pd.Series) -> pd.Series:
    """Coerce to float, drop inf/NaN."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)


def _safe_forecast(res, steps: int, index) -> pd.Series:
    """Robust forecast that tries statespace path and cleans non-finites."""
    try:
        fc = res.get_forecast(steps=steps).predicted_mean
        fc.index = index
    except Exception:
        try:
            fc = pd.Series(res.forecast(steps=steps), index=index)
        except Exception:
            fc = pd.Series([np.nan] * steps, index=index)
    return pd.to_numeric(fc, errors="coerce").replace([np.inf, -np.inf], np.nan)


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def fit_arima(series: pd.Series, order: Tuple[int, int, int]):
    """Fit ARIMA with a small fallback ladder."""
    series = _clean_series(series)

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

    # backoff orders commonly stable on returns
    for alt_order in [(1, 0, 0), (1, 0, 1)]:
        try:
            model = ARIMA(
                series,
                order=alt_order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit(method_kwargs={"warn_convergence": False})
        except Exception:
            continue

    # last resort
    model = ARIMA(
        series, order=(1, 0, 0), trend="n",
        enforce_stationarity=False, enforce_invertibility=False
    )
    return model.fit(method_kwargs={"warn_convergence": False})


# -----------------------------
# Order selection (with validation on TRAIN ONLY)
# -----------------------------
@dataclass
class _OrderScore:
    p: int
    d: int
    q: int
    aic: float
    bic: float
    rmse: float
    diracc: float


def pick_order_with_validation(
    series: pd.Series,
    grid=AIC_GRID,
    val_frac: float = 0.2,
    tie_break: str = "bic",        # or "aic"
    ticker: Optional[str] = None,
    save_scores_csv: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Choose (p,d,q) by maximizing validation DirAcc; tie-break by BIC/AIC, then RMSE.
    Trains on train_core; validates on last val_frac of the *training* series.
    """
    s = _clean_series(series)
    n = len(s)
    if n < 80:  # tiny guard
        return (1, 0, 0)

    split_val = int(n * (1 - val_frac))
    train_core, val = s.iloc[:split_val], s.iloc[split_val:]

    scores: List[_OrderScore] = []
    for p in grid["p"]:
        for d in grid["d"]:
            for q in grid["q"]:
                try:
                    r = ARIMA(
                        train_core, order=(p, d, q), trend="n",
                        enforce_stationarity=False, enforce_invertibility=False
                    ).fit(method_kwargs={"warn_convergence": False})
                    fc = _safe_forecast(r, len(val), val.index)
                    comp = pd.DataFrame({"y": val, "yhat": fc}).dropna()
                    if comp.empty:
                        continue
                    rmse = float(np.sqrt(mean_squared_error(comp["y"], comp["yhat"])))
                    diracc = direction_accuracy(comp["y"].values, comp["yhat"].values)
                    aic = float(r.aic) if np.isfinite(r.aic) else np.inf
                    bic = float(r.bic) if np.isfinite(r.bic) else np.inf
                    scores.append(_OrderScore(p, d, q, aic, bic, rmse, diracc))
                except Exception:
                    continue

    if not scores:
        return (1, 0, 0)

    # Rank: higher DirAcc, then lower BIC/AIC, then lower RMSE
    if tie_break == "bic":
        scores.sort(key=lambda s: (-s.diracc, s.bic, s.rmse))
    else:
        scores.sort(key=lambda s: (-s.diracc, s.aic, s.rmse))

    if save_scores_csv:
        out = pd.DataFrame([s.__dict__ for s in scores])
        out["ticker"] = ticker
        os.makedirs(os.path.dirname(save_scores_csv), exist_ok=True)
        out.to_csv(save_scores_csv, index=False)

    best = scores[0]
    return (best.p, best.d, best.q)


# -----------------------------
# Evaluation (train/test explicitly passed to avoid leakage)
# -----------------------------
def _price_metrics_from_returns(
    comp_returns: pd.DataFrame,
    log_price: pd.Series,
    last_train_log: float,
) -> Tuple[float, float, float]:
    """Compute MAE/RMSE/MAPE on price levels from return predictions."""
    pred_log = last_train_log + comp_returns["yhat"].cumsum()
    act_log = log_price.loc[comp_returns.index]
    pred_price = np.exp(pred_log)
    act_price = np.exp(act_log)
    price_mae = mean_absolute_error(act_price, pred_price)
    price_rmse = np.sqrt(mean_squared_error(act_price, pred_price))
    price_mape = (np.abs((act_price - pred_price) / act_price)).dropna().mean() * 100.0
    return float(price_mae), float(price_rmse), float(price_mape)


def evaluate_arima_split(
    train: pd.Series,
    test: pd.Series,
    order: Tuple[int, int, int],
    log_price: Optional[pd.Series] = None,
    compute_price_metrics: bool = False,
) -> dict:
    """Fit on train, forecast test; optional price metrics via re-integration."""
    train = _clean_series(train)
    test = _clean_series(test)

    res = fit_arima(train, order)
    fc = _safe_forecast(res, len(test), test.index)

    # fallback attempts if forecast is all-NaN
    if fc.isna().all():
        try:
            res2 = ARIMA(
                train, order=order, trend="c",
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(method_kwargs={"warn_convergence": False})
            fc = _safe_forecast(res2, len(test), test.index)
        except Exception:
            pass

    if fc.isna().all():
        for alt_order in [(1, 0, 0), (1, 0, 1)]:
            try:
                res3 = fit_arima(train, alt_order)
                fc = _safe_forecast(res3, len(test), test.index)
                if not fc.isna().all():
                    break
            except Exception:
                continue

    comp = pd.DataFrame({"y": test, "yhat": fc}).dropna()
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

    out = {
        "order": order,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "direction_acc": float(dir_acc),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    # Optional price-level metrics
    if compute_price_metrics and (log_price is not None):
        try:
            last_train_log = float(log_price.loc[train.index[-1]])
            price_mae, price_rmse, price_mape = _price_metrics_from_returns(comp, log_price, last_train_log)
            out.update({
                "price_mae": price_mae,
                "price_rmse": price_rmse,
                "price_mape": price_mape,
            })
        except Exception:
            # If alignment fails, skip price metrics gracefully
            pass

    return out


# -----------------------------
# Baselines (for sanity)
# -----------------------------
def evaluate_naive_baselines(
    train: pd.Series,
    test: pd.Series,
    log_price: Optional[pd.Series] = None,
    compute_price_metrics: bool = False,
) -> dict:
    """
    Two simple baselines on returns:
      - zero: predict 0 return (random-walk on log-price)
      - last: predict the last observed train return for all test steps
    """
    train = _clean_series(train)
    test = _clean_series(test)

    # Baseline 1: zero returns
    zero_hat = pd.Series(0.0, index=test.index)
    comp_zero = pd.DataFrame({"y": test, "yhat": zero_hat})
    zero_mae = mean_absolute_error(comp_zero["y"], comp_zero["yhat"])
    zero_rmse = np.sqrt(mean_squared_error(comp_zero["y"], comp_zero["yhat"]))
    zero_dir = direction_accuracy(comp_zero["y"].values, comp_zero["yhat"].values)  # usually ~0 unless zeros present

    out = {
        "baseline_zero_mae": float(zero_mae),
        "baseline_zero_rmse": float(zero_rmse),
        "baseline_zero_diracc": float(zero_dir),
    }

    # Baseline 2: last return persistence
    last_r = float(train.iloc[-1]) if len(train) else 0.0
    last_hat = pd.Series(last_r, index=test.index)
    comp_last = pd.DataFrame({"y": test, "yhat": last_hat})
    last_mae = mean_absolute_error(comp_last["y"], comp_last["yhat"])
    last_rmse = np.sqrt(mean_squared_error(comp_last["y"], comp_last["yhat"]))
    last_dir = direction_accuracy(comp_last["y"].values, comp_last["yhat"].values)

    out.update({
        "baseline_last_mae": float(last_mae),
        "baseline_last_rmse": float(last_rmse),
        "baseline_last_diracc": float(last_dir),
    })

    # Optional price metrics for baselines
    if compute_price_metrics and (log_price is not None):
        try:
            last_train_log = float(log_price.loc[train.index[-1]])
            # zero baseline → flat log path
            price_mae0, price_rmse0, price_mape0 = _price_metrics_from_returns(comp_zero, log_price, last_train_log)
            # last baseline → constant return path
            price_maeL, price_rmseL, price_mapeL = _price_metrics_from_returns(comp_last, log_price, last_train_log)
            out.update({
                "baseline_zero_price_mae": price_mae0,
                "baseline_zero_price_rmse": price_rmse0,
                "baseline_zero_price_mape": price_mape0,
                "baseline_last_price_mae": price_maeL,
                "baseline_last_price_rmse": price_rmseL,
                "baseline_last_price_mape": price_mapeL,
            })
        except Exception:
            pass

    return out


# -----------------------------
def _fmt_num(x, d=6):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return "—"

def _fmt_pct(x, d=1):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{100.0*float(x):.{d}f}%"
    except Exception:
        return "—"

def _has_price_metrics(m):
    return all(k in m for k in ["price_mae", "price_rmse", "price_mape"])

def _print_ticker_report(ticker, order, m):
    # Header
    print(f"\n┌─ {ticker} ─────────────────────────────────────────────────────────")
    print(f"│ Model: ARIMA{order} on returns   |  Split: train={m['n_train']}, test={m['n_test']}")

    # Returns-level
    ret_line = (
        f"│ Returns (test) → "
        f"MAE={_fmt_num(m.get('mae'), 6)}, "
        f"RMSE={_fmt_num(m.get('rmse'), 6)}, "
        f"DirAcc={_fmt_pct(m.get('direction_acc'), 1)}"
    )
    base_line = (
        f"│ Baselines       "
        f"zero: MAE={_fmt_num(m.get('baseline_zero_mae'), 6)}, RMSE={_fmt_num(m.get('baseline_zero_rmse'), 6)}, DirAcc={_fmt_pct(m.get('baseline_zero_diracc'), 1)}  |  "
        f"last: MAE={_fmt_num(m.get('baseline_last_mae'), 6)}, RMSE={_fmt_num(m.get('baseline_last_rmse'), 6)}, DirAcc={_fmt_pct(m.get('baseline_last_diracc'), 1)}"
    )
    print(ret_line)
    print(base_line)

    # Price-level (optional)
    if _has_price_metrics(m):
        price_line = (
            f"│ Prices (reintegrated) → "
            f"MAE={_fmt_num(m.get('price_mae'), 4)}, "
            f"RMSE={_fmt_num(m.get('price_rmse'), 4)}, "
            f"MAPE={_fmt_num(m.get('price_mape'), 2)}%"
        )
        zb = "baseline_zero_"
        if all(k in m for k in [zb+"price_mae", zb+"price_rmse", zb+"price_mape"]):
            price_base = (
                f"│ Random-walk base  → "
                f"MAE={_fmt_num(m.get(zb+'price_mae'), 4)}, "
                f"RMSE={_fmt_num(m.get(zb+'price_rmse'), 4)}, "
                f"MAPE={_fmt_num(m.get(zb+'price_mape'), 2)}%"
            )
            print(price_line)
            print(price_base)
        else:
            print(price_line)

    print("└────────────────────────────────────────────────────────────────────")

# -----------------------------
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
    summary_rows = []

    for t in tickers:
        g = df[df["Ticker"] == t].copy()
        g["LogClose"] = np.log(g[price_col])
        g["Ret"] = g["LogClose"].diff()
        g = g.dropna(subset=["Ret"])

        if len(g) < 60:
            print(f"[ARIMA][{t}] Not enough data ({len(g)} rows). Skipping.")
            continue

        # Ensure Date index for alignment
        if "Date" in g.columns:
            g = g.set_index("Date")

        # cleaned returns
        y_full = _clean_series(g["Ret"])
        if len(y_full) < 60:
            print(f"[ARIMA][{t}] Not enough data ({len(y_full)} rows). Skipping.")
            continue

        # final train/test split (no leakage)
        split = int(len(y_full) * (1 - TEST_FRAC))
        y_train, y_test = y_full.iloc[:split], y_full.iloc[split:]
        log_price = g["LogClose"]

        # pick order on TRAIN ONLY (validation inside)
        grid_csv = os.path.join("data", "processed", f"arima_grid_{t}.csv") if SAVE_GRID_SCORES else None
        order = pick_order_with_validation(
            y_train, grid=AIC_GRID, val_frac=0.2, tie_break="bic",
            ticker=t, save_scores_csv=grid_csv
        )

        # evaluate model and baselines
        metrics = evaluate_arima_split(
            y_train, y_test, order,
            log_price=log_price, compute_price_metrics=COMPUTE_PRICE_METRICS,
        )
        base = evaluate_naive_baselines(
            y_train, y_test, log_price=log_price, compute_price_metrics=COMPUTE_PRICE_METRICS
        )

        # combine for output & persistence
        metrics["ticker"] = t
        metrics.update(base)
        results.append(metrics)

        # per-ticker readable block
        _print_ticker_report(t, order, metrics)

        # compact summary row
        summary_rows.append({
            "Ticker": t,
            "Order": str(order),
            "Ret_MAE": metrics.get("mae"),
            "Ret_RMSE": metrics.get("rmse"),
            "DirAcc": metrics.get("direction_acc"),
            "Base_last_DirAcc": metrics.get("baseline_last_diracc"),
            "Price_MAE": metrics.get("price_mae") if "price_mae" in metrics else np.nan,
            "Price_MAPE": metrics.get("price_mape") if "price_mape" in metrics else np.nan,
        })

    # Persist full metrics
    if results:
        out = pd.DataFrame(results).sort_values(["ticker"])
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        out_path = os.path.join("data", "processed", "arima_results.csv")
        out.to_csv(out_path, index=False)
        print(f"\n[ARIMA] Saved full metrics → {out_path}")

        # Print compact summary
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            # Format key columns for readability in console
            show = summary_df.copy()
            show["Ret_MAE"] = show["Ret_MAE"].apply(lambda v: _fmt_num(v, 6))
            show["Ret_RMSE"] = show["Ret_RMSE"].apply(lambda v: _fmt_num(v, 6))
            show["DirAcc"] = show["DirAcc"].apply(lambda v: _fmt_pct(v, 1))
            show["Base_last_DirAcc"] = show["Base_last_DirAcc"].apply(lambda v: _fmt_pct(v, 1))
            if "Price_MAE" in show.columns:
                show["Price_MAE"] = show["Price_MAE"].apply(lambda v: _fmt_num(v, 4))
            if "Price_MAPE" in show.columns:
                show["Price_MAPE"] = show["Price_MAPE"].apply(lambda v: (_fmt_num(v, 2) + "%") if pd.notnull(v) else "—")

            print("\nSUMMARY (per ticker)")
            print(show.to_string(index=False))
    else:
        print("[ARIMA] No results computed. Check data availability.")


if __name__ == "__main__":
    main()
