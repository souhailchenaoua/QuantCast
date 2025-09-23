# modeling/ml_direction_classifier.py
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from utils.utils_s3 import s3_upload_if_configured

DATA_PATH = os.getenv(
    "PRICES_FEATURES_PATH",
    os.path.join("data","processed","prices_features.csv")
)

PREFERRED_FEATURES = [
    "Return_lag1","MA7_lag1","MA30_lag1","Vol20_lag1","RSI14_lag1",
    "BB_Upper_lag1","BB_Lower_lag1","MACD_lag1","MACD_Signal_lag1",
    "OvernightGap_lag1","Range_lag1","MACD_Hist_lag1",
    "Return_lag2_lag1","Return_lag3_lag1","Return_lag4_lag1","Return_lag5_lag1",
    "DOW_0_lag1","DOW_1_lag1","DOW_2_lag1","DOW_3_lag1","DOW_4_lag1",
]

# ---------- utils ----------
def select_features(df: pd.DataFrame) -> List[str]:
    cols = [c for c in PREFERRED_FEATURES if c in df.columns]
    if cols:
        return cols
    bad = {"Target_Return", "y"}
    return sorted(
        c for c in df.columns
        if c.endswith("_lag1") and c not in bad and np.issubdtype(df[c].dtype, np.number)
    )

def train_val_split_time(X: np.ndarray, y: np.ndarray, val_frac: float = 0.15) -> Tuple:
    n = len(X)
    cut = int((1 - val_frac) * n)
    # time-ordered split: [0:cut) = train, [cut:n) = val
    return X[:cut], X[cut:], y[:cut], y[cut:]

def threshold_max_balacc(p_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Pick the probability threshold that maximizes balanced accuracy on VALIDATION.
    """
    grid = np.clip(np.linspace(0.15, 0.85, 301), 1e-3, 1 - 1e-3)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        y_hat = (p_val >= t).astype(int)
        s = balanced_accuracy_score(y_val, y_hat)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out

def evaluate_hard(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out

def fmt_metrics_row(ticker: str, model: str, thr: float, metrics: Dict[str, float]) -> Dict:
    return {
        "Ticker": ticker,
        "Model": model,
        "thr": None if (thr is None or (isinstance(thr, float) and np.isnan(thr))) else round(thr, 3),
        "acc": round(metrics["acc"], 6),
        "bal_acc": round(metrics["bal_acc"], 6),
        "prec": round(metrics["prec"], 6),
        "rec": round(metrics["rec"], 6),
        "f1": round(metrics["f1"], 6),
        "roc_auc": round(metrics["roc_auc"], 6) if not np.isnan(metrics["roc_auc"]) else np.nan,
        "tn": int(metrics["tn"]), "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]), "tp": int(metrics["tp"]),
    }

# ---------- per-ticker ----------
def run_for_ticker(df_ticker: pd.DataFrame) -> pd.DataFrame:
    df_ticker = df_ticker.sort_values("Date").copy()
    ticker = str(df_ticker["Ticker"].iloc[0])

    # Target: next-day direction
    df_ticker["y"] = (df_ticker["Target_Return"] > 0).astype(int)

    FEATURES = select_features(df_ticker)
    if not FEATURES:
        raise ValueError("No feature columns found. Ensure *_lag1 features exist in the CSV.")

    n = len(df_ticker)
    if n < 300:
        raise ValueError(f"Not enough samples ({n}) for a stable split).")

    split = int(n * 0.8)
    X = df_ticker[FEATURES].values
    y = df_ticker["y"].values

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[{ticker}] train pos_rate={y_train.mean():.3f}, test pos_rate={y_test.mean():.3f}")

    # Time-ordered calibration split inside TRAIN
    X_tr, X_val, y_tr, y_val = train_val_split_time(X_train, y_train, val_frac=0.15)

    # ----- Logistic Regression: L1 + Platt calibration -----
    base_logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, C=0.1, penalty="l1", solver="saga", class_weight=None
        )),
    ])
    base_logit.fit(X_tr, y_tr)
    logit_cal = CalibratedClassifierCV(base_logit, cv="prefit", method="sigmoid")
    logit_cal.fit(X_val, y_val)

    p_val_logit = logit_cal.predict_proba(X_val)[:, 1]
    thr_logit = threshold_max_balacc(p_val_logit, y_val)

    p_test_logit = logit_cal.predict_proba(X_test)[:, 1]
    pred_pos_rate_logit = float((p_test_logit >= thr_logit).mean())
    print(f"[{ticker}][LogReg+cal L1] thr={thr_logit:.3f} | pred_pos_rate_test={pred_pos_rate_logit:.3f}")
    m_logit = evaluate(y_test, p_test_logit, threshold=thr_logit)

    # ----- RandomForest + Platt calibration -----
    rf_base = RandomForestClassifier(
        n_estimators=600, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    rf_base.fit(X_tr, y_tr)
    rf_cal = CalibratedClassifierCV(rf_base, cv="prefit", method="sigmoid")
    rf_cal.fit(X_val, y_val)

    p_val_rf = rf_cal.predict_proba(X_val)[:, 1]
    thr_rf = threshold_max_balacc(p_val_rf, y_val)

    p_test_rf = rf_cal.predict_proba(X_test)[:, 1]
    pred_pos_rate_rf = float((p_test_rf >= thr_rf).mean())
    print(f"[{ticker}][RF+cal]      thr={thr_rf:.3f} | pred_pos_rate_test={pred_pos_rate_rf:.3f}")
    m_rf = evaluate(y_test, p_test_rf, threshold=thr_rf)

    # ----- Baselines -----
    y_pred_up = np.ones_like(y_test)
    m_up = evaluate_hard(y_test, y_pred_up)

    if "Return_lag1" in FEATURES:
        y_pred_sign = (df_ticker.iloc[split:]["Return_lag1"].values > 0).astype(int)
        m_sign = evaluate_hard(y_test, y_pred_sign)
    else:
        m_sign = {k: np.nan for k in ["acc","bal_acc","prec","rec","f1","roc_auc","tn","fp","fn","tp"]}
        m_sign.update({"tn":0,"fp":0,"fn":0,"tp":0})

    rows = [
        fmt_metrics_row(ticker, "LogReg(L1)+cal(max-balacc)", thr_logit, m_logit),
        fmt_metrics_row(ticker, "RF+cal(max-balacc)",        thr_rf,   m_rf),
        fmt_metrics_row(ticker, "Baseline:AlwaysUp",         np.nan,   m_up),
        fmt_metrics_row(ticker, "Baseline:Sign(Return_lag1)",np.nan,   m_sign),
    ]
    return pd.DataFrame(rows)

# ---------- script entrypoint (metrics report) ----------
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    if "Target_Return" not in df.columns:
        raise ValueError("Missing 'Target_Return' in processed CSV. Re-run feature_engineering step.")

    out_all = []
    for ticker, g in df.groupby("Ticker"):
        try:
            out_all.append(run_for_ticker(g))
        except Exception as e:
            print(f"[WARN] Skipping {ticker}: {e}")

    if not out_all:
        print("No results produced. Check data and features.")
        return

    report = pd.concat(out_all, ignore_index=True)
    order = {
        "LogReg(L1)+cal(max-balacc)": 0,
        "RF+cal(max-balacc)": 1,
        "Baseline:AlwaysUp": 2,
        "Baseline:Sign(Return_lag1)": 3,
    }
    report["model_rank"] = report["Model"].map(order).fillna(99)
    report = report.sort_values(["Ticker", "model_rank"]).drop(columns=["model_rank"])

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(report.to_string(index=False))

    out_path = os.path.join("data", "reports", "ml_direction_report.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")
    s3_upload_if_configured(out_path, f"reports/{os.path.basename(out_path)}")

# ---------- API used by dashboard ----------

def _train_and_choose_model(df_ticker: pd.DataFrame):
    """
    Train both (LogReg L1 + cal) and (RF + cal), choose by validation balanced accuracy.
    Return (best_model, best_thr, info_dict).
    """
    df_t = df_ticker.sort_values("Date").copy()
    df_t["y"] = (df_t["Target_Return"] > 0).astype(int)
    FEATURES = select_features(df_t)
    if not FEATURES:
        raise ValueError("No feature columns found to predict.")

    n = len(df_t)
    if n < 300:
        raise ValueError(f"Not enough samples ({n}). Need >= 300.")

    split = int(n * 0.8)
    X = df_t[FEATURES].values
    y = df_t["y"].values

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # time-ordered val split inside train
    X_tr, X_val, y_tr, y_val = train_val_split_time(X_train, y_train, val_frac=0.15)

    # model A
    base_logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.1, penalty="l1", solver="saga", class_weight=None)),
    ])
    base_logit.fit(X_tr, y_tr)
    logit_cal = CalibratedClassifierCV(base_logit, cv="prefit", method="sigmoid")
    logit_cal.fit(X_val, y_val)
    p_val_a = logit_cal.predict_proba(X_val)[:, 1]
    thr_a = threshold_max_balacc(p_val_a, y_val)
    balacc_a = balanced_accuracy_score(y_val, (p_val_a >= thr_a).astype(int))

    # model B
    rf_base = RandomForestClassifier(n_estimators=600, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_base.fit(X_tr, y_tr)
    rf_cal = CalibratedClassifierCV(rf_base, cv="prefit", method="sigmoid")
    rf_cal.fit(X_val, y_val)
    p_val_b = rf_cal.predict_proba(X_val)[:, 1]
    thr_b = threshold_max_balacc(p_val_b, y_val)
    balacc_b = balanced_accuracy_score(y_val, (p_val_b >= thr_b).astype(int))

    if balacc_a >= balacc_b:
        return logit_cal, float(thr_a), {"model": "LogReg(L1)+cal", "bal_acc_val": float(balacc_a)}
    else:
        return rf_cal, float(thr_b), {"model": "RF+cal", "bal_acc_val": float(balacc_b)}

def predict_for_tickers(
    tickers: Optional[List[str]] = None,
    df: Optional[pd.DataFrame] = None
) -> List[Dict]:
    """
    Returns a list of dicts:
      {date, ticker, prediction, probability, price, as_of}
    - date: next business day after the last row in data for that ticker
    - prediction: 'UP' or 'DOWN' (using chosen model + threshold)
    - probability: calibrated probability of UP
    - price: last known Close (if present), else NaN
    """
    if df is None:
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    if "Target_Return" not in df.columns:
        raise ValueError("Missing 'Target_Return' in processed CSV.")

    want = set(tickers) if tickers else set(df["Ticker"].unique())
    out: List[Dict] = []
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    for tkr, g in df.groupby("Ticker"):
        if tkr not in want:
            continue
        try:
            model, thr, info = _train_and_choose_model(g)
            FEATURES = select_features(g)
            X_last = g.sort_values("Date").iloc[-1:][FEATURES].values  # latest row features
            p_up = float(model.predict_proba(X_last)[:, 1][0])
            pred = "UP" if p_up >= thr else "DOWN"

            last_date = g["Date"].max()
            next_date = (last_date + BDay(1)).date().isoformat()

            # try to find a price-like column
            lc = [c for c in g.columns if c.lower() in ("close", "adj close", "adj_close", "adjclose")]
            price = float(g.iloc[-1][lc[0]]) if lc else float("nan")

            out.append({
                "date": next_date,
                "ticker": tkr,
                "prediction": pred,
                "probability": p_up,
                "price": price,
                "as_of": now_iso,
                "model_used": info["model"],
                "thr": thr,
                "bal_acc_val": info["bal_acc_val"],
            })
        except Exception as e:
            print(f"[WARN] predict_for_tickers: skipping {tkr}: {e}")

    return out

if __name__ == "__main__":
    main()
