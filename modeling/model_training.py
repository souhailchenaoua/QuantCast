import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Accept either filename
CANDIDATES = [
    os.path.join("data", "processed", "features.csv"),
    os.path.join("data", "processed", "prices_features.csv"),
]
DATA_FILE = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATA_FILE is None:
    raise FileNotFoundError(f"Couldn't find any of: {CANDIDATES}. Run feature_engineering.py first.")

def time_split_per_ticker(df, test_frac=0.2):
    """Return a copy with a __split column set to 'train'/'test' per ticker, split by time."""
    parts = []
    for t, g in df.groupby("Ticker", group_keys=False):
        g = g.sort_values("Date").copy()
        n = len(g)
        split_idx = int(n * (1 - test_frac))
        g_train = g.iloc[:split_idx].copy()
        g_test  = g.iloc[split_idx:].copy()
        g_train.loc[:, "__split"] = "train"
        g_test.loc[:, "__split"]  = "test"
        parts.append(pd.concat([g_train, g_test], axis=0))
    out = pd.concat(parts, axis=0).sort_values(["Ticker","Date"]).reset_index(drop=True)
    return out

def main():
    # Load engineered features
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)

    # Target: tomorrow up (1) / down (0)
    df["Target"] = (df.groupby("Ticker")["Return"].shift(-1) > 0).astype(int)

    # Drop NaNs from rolling/shift
    req = ["Return","MA7","MA30","Vol20","RSI14","Target"]
    df = df.dropna(subset=req).copy()

    # Time-based split per ticker
    df = time_split_per_ticker(df, test_frac=0.2)

    feature_cols = ["Return","MA7","MA30","Vol20","RSI14"]
    train_mask = df["__split"].eq("train")
    test_mask  = df["__split"].eq("test")

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "Target"]
    X_test  = df.loc[test_mask,  feature_cols]
    y_test  = df.loc[test_mask,  "Target"]

    # Pipeline: scale + balanced logistic regression
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, solver="lbfgs")),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Overall metrics
    print("\n=== Overall (All tickers) ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # ----- Per-ticker metrics (fixed indexing) -----
    # Align predictions with the test-index so boolean masks match
    test_index = y_test.index
    y_pred_s = pd.Series(y_pred, index=test_index)

    print("\n=== Per-ticker metrics ===")
    for t in df["Ticker"].unique():
        mask_t = test_mask & df["Ticker"].eq(t)
        idx_t = df.index[mask_t]
        if len(idx_t) == 0:
            continue
        print(f"\n[{t}]")
        print(classification_report(y_test.loc[idx_t], y_pred_s.loc[idx_t], digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_test.loc[idx_t], y_pred_s.loc[idx_t]))

if __name__ == "__main__":
    main()
