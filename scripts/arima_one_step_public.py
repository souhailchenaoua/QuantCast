
import os
import pandas as pd
import numpy as np
from pathlib import Path
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima.model import ARIMA

FEATURES_PATH = os.environ.get("PRICES_FEATURES_PATH", "data/processed/prices_features_public.csv")
OUT = Path("data/processed/arima_results_public.csv")

df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
# pick price column
price_col = None
for c in df.columns:
    cl = c.strip().lower()
    if cl in ("adjclose","adj close","close","adj_close"):
        price_col = c
        break
if price_col is None:
    raise SystemExit("No price column (AdjClose/Close) found in features CSV.")

rows = []
for tkr, g in df.groupby("Ticker"):
    g = g.sort_values("Date")
    px = g[price_col].astype(float).values
    if len(px) < 60:
        continue
    ret = np.diff(px) / px[:-1]  # simple returns

    # fit ARIMA(1,0,0) on returns (ignore warnings for small samples)
    try:
        model = ARIMA(ret, order=(1,0,0))
        res = model.fit(method_kwargs={"warn_convergence": False})
        # 1-step ahead forecast of returns
        f = res.get_forecast(steps=1).predicted_mean
        next_ret = float(f.iloc[0]) if hasattr(f, "iloc") else float(np.array(f)[0])
    except Exception as e:
        print(f"[ARIMA one-step] skip {tkr}: {e}")
        continue

    last_close = float(px[-1])
    pred_close = last_close * (1.0 + next_ret)

    last_date = g["Date"].max()
    forecast_date = (last_date + BDay(1)).date().isoformat()

    rows.append({"Ticker": str(tkr).upper(), "ForecastDate": forecast_date, "PredictedClose": pred_close})

if not rows:
    raise SystemExit("No ARIMA rows produced — check features CSV/path.")

out = pd.DataFrame(rows, columns=["Ticker","ForecastDate","PredictedClose"])
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"✅ wrote {OUT} rows: {len(out)}")
