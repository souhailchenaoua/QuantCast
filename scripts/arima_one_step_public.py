import pandas as pd
import numpy as np
from pathlib import Path
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "prices_features.csv"
PUBLIC_DIR = ROOT / "public"
OUT_CSV = PUBLIC_DIR / "predictions.csv"
OUT_JSON = PUBLIC_DIR / "predictions.json"   # optional
# ---------------------------------------------------------------------

df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])

# pick price column dynamically
price_col = next(
    (c for c in df.columns if c.lower().replace(" ", "") in ["adjclose", "adj_close", "close"]),
    None
)
if price_col is None:
    raise SystemExit("❌ No price column found (AdjClose/Close).")

rows = []

for tkr, g in df.groupby("Ticker"):
    g = g.sort_values("Date")
    px = g[price_col].astype(float).values

    if len(px) < 60:
        print(f"⚠️ Skipping {tkr}: not enough history")
        continue

    ret = np.diff(px) / px[:-1]

    try:
        model = ARIMA(ret, order=(1, 0, 0))
        res = model.fit(method_kwargs={"warn_convergence": False})
        f = res.get_forecast(steps=1).predicted_mean
        next_ret = float(f.iloc[0]) if hasattr(f, "iloc") else float(np.array(f)[0])
    except Exception as e:
        print(f"⚠️ ARIMA failed for {tkr}: {e}")
        continue

    last_close = float(px[-1])
    pred_close = last_close * (1.0 + next_ret)

    last_date = g["Date"].max()
    forecast_date = (last_date + BDay(1)).date().isoformat()

    rows.append({
        "Ticker": str(tkr).upper(),
        "ForecastDate": forecast_date,
        "PredictedClose": round(pred_close, 4)
    })

if not rows:
    raise SystemExit("❌ No predictions generated — check your data.")

out_df = pd.DataFrame(rows)

PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# Write CSV for website
out_df.to_csv(OUT_CSV, index=False)
print(f"✅ Wrote {OUT_CSV} rows: {len(out_df)}")

# Optional JSON export (if your web uses JSON)
out_df.to_json(OUT_JSON, orient="records")
print(f"📄 JSON also written → {OUT_JSON}")
