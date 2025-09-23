import os, glob
import pandas as pd
from pathlib import Path

BASE = Path("data/processed")
out_path = BASE / "arima_results_public.csv"

rows = []
for fp in glob.glob(str(BASE / "arima_grid_*.csv")):
    tkr = Path(fp).stem.replace("arima_grid_","").upper()
    df = pd.read_csv(fp)

    # Normalize likely column names
    cols = {c.lower(): c for c in df.columns}
    # date column
    date_col = cols.get("forecastdate") or cols.get("date")
    # predicted close column
    pred_col = cols.get("predictedclose") or cols.get("predclose") or cols.get("yhat") or cols.get("forecast")
    if not date_col or not pred_col:
        print(f"[ARIMA-PUBLIC] skip {fp}: missing needed columns"); 
        continue

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, pred_col]).sort_values(date_col)

    if df.empty:
        continue

    # Take the *last* forecast row
    last = df.iloc[-1]
    rows.append({
        "Ticker": tkr,
        "ForecastDate": pd.to_datetime(last[date_col]).date().isoformat(),
        "PredictedClose": float(last[pred_col]),
    })

if not rows:
    raise SystemExit("[ARIMA-PUBLIC] No usable arima_grid_*.csv files found")

out = pd.DataFrame(rows, columns=["Ticker","ForecastDate","PredictedClose"])
out.to_csv(out_path, index=False)
print(f"? wrote {out_path} rows: {len(out)}")
