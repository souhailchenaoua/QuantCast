from __future__ import annotations
import math

# --- make project root importable (so "modeling" resolves) ---
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# stdlib
import csv, json, os, importlib, datetime, traceback
from typing import List, Dict, Any

# your classifier (expects predict_for_tickers to exist as added earlier)
from modeling.ml_direction_classifier import predict_for_tickers as _predict

# --- paths (rooted at project root) ---
ROOT   = PROJECT_ROOT
PUBLIC = ROOT / "public"
REPORTS = ROOT / "reports"
DATA    = ROOT / "data"

DEFAULT_TICKERS = ["AAPL", "MSFT", "BTC-USD"]
AS_OF = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ---------- helpers ----------
def load_arima_public(path: str) -> dict[str, float]:
    """
    Load 'Ticker, ForecastDate, PredictedClose' and return {ticker: predicted_close}.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path, parse_dates=["ForecastDate"])
        out = {}
        for tkr, g in df.groupby("Ticker"):
            g = g.sort_values("ForecastDate")
            out[str(tkr)] = float(g["PredictedClose"].iloc[-1])
        return out
    except Exception as e:
        print("[dashboard] ARIMA public load failed:", e)
        return {}

def fetch_live_price(ticker: str, fallback: float | None = None) -> float | None:
    """
    Try yfinance for the latest price; fall back to provided value.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker)
        px = info.fast_info.get("last_price") or info.fast_info.get("last_close")
        if px is None and hasattr(info, "history"):
            h = info.history(period="1d")
            if not h.empty and "Close" in h.columns:
                px = float(h["Close"].iloc[-1])
        return float(px) if px is not None and not (isinstance(px, float) and math.isnan(px)) else fallback
    except Exception as e:
        print(f"[dashboard] yfinance failed for {ticker}: {e}")
        return fallback
    
def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a prediction row into canonical schema."""
    out: Dict[str, Any] = {}
    out["ticker"] = str(row.get("ticker") or row.get("symbol") or "").upper()
    out["date"]   = str(row.get("date") or row.get("target_date") or row.get("prediction_date") or "")
    pred = (row.get("prediction") or row.get("signal") or row.get("label") or "").upper()
    if not pred and isinstance(row.get("pred"), str):
        pred = row["pred"].upper()
    out["prediction"] = pred or ""
    p = row.get("probability", row.get("confidence", row.get("prob")))
    try:
        out["probability"] = float(p)
    except Exception:
        try:
            out["probability"] = float(str(p).rstrip("%")) / 100.0
        except Exception:
            out["probability"] = None
    price = row.get("price") or row.get("close") or row.get("ref_price")
    try:
        out["price"] = float(price)
    except Exception:
        out["price"] = None
    out["as_of"] = row.get("as_of") or AS_OF
    return out

def run_inference(tickers: List[str]) -> List[Dict[str, Any]]:
    """Call your classifier; raise if it fails (so CI shows it)."""
    try:
        rows = _predict(tickers)
        # allow DataFrame or list[dict]
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict(orient="records")
        return [_normalize_row(r) for r in rows]
    except Exception as e:
        print("[dashboard] predict_for_tickers failed:", e)
        traceback.print_exc()
        raise

# ---------- render outputs ----------
def write_outputs(rows: List[Dict[str, Any]]) -> None:
    PUBLIC.mkdir(exist_ok=True)

    def safe_date(s: str) -> datetime.datetime:
        for fmt in ("%Y-%m-%d",):
            try:
                return datetime.datetime.strptime(s, fmt)
            except Exception:
                pass
        try:
            return datetime.datetime.fromisoformat(s)
        except Exception:
            return datetime.datetime.min

    norm = [_normalize_row(r) for r in rows]
    norm.sort(key=lambda r: (safe_date(r.get("date", "")), r.get("ticker", "")), reverse=True)

    as_of_values = [r.get("as_of") for r in norm if r.get("as_of")]
    as_of = max(as_of_values) if as_of_values else AS_OF

    # CSV
    csv_path = PUBLIC / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["as_of", "date", "ticker", "prediction", "probability", "price"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in norm:
            w.writerow({
                "as_of": r.get("as_of", as_of),
                "date": r.get("date", ""),
                "ticker": r.get("ticker", ""),
                "prediction": r.get("prediction", ""),
                "probability": r.get("probability"),
                "price": r.get("price"),
            })

    # JSON
    data = {
        "generated_at": AS_OF,
        "as_of": as_of,
        "count": len(norm),
        "predictions": norm,
    }
    (PUBLIC / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # HTML
    def fmt_prob(p):
        try:
            return f"{float(p) * 100:.1f}%"
        except Exception:
            return str(p) if p is not None else ""

    rows_html = []
    for r in norm:
        pred = (r.get("prediction") or "").upper()
        pill = "pill"
        if pred in ("UP", "BUY"):
            pill = "pill up"
        elif pred in ("DOWN", "SELL"):
            pill = "pill down"
        rows_html.append(f"""
<tr>
  <td><span class="muted">{r.get('date','')}</span></td>
  <td class="ticker">{r.get('ticker','')}</td>
  <td><span class="{pill}">{pred}</span></td>
  <td class="right">{fmt_prob(r.get('probability'))}</td>
  <td class="right">{("" if r.get('price') is None else round(float(r['price']), 2))}</td>
  <td class="right">{("" if r.get('arima_pred') is None else round(float(r['arima_pred']), 3))}</td>
  <td class="right">{("" if r.get('arima_delta_pct') is None else f"{float(r['arima_delta_pct']):.2f}%")}</td>
  <td class="right">{("" if r.get('live_price') is None else round(float(r['live_price']), 3))}</td>
</tr>
""")

    rows_html = "\n".join(rows_html)

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>QuantCast — Predictions</title>
<style>
:root {{ --bg:#0b0e14; --card:#121826; --muted:#7a869a; --text:#e6edf3; --up:#2ecc71; --down:#ff6b6b; }}
html,body {{ background:var(--bg); color:var(--text); font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial; margin:0; }}
.wrap {{ max-width:1100px; margin:40px auto; padding:0 20px; }}
.card {{ background:var(--card); border-radius:14px; padding:18px; box-shadow:0 10px 25px rgba(0,0,0,.25); }}
h1 {{ margin:0 0 6px; font-size:24px; }}
.meta {{ color:var(--muted); font-size:13px; margin-bottom:12px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid #1f2937; text-align:left; white-space:nowrap; }}
th {{ color:#b7c0ce; font-weight:600; background:#0e1525; position:sticky; top:0; }}
tr:hover td {{ background:#0e1525; }}
.pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; background:#243047; }}
.up {{ background:rgba(46,204,113,.15); color:var(--up); }}
.down {{ background:rgba(255,107,107,.15); color:var(--down); }}
.muted {{ color:var(--muted); }}
.right {{ text-align:right; }}
.ticker {{ font-weight:700; letter-spacing:.3px; }}
.footer {{ color:var(--muted); font-size:12px; margin-top:12px; }}
</style></head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>QuantCast — Predictions</h1>
      <div class="meta">Generated: <strong>{AS_OF}</strong> | Model as-of: <strong>{as_of}</strong> | Rows: <strong>{len(norm)}</strong></div>
      <div style="overflow:auto; max-height:75vh;">
        <table>
          <thead>
          <tr><th>Date</th><th>Ticker</th><th>Prediction</th><th class="right">Confidence</th><th class="right">Ref Price</th> <th class="right">Confidence</th>
  <th class="right">Ref Price</th>
  <th class="right">ARIMA Pred</th>
  <th class="right">Δ% (ARIMA vs Ref)</th>
  <th class="right">Live Price</th></tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
      <div class="footer">Static site generated by GitHub Actions · No cookies, no JS libs.</div>
    </div>
  </div>
</body></html>"""
    (PUBLIC / "index.html").write_text(html, encoding="utf-8")
    print(f"[dashboard] wrote {csv_path}")
    print(f"[dashboard] wrote {PUBLIC/'data.json'}")
    print(f"[dashboard] wrote {PUBLIC/'index.html'}")

# ---------- entrypoint ----------
def main():
    tickers = os.getenv("TICKERS", ",".join(DEFAULT_TICKERS)).split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    # 1) direction predictions
    rows = run_inference(tickers)

    # 2) ARIMA one-step forecast (public)
    arima_path = os.getenv("ARIMA_PUBLIC_PATH", os.path.join(ROOT.parent, "data", "processed", "arima_results_public.csv"))
    arima_map = load_arima_public(arima_path)

    # 3) decorate rows with arima_pred & live price
    for r in rows:
        tkr = r.get("ticker")
        ref_price = r.get("price")
        # ARIMA predicted close
        apred = arima_map.get(tkr)
        r["arima_pred"] = apred
        r["arima_delta_pct"] = None
        if apred is not None and ref_price not in (None, 0):
            try:
                r["arima_delta_pct"] = 100.0 * (apred - float(ref_price)) / float(ref_price)
            except Exception:
                pass
        # live price
        r["live_price"] = fetch_live_price(tkr, fallback=ref_price)

    write_outputs(rows)


if __name__ == "__main__":
    main()
