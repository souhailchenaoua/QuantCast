from __future__ import annotations

# stdlib
import csv, json, math, os, sys, pathlib, datetime, traceback, glob
from typing import List, Dict, Any, Optional

# deps
import pandas as pd

# project root on path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# project imports
from modeling.ml_direction_classifier import predict_for_tickers as _predict

# paths / constants
ROOT   = PROJECT_ROOT
PUBLIC = ROOT / "public"
DEFAULT_TICKERS = ["AAPL", "MSFT", "INTC", "BTC-USD"]
AS_OF = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["ticker"] = str(row.get("ticker") or row.get("symbol") or "").upper()
    out["date"] = str(row.get("date") or row.get("target_date") or row.get("prediction_date") or "")
    pred = (row.get("prediction") or row.get("signal") or row.get("label") or "")
    if not pred and isinstance(row.get("pred"), str): pred = row["pred"]
    out["prediction"] = str(pred).upper() if pred else ""
    p = row.get("probability", row.get("confidence", row.get("prob")))
    try: out["probability"] = float(p)
    except Exception:
        try: out["probability"] = float(str(p).rstrip("%")) / 100.0
        except Exception: out["probability"] = None
    price = row.get("price") or row.get("close") or row.get("ref_price") or row.get("AdjClose")
    try: out["price"] = float(price)
    except Exception: out["price"] = None
    out["as_of"] = row.get("as_of") or AS_OF
    return out


def run_inference(tickers: List[str]) -> List[Dict[str, Any]]:
    try:
        rows = _predict(tickers)
        if hasattr(rows, "to_dict"): rows = rows.to_dict(orient="records")
        return [_normalize_row(r) for r in rows]
    except Exception as e:
        print("[dashboard] predict_for_tickers failed:", e)
        traceback.print_exc()
        raise


def load_arima_map() -> Dict[str, float]:
    """Return {TICKER: predicted_close} from ARIMA_PUBLIC_PATH or arima_grid_*.csv."""
    path = os.getenv("ARIMA_PUBLIC_PATH") or str(ROOT / "data" / "processed" / "arima_results_public.csv")

    # single csv
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            if "forecastdate" not in df.columns and "date" in df.columns:
                df.rename(columns={"date": "forecastdate"}, inplace=True)
            for src in ("predictedclose","predclose","yhat","forecast","prediction","pred","arima_pred","price_pred","forecast_price"):
                if src in df.columns:
                    df.rename(columns={src: "predictedclose"}, inplace=True)
                    break
            if "ticker" in df.columns and "predictedclose" in df.columns:
                return {
                    str(t).upper(): float(v)
                    for t, v in zip(df["ticker"], df["predictedclose"])
                    if pd.notna(v)
                }
        except Exception as e:
            print(f"[dashboard] ARIMA load error ({path}): {e}")

    # grid fallbacks
    out: Dict[str, float] = {}
    try:
        pattern = str(ROOT / "data" / "processed" / "arima_grid_*.csv")
        for fp in glob.glob(pattern):
            try:
                df = pd.read_csv(fp)
                df.columns = [c.strip().lower() for c in df.columns]
                date_col = next((k for k in ("forecastdate","date","ds") if k in df.columns), None)
                pred_col = next((k for k in ("predictedclose","predclose","yhat","forecast","prediction","pred") if k in df.columns), None)
                if not pred_col:
                    for c in df.columns:
                        if ("pred" in c or "fore" in c) and ("close" in c or "yhat" in c or "price" in c):
                            pred_col = c; break
                if not date_col or not pred_col: continue
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col, pred_col]).sort_values(date_col)
                if df.empty: continue
                tkr = pathlib.Path(fp).stem.replace("arima_grid_", "").upper()
                out[tkr] = float(df[pred_col].iloc[-1])
            except Exception as e:
                print(f"[dashboard] skip {fp}: {e}")
    except Exception as e:
        print("[dashboard] ARIMA grid scan failed:", e)
    return out


def fetch_live_price(ticker: str, fallback: Optional[float] = None) -> Optional[float]:
    if os.getenv("FETCH_LIVE","1") == "0":
        return fallback
    try:
        import yfinance as yf
        info = yf.Ticker(ticker)
        px = None
        if hasattr(info, "fast_info") and isinstance(info.fast_info, dict):
            px = info.fast_info.get("last_price") or info.fast_info.get("last_close")
        if px is None and hasattr(info, "history"):
            h = info.history(period="1d")
            if hasattr(h, "empty") and not h.empty and "Close" in h.columns:
                px = float(h["Close"].iloc[-1])
        if px is None or (isinstance(px, float) and math.isnan(px)): return fallback
        return float(px)
    except Exception as e:
        print(f"[dashboard] yfinance failed for {ticker}: {e}")
        return fallback


def write_outputs(rows: List[Dict[str, Any]]) -> None:
    PUBLIC.mkdir(exist_ok=True)

    def safe_date(s: str) -> datetime.datetime:
        for fmt in ("%Y-%m-%d",):
            try: return datetime.datetime.strptime(s, fmt)
            except Exception: pass
        try: return datetime.datetime.fromisoformat(s)
        except Exception: return datetime.datetime.min

    # normalize + keep ARIMA/live fields, coercing to floats
    def _coerce(x):
        if x is None: return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan": return None
        try: return float(s)
        except Exception: return None

    norm: List[Dict[str, Any]] = []
    for r in rows:
        base = _normalize_row(r)
        base["arima_pred"]      = _coerce(r.get("arima_pred"))
        base["arima_delta_pct"] = _coerce(r.get("arima_delta_pct"))
        base["live_price"]      = _coerce(r.get("live_price"))
        norm.append(base)

    norm.sort(key=lambda r: (safe_date(r.get("date","")), r.get("ticker","")), reverse=True)
    as_of_values = [r.get("as_of") for r in norm if r.get("as_of")]
    as_of = max(as_of_values) if as_of_values else AS_OF

    # round for CSV readability (keep raw in JSON if you prefer—here we keep same)
    for r in norm:
        if r.get("arima_pred") is not None: r["arima_pred"] = round(float(r["arima_pred"]), 3)
        if r.get("arima_delta_pct") is not None: r["arima_delta_pct"] = round(float(r["arima_delta_pct"]), 3)
        if r.get("live_price") is not None: r["live_price"] = round(float(r["live_price"]), 3)

    # CSV
    csv_path = PUBLIC / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["as_of","date","ticker","prediction","probability","price","arima_pred","arima_delta_pct","live_price"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in norm: w.writerow({
            "as_of": r.get("as_of", as_of),
            "date": r.get("date", ""),
            "ticker": r.get("ticker", ""),
            "prediction": r.get("prediction", ""),
            "probability": r.get("probability"),
            "price": r.get("price"),
            "arima_pred": r.get("arima_pred"),
            "arima_delta_pct": r.get("arima_delta_pct"),
            "live_price": r.get("live_price"),
        })

    # JSON
    data = {"generated_at": AS_OF, "as_of": as_of, "count": len(norm), "predictions": norm}
    (PUBLIC / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # HTML render
    def fmt_prob(p):
        try: return f"{float(p)*100:.1f}%"
        except Exception: return "" if p is None else str(p)

    def fmt_num(x, nd=2):
        try:
            v = float(x)
            if math.isnan(v): return ""
            return f"{v:.{nd}f}"
        except Exception:
            return ""

    rows_html: List[str] = []
    for r in norm:
        pred = (r.get("prediction") or "").upper()
        pill = "pill" if pred not in ("UP","BUY","DOWN","SELL") else ("pill up" if pred in ("UP","BUY") else "pill down")
        delta_txt = "" if r.get("arima_delta_pct") is None else f"{fmt_num(r['arima_delta_pct'], 2)}%"
        rows_html.append(f"""
<tr>
  <td><span class="muted">{r.get('date','')}</span></td>
  <td class="ticker">{r.get('ticker','')}</td>
  <td><span class="{pill}">{pred}</span></td>
  <td class="right">{fmt_prob(r.get('probability'))}</td>
  <td class="right">{fmt_num(r.get('price'), 2)}</td>
  <td class="right">{fmt_num(r.get('arima_pred'), 3)}</td>
  <td class="right">{delta_txt}</td>
  <td class="right">{fmt_num(r.get('live_price'), 3)}</td>
</tr>""")
    rows_html = "\n".join(rows_html)

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>QuantCast — Predictions</title>
<style>
:root {{ --bg:#0b0e14; --card:#121826; --muted:#7a869a; --text:#e6edf3; --up:#2ecc71; --down:#ff6b6b; }}
html,body {{ background:var(--bg); color:var(--text); font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial; margin:0; }}
.wrap {{ max-width:1200px; margin:40px auto; padding:0 20px; }}
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
            <tr>
              <th>Date</th>
              <th>Ticker</th>
              <th>Prediction</th>
              <th class="right">Confidence</th>
              <th class="right">Ref Price</th>
              <th class="right">ARIMA Pred</th>
              <th class="right">Δ% (ARIMA vs Ref)</th>
              <th class="right">Live Price</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
      <div class="footer">Static site generated by GitHub Actions · No cookies, no JS libs.</div>
    </div>
  </div>
</body></html>"""
    (PUBLIC / "index.html").write_text(html, encoding="utf-8")
    print(f"[dashboard] wrote {csv_path}")
    print(f"[dashboard] wrote {PUBLIC / 'data.json'}")
    print(f"[dashboard] wrote {PUBLIC / 'index.html'}")


def main():
    tickers_env = os.getenv("TICKERS", ",".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
    rows = run_inference(tickers)
    arima_map = load_arima_map()

    for r in rows:
        tkr = r.get("ticker")
        ref_price = r.get("price")
        apred = arima_map.get(tkr)
        r["arima_pred"] = apred
        r["arima_delta_pct"] = None
        if apred is not None and ref_price not in (None, 0):
            try: r["arima_delta_pct"] = 100.0 * (float(apred) - float(ref_price)) / float(ref_price)
            except Exception: pass
        lp = fetch_live_price(tkr, fallback=ref_price)
        r["live_price"] = lp if lp is not None else ref_price

    write_outputs(rows)


if __name__ == "__main__":
    main()
