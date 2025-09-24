from __future__ import annotations

# ---------- stdlib ----------
import csv
import json
import math
import os
import sys
import pathlib
import datetime
import traceback
from typing import List, Dict, Any, Optional

# ---------- make project root importable ----------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------- project imports ----------
from modeling.ml_direction_classifier import predict_for_tickers as _predict

# ---------- paths ----------
ROOT    = PROJECT_ROOT
PUBLIC  = ROOT / "public"
DATA    = ROOT / "data"

DEFAULT_TICKERS = ["AAPL", "MSFT", "INTC", "BTC-USD"]
AS_OF = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# =========================
# Helpers
# =========================
def _norm_cols(df):
    """lower + strip once; return mapping old->normalized and apply inplace rename"""
    mapping = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=mapping, inplace=True)
    return mapping

def _pick_date_col(df):
    for k in ("forecastdate", "date", "ds"):
        if k in df.columns: return k
    return None

def _pick_pred_col(df):
    # most likely names first
    for k in ("predictedclose", "predclose", "yhat", "forecast", "prediction", "pred"):
        if k in df.columns: return k
    # looser contains search
    for c in df.columns:
        if ("pred" in c or "fore" in c) and ("close" in c or "yhat" in c or "price" in c):
            return c
    return None

def _load_arima_public_single_csv(path: str) -> Dict[str, float]:
    """
    Try to read ARIMA_PUBLIC_PATH (a single CSV) and return {TICKER: last_predicted_close}.
    Accepts Date/ForecastDate and PredClose/PredictedClose/yhat/etc.
    """
    out: Dict[str, float] = {}
    try:
        import pandas as pd
        if not os.path.exists(path):
            return out

        df = pd.read_csv(path)
        _ = _norm_cols(df)

        # ensure we have 'ticker'
        if "ticker" not in df.columns:
            raise ValueError("ARIMA file missing 'Ticker' column.")

        date_col = _pick_date_col(df)
        pred_col = _pick_pred_col(df)
        if not date_col or not pred_col:
            raise ValueError(f"ARIMA file missing date/pred columns. date_col={date_col}, pred_col={pred_col}")

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, pred_col])

        for tkr, g in df.groupby("ticker"):
            g = g.sort_values(date_col)
            try:
                out[tkr] = float(g[pred_col].iloc[-1])
            except Exception:
                pass
        return out

    except Exception as e:
        print(f"[dashboard] ARIMA public load failed ({path}): {e}")
        return {}

def _load_arima_from_grids(dir_path: str) -> Dict[str, float]:
    """
    Fallback: scan data/processed/arima_grid_*.csv and pick the latest forecast per ticker.
    """
    out: Dict[str, float] = {}
    try:
        import pandas as pd, glob
        pattern = str(pathlib.Path(dir_path) / "arima_grid_*.csv")
        files = glob.glob(pattern)
        if not files:
            return out

        for fp in files:
            tkr = pathlib.Path(fp).stem.replace("arima_grid_", "").upper()
            try:
                df = pd.read_csv(fp)
                _ = _norm_cols(df)
                date_col = _pick_date_col(df)
                pred_col = _pick_pred_col(df)
                if not date_col or not pred_col:
                    # skip quietly; this file shape may be metrics only
                    continue
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col, pred_col]).sort_values(date_col)
                if not df.empty:
                    out[tkr] = float(df[pred_col].iloc[-1])
            except Exception as e:
                print(f"[dashboard] skip {fp}: {e}")
        return out
    except Exception as e:
        print("[dashboard] ARIMA grid scan failed:", e)
        return {}

def load_arima_map() -> Dict[str, float]:
    """
    Robust loader that tries:
      1) ARIMA_PUBLIC_PATH (single CSV)
      2) data/processed/arima_results_public.csv
      3) fallback to scanning data/processed/arima_grid_*.csv
    """
    # 1) explicit env
    env_path = os.getenv("ARIMA_PUBLIC_PATH")
    if env_path:
        m = _load_arima_public_single_csv(env_path)
        if m: return m

    # 2) default public results
    default_public = ROOT / "data" / "processed" / "arima_results_public.csv"
    m = _load_arima_public_single_csv(str(default_public))
    if m: return m

    # 3) fallback to grids
    return _load_arima_from_grids(str(ROOT / "data" / "processed"))

def fetch_live_price(ticker: str, fallback: Optional[float] = None) -> Optional[float]:
    """
    Try yfinance; fallback to provided ref price (CI often blocks network).
    """
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
        if px is None or (isinstance(px, float) and math.isnan(px)):
            return fallback
        return float(px)
    except Exception as e:
        print(f"[dashboard] yfinance failed for {ticker}: {e}")
        return fallback

def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a prediction row into canonical schema."""
    out: Dict[str, Any] = {}
    out["ticker"] = str(row.get("ticker") or row.get("symbol") or "").upper()
    out["date"]   = str(row.get("date") or row.get("target_date") or row.get("prediction_date") or "")

    pred = (row.get("prediction") or row.get("signal") or row.get("label") or "")
    if not pred and isinstance(row.get("pred"), str):
        pred = row["pred"]
    out["prediction"] = str(pred).upper() if pred else ""

    p = row.get("probability", row.get("confidence", row.get("prob")))
    try:
        out["probability"] = float(p)
    except Exception:
        try:
            out["probability"] = float(str(p).rstrip("%")) / 100.0
        except Exception:
            out["probability"] = None

    price = row.get("price") or row.get("close") or row.get("ref_price") or row.get("AdjClose")
    try:
        out["price"] = float(price)
    except Exception:
        out["price"] = None

    out["as_of"] = row.get("as_of") or AS_OF
    return out

def run_inference(tickers: List[str]) -> List[Dict[str, Any]]:
    """Call your classifier; bubble errors so CI fails loudly if broken."""
    try:
        rows = _predict(tickers)
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict(orient="records")
        return [_normalize_row(r) for r in rows]
    except Exception as e:
        print("[dashboard] predict_for_tickers failed:", e)
        traceback.print_exc()
        raise


# =========================
# Renderers (CSV / JSON / HTML)
# =========================
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
        fieldnames = ["as_of","date","ticker","prediction","probability","price","arima_pred","arima_delta_pct","live_price"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in norm:
            w.writerow({
                "as_of": r.get("as_of", as_of),
                "date": r.get("date",""),
                "ticker": r.get("ticker",""),
                "prediction": r.get("prediction",""),
                "probability": r.get("probability"),
                "price": r.get("price"),
                "arima_pred": r.get("arima_pred"),
                "arima_delta_pct": r.get("arima_delta_pct"),
                "live_price": r.get("live_price"),
            })

    # JSON
    data = {"generated_at": AS_OF, "as_of": as_of, "count": len(norm), "predictions": norm}
    (PUBLIC / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # HTML helpers
    def fmt_prob(p):
        try: return f"{float(p)*100:.1f}%"
        except: return "" if p is None else str(p)

    def fmt_num(x, nd=2):
        try: return f"{float(x):.{nd}f}"
        except: return ""

    rows_html = []
    for r in norm:
        pred = (r.get("prediction") or "").upper()
        pill = "pill"
        if pred in ("UP","BUY"): pill = "pill up"
        elif pred in ("DOWN","SELL"): pill = "pill down"
        rows_html.append(f"""
<tr>
  <td><span class="muted">{r.get('date','')}</span></td>
  <td class="ticker">{r.get('ticker','')}</td>
  <td><span class="{pill}">{pred}</span></td>
  <td class="right">{fmt_prob(r.get('probability'))}</td>
  <td class="right">{fmt_num(r.get('price'), 2)}</td>
  <td class="right">{fmt_num(r.get('arima_pred'), 3)}</td>
  <td class="right">{("" if r.get('arima_delta_pct') is None else f"{float(r['arima_delta_pct']):.2f}%")}</td>
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


# =========================
# Entrypoint
# =========================
def main():
    # tickers
    tickers_env = os.getenv("TICKERS", ",".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]

    # 1) Directional predictions
    rows = run_inference(tickers)

    # 2) ARIMA map (robust)
    arima_map = load_arima_map()

    # 3) Decorate rows with ARIMA + live price
    for r in rows:
        tkr = r.get("ticker")
        ref_price = r.get("price")

        # ARIMA predicted close (if available)
        apred = arima_map.get(tkr)
        r["arima_pred"] = apred
        r["arima_delta_pct"] = None
        if apred is not None and ref_price not in (None, 0):
            try:
                r["arima_delta_pct"] = 100.0 * (float(apred) - float(ref_price)) / float(ref_price)
            except Exception:
                pass

        # live price (best effort; fall back to ref)
        lp = fetch_live_price(tkr, fallback=ref_price)
        r["live_price"] = lp if lp is not None else ref_price

    # 4) Write artifacts
    write_outputs(rows)


if __name__ == "__main__":
    main()
