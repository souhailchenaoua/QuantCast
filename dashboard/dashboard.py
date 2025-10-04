from __future__ import annotations

# stdlib
import csv, json, math, os, sys, pathlib, datetime, traceback, glob
from typing import List, Dict, Any, Optional, Tuple

# deps
import pandas as pd

# project root on path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# project imports
from modeling.ml_direction_classifier import predict_for_tickers as _predict


# ── Settings / constants ───────────────────────────────────────────────────────
ROOT   = PROJECT_ROOT
PUBLIC = ROOT / "public"

# Override with env: TICKERS="AAPL,MSFT,TSLA,GOOGL,BTC-USD"
DEFAULT_TICKERS = [
    "AAPL", "INTC", "MSFT", "GOOGL", "TSLA",
    "AMZN", "NVDA", "META", "BTC-USD", "ETH-USD"
]

# Single source of "generated at" truth (UTC ISO)
GENERATED_AT = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# Explicit labels embedded in JSON rows
PREDICTION_TARGET        = "Tomorrow Close (next trading session)"
REFERENCE_PRICE_BASIS    = "Last completed session close (Adj Close for equities; Close for crypto)"
ARIMA_FORECAST_BASIS     = "Tomorrow Close (ARIMA t+1)"
ARIMA_DELTA_DEFINITION   = "Percent change ((ARIMA - Reference Close) / Reference Close * 100)"


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_crypto_ticker(ticker: str) -> bool:
    t = (ticker or "").upper()
    # Extend if you add more crypto tickers
    return t in {"BTC-USD", "ETH-USD"}

def next_trading_weekday(d: datetime.date) -> datetime.date:
    """
    Return the next weekday (Mon–Fri). Simple weekend skip (no holiday calendar).
    If d is Fri/Sat/Sun → Monday; else → d+1.
    """
    wd = d.weekday()  # Mon=0..Sun=6
    if wd >= 4:
        return d + datetime.timedelta(days=7 - wd)
    return d + datetime.timedelta(days=1)

def compute_dates_for_row(ticker: str) -> Tuple[str, str]:
    """
    Returns (as_of_local_date_str, target_date_str) both YYYY-MM-DD.
    - as_of_local: Asia/Riyadh calendar date (human-friendly). Falls back to UTC if pytz missing.
    - target_date: crypto → UTC + 1 day; equities → next trading weekday.
    """
    try:
        import pytz
        tz_riyadh = pytz.timezone("Asia/Riyadh")
        as_of_local = datetime.datetime.now(tz_riyadh).date()
    except Exception:
        as_of_local = datetime.datetime.utcnow().date()

    today_utc = datetime.datetime.utcnow().date()
    if is_crypto_ticker(ticker):
        target_date = today_utc + datetime.timedelta(days=1)
    else:
        target_date = next_trading_weekday(today_utc)

    return as_of_local.strftime("%Y-%m-%d"), target_date.strftime("%Y-%m-%d")

def _to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None

def _normalize_probability(p: Any) -> Optional[float]:
    """
    Normalize probability to 0..1 (float).
    Accepts 0..1 floats, '57.9%' strings, or 0..100 numbers (assumed percent).
    """
    if p is None:
        return None
    s = str(p).strip()
    if s.endswith("%"):
        s = s[:-1]
    try:
        val = float(s)
    except Exception:
        return None
    if val > 1.0:
        return max(0.0, min(val / 100.0, 1.0))
    return max(0.0, min(val, 1.0))

def _fmt_prob_pct(p: Any) -> str:
    try:
        v = float(p)
        if v <= 1.0:  # assume 0..1
            return f"{v*100:.1f}%"
        return f"{v:.1f}%"
    except Exception:
        return ""

def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return ""
        return f"{v:.{nd}f}"
    except Exception:
        return ""


# ── Core pipeline steps ───────────────────────────────────────────────────────

def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a model row into canonical fields and inject semantic labels.
    """
    out: Dict[str, Any] = {}

    out["ticker"] = str(row.get("ticker") or row.get("symbol") or "").upper()
    out["date"] = str(row.get("date") or row.get("target_date") or row.get("prediction_date") or "")

    pred = (row.get("prediction") or row.get("signal") or row.get("label") or "")
    if not pred and isinstance(row.get("pred"), str):
        pred = row["pred"]
    out["prediction"] = str(pred).upper() if pred else ""

    out["probability"] = _normalize_probability(
        row.get("probability", row.get("confidence", row.get("prob")))
    )

    price = row.get("price") or row.get("close") or row.get("ref_price") or row.get("AdjClose")
    out["price"] = _to_float_or_none(price)

    out["as_of"] = row.get("as_of") or GENERATED_AT

    out["prediction_target"]       = PREDICTION_TARGET
    out["reference_price_basis"]   = REFERENCE_PRICE_BASIS
    out["arima_forecast_basis"]    = ARIMA_FORECAST_BASIS
    out["arima_delta_definition"]  = ARIMA_DELTA_DEFINITION
    return out

def run_inference(tickers: List[str]) -> List[Dict[str, Any]]:
    try:
        rows = _predict(tickers)
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict(orient="records")
        normed = [_normalize_row(r) for r in rows]

        # Enforce correct target date for every row (ignore whatever upstream sent)
        for r in normed:
            _, target = compute_dates_for_row(r.get("ticker", ""))
            r["date"] = target

        return normed
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
            for src in (
                "predictedclose","predclose","yhat","forecast",
                "prediction","pred","arima_pred","price_pred","forecast_price"
            ):
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
                if not date_col or not pred_col:
                    continue
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col, pred_col]).sort_values(date_col)
                if df.empty:
                    continue
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
        if px is None or (isinstance(px, float) and math.isnan(px)):
            return fallback
        return float(px)
    except Exception as e:
        print(f"[dashboard] yfinance failed for {ticker}: {e}")
        return fallback


# ── Output writers ────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "Generated_At",
    "Target_Date",
    "Ticker",
    "Prediction_Direction_Tomorrow",
    "Confidence_Score",
    "Reference_Price_Close",
    "ARIMA_Predicted_Price_TomorrowClose",
    "ARIMA_Change_Pct",
    "Live_Price",
    "Prediction_Target",
    "Reference_Price_Basis",
    "ARIMA_Forecast_Basis"
]

def write_outputs(rows: List[Dict[str, Any]]) -> None:
    PUBLIC.mkdir(exist_ok=True)

    # normalize + keep ARIMA/live fields, coercing to floats
    norm: List[Dict[str, Any]] = []
    for r in rows:
        base = _normalize_row(r)
        base["arima_pred"]      = _to_float_or_none(r.get("arima_pred"))
        base["arima_delta_pct"] = _to_float_or_none(r.get("arima_delta_pct"))
        base["live_price"]      = _to_float_or_none(r.get("live_price"))

        # Ensure date exists & is YYYY-MM-DD (enforce again)
        if not base.get("date"):
            _, target = compute_dates_for_row(base.get("ticker", ""))
            base["date"] = target
        norm.append(base)

    # derived model "as_of" (max of row as_of as ISO if present)
    as_of_values = [r.get("as_of") for r in norm if r.get("as_of")]
    model_as_of = max(as_of_values) if as_of_values else GENERATED_AT

    # sort stable: latest target date first, then ticker
    def _safe_datekey(s: str) -> datetime.datetime:
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            try:
                return datetime.datetime.fromisoformat(s)
            except Exception:
                return datetime.datetime.min
    norm.sort(key=lambda r: (_safe_datekey(r.get("date","")), r.get("ticker","")), reverse=True)

    # rounding for CSV/HTML readability
    for r in norm:
        if r.get("arima_pred") is not None:      r["arima_pred"]      = round(float(r["arima_pred"]), 3)
        if r.get("arima_delta_pct") is not None: r["arima_delta_pct"] = round(float(r["arima_delta_pct"]), 2)
        if r.get("live_price") is not None:      r["live_price"]      = round(float(r["live_price"]), 3)
        if r.get("price") is not None:           r["price"]           = round(float(r["price"]), 2)
        if r.get("probability") is not None:     r["probability"]     = float(r["probability"])  # keep 0..1

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = PUBLIC / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in norm:
            w.writerow({
                "Generated_At": GENERATED_AT,
                "Target_Date": r.get("date", ""),
                "Ticker": r.get("ticker", ""),
                "Prediction_Direction_Tomorrow": r.get("prediction", ""),
                "Confidence_Score": r.get("probability"),
                "Reference_Price_Close": r.get("price"),
                "ARIMA_Predicted_Price_TomorrowClose": r.get("arima_pred"),
                "ARIMA_Change_Pct": r.get("arima_delta_pct"),
                "Live_Price": r.get("live_price"),
                "Prediction_Target": r.get("prediction_target", PREDICTION_TARGET),
                "Reference_Price_Basis": r.get("reference_price_basis", REFERENCE_PRICE_BASIS),
                "ARIMA_Forecast_Basis": r.get("arima_forecast_basis", ARIMA_FORECAST_BASIS),
            })

    # ── JSON (rich) ───────────────────────────────────────────────────────────
    data = {
        "generated_at": GENERATED_AT,
        "as_of": model_as_of,
        "count": len(norm),
        "predictions": norm,
        "schema": {
            "prediction_target": PREDICTION_TARGET,
            "reference_price_basis": REFERENCE_PRICE_BASIS,
            "arima_forecast_basis": ARIMA_FORECAST_BASIS,
            "arima_delta_definition": ARIMA_DELTA_DEFINITION
        }
    }
    (PUBLIC / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── HTML (static) ─────────────────────────────────────────────────────────
    rows_html: List[str] = []
    for r in norm:
        pred = (r.get("prediction") or "").upper()
        pill = "pill" if pred not in ("UP","BUY","DOWN","SELL") else ("pill up" if pred in ("UP","BUY") else "pill down")
        delta_txt = "" if r.get("arima_delta_pct") is None else f"{_fmt_num(r['arima_delta_pct'], 2)}%"
        rows_html.append(f"""
<tr>
  <td><span class="muted">{r.get('date','')}</span></td>
  <td class="ticker">{r.get('ticker','')}</td>
  <td><span class="{pill}">{pred}</span></td>
  <td class="right">{_fmt_prob_pct(r.get('probability'))}</td>
  <td class="right" title="{REFERENCE_PRICE_BASIS}">{_fmt_num(r.get('price'), 2)}</td>
  <td class="right" title="{ARIMA_FORECAST_BASIS}">{_fmt_num(r.get('arima_pred'), 3)}</td>
  <td class="right" title="{ARIMA_DELTA_DEFINITION}">{delta_txt}</td>
  <td class="right">{_fmt_num(r.get('live_price'), 3)}</td>
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
.footer {{ color:var(--muted); font-size:12px; margin-top:12px; line-height:1.4; }}
.legend small {{ display:block; color:#9aa6b2; }}
</style></head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>QuantCast — Predictions</h1>
      <div class="meta">Generated: <strong>{GENERATED_AT}</strong> | Model as-of: <strong>{model_as_of}</strong> | Rows: <strong>{len(norm)}</strong></div>
      <div class="legend" style="margin:8px 0 16px;">
        <small><strong>Prediction Target:</strong> {PREDICTION_TARGET}</small>
        <small><strong>Reference Price:</strong> {REFERENCE_PRICE_BASIS}</small>
        <small><strong>ARIMA Forecast:</strong> {ARIMA_FORECAST_BASIS}</small>
        <small><strong>Δ%:</strong> {ARIMA_DELTA_DEFINITION}</small>
      </div>
      <div style="overflow:auto; max-height:75vh;">
        <table>
          <thead>
            <tr>
              <th>Target Date</th>
              <th>Ticker</th>
              <th>Prediction (Tomorrow)</th>
              <th class="right">Confidence</th>
              <th class="right">Reference Close</th>
              <th class="right">ARIMA Pred (Tomorrow Close)</th>
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tickers_env = os.getenv("TICKERS", ",".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]

    # 1) predictions (normalized & target dates enforced)
    rows = run_inference(tickers)

    # 2) ARIMA map
    arima_map = load_arima_map()
    print("[dashboard] ARIMA tickers loaded:", len(arima_map), arima_map.keys())

    # 3) merge ARIMA + live price + Δ% + enforce dates (safety double-lock)
    for r in rows:
        tkr = r.get("ticker")
        ref_price = r.get("price")
        apred = arima_map.get(tkr)

        r["arima_pred"] = apred
        r["arima_delta_pct"] = None
        if apred is not None and ref_price not in (None, 0):
            try:
                r["arima_delta_pct"] = 100.0 * (float(apred) - float(ref_price)) / float(ref_price)
            except Exception:
                r["arima_delta_pct"] = None

        lp = fetch_live_price(tkr, fallback=ref_price)
        r["live_price"] = lp if lp is not None else ref_price

        # Enforce target date (ignore upstream)
        _, target = compute_dates_for_row(tkr or "")
        r["date"] = target

        if r.get("prediction"):
            r["prediction"] = r["prediction"].upper()

    # 4) write artifacts
    write_outputs(rows)


if __name__ == "__main__":
    main()
