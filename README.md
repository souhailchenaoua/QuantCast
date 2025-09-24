# 📈 QuantCast — Stock Price Forecasting Dashboard

QuantCast is a hybrid ML + time-series pipeline that predicts **next-day price direction** and **compares with ARIMA forecasts** for selected tickers (AAPL, INTC, etc.).  
Outputs are published as a **static dashboard** using GitHub Pages.

---

## 🚀 Project Structure

```
QuantCast/
│
├── dashboard/                 # Dashboard renderer (CSV, JSON, HTML)
│   └── dashboard.py
│
├── modeling/                  # ML + ARIMA models
│   ├── ml_direction_classifier.py
│   └── time_series_arima.py
│
├── data/
│   └── processed/             # Processed datasets + ARIMA outputs
│       ├── prices_features_public.csv
│       └── arima_results_public.csv
│
├── scripts/                   # (optional) one-off utility scripts
│
├── public/                    # Generated artifacts (local runs)
│   ├── predictions.csv
│   ├── data.json
│   └── index.html
│
├── docs/                      # Published folder for GitHub Pages
│   └── index.html             # (copied from public/)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-user>/QuantCast.git
   cd QuantCast
   ```

2. **Create venv + install deps**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows
   pip install -r requirements.txt
   ```

3. **Check dependencies**  
   Make sure these libs are available: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `yfinance`.

---

## 🛠️ How to Run Locally

1. **Generate features + ARIMA outputs**
   ```powershell
   python data/etl/feature_engineering.py
   python modeling/time_series_arima.py
   ```

   This produces:
   - `data/processed/prices_features_public.csv`
   - `data/processed/arima_results_public.csv`

2. **Build the dashboard**
   ```powershell
   $env:PRICES_FEATURES_PATH = "data\processed\prices_features_public.csv"
   $env:ARIMA_PUBLIC_PATH    = "data\processed\arima_results_public.csv"
   python -m dashboard.dashboard
   ```

   Outputs go into `public/`:
   - `predictions.csv`
   - `data.json`
   - `index.html`

3. **Preview locally**  
   Open `public/index.html` in your browser.

---

## 🌐 Publishing to GitHub Pages

We no longer use AWS S3/Fargate. Deployment is **GitHub Pages only**.

1. Copy generated files to `/docs`:
   ```powershell
   mkdir docs 2>$null
   Copy-Item public\* docs\ -Recurse -Force
   ```

2. Commit + push:
   ```powershell
   git add docs/*
   git commit -m "update dashboard snapshot"
   git push
   ```

3. Ensure GitHub Pages is set to **Branch = `main`, Folder = `/docs`** in repo settings.

4. Open your site:  
   ```
   https://<your-user>.github.io/QuantCast/
   ```

---

## 🔍 Notes

- `public/` = local build outputs (not served directly).  
- `docs/` = published snapshot for Pages.  
- `data/processed/` must contain the CSVs before building the dashboard.  
- AWS/S3 references in older code are deprecated and no longer required.  
