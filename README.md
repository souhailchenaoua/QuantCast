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

# QuantCast — Update & Publish Playbook

This is the **exact sequence** to refresh data, rebuild models, regenerate the dashboard, and publish it to GitHub Pages.

> **Important:** Always do a final `git add .` before committing — this ensures **all** changed files (including `public/`) are included. This solved previous publish issues.

---

## Prereqs
- Open **PowerShell** at the repo root (e.g., `C:\Users\tonyail\Desktop\QuantCast`).
- Activate your virtualenv:  
  ```powershell
  .\.venv\Scripts\activate
  ```
- Verify you can run Python: `python --version`

---

## Full refresh → publish (daily run)

```powershell
# 0) Repo + venv
cd C:\Users\tonyail\Desktop\QuantCast
.\.venv\Scripts\activate
git pull

# 1) Run config (edit as you like)
$env:TICKERS       = "AAPL,MSFT,INTC,NVDA,META,AMZN,GOOGL,TSLA,BTC-USD,ETH-USD"
$env:DIR_THRESHOLD = "0.50"   # classifier UP/DOWN cutoff (e.g., 0.55 to be stricter)

# 2) ETL → data\raw\* + combined
python .\etl\data_collection.py

# 3) Feature engineering → data\processed\prices_features.csv
python .\etl\feature_engineering.py

# 4) Direction classifier (updates assets/reports)
python -m modeling.ml_direction_classifier

# 5) ARIMA training (grids + results) → data\processed\arima_grid_*.csv, arima_results.csv
python .\modeling\time_series_arima.py

# 6) Build ARIMA public file used by the dashboard (writes arima_results_public.csv)
python .\scripts\arima_one_step_public.py

# 7) Wire dashboard to generated files (if not already defaulted in code)
$env:PRICES_FEATURES_PATH = "data\processed\prices_features.csv"
$env:ARIMA_PUBLIC_PATH    = "data\processed\arima_results_public.csv"

# 8) Generate the static site → public\predictions.csv, data.json, index.html
python -m dashboard.dashboard

# 9) Publish (GitHub Pages via Actions or /docs branch)
git add .
git commit -m "Publish: refreshed ETL, features, ARIMA public, classifier, and dashboard"
git push
```
After pushing, the **GitHub Pages** workflow deploys `public/`. If you use the **branch mode** (`/docs`), see the section below.

---

## Quick refresh & publish (fast path)

```powershell
cd C:\Users\tonyail\Desktop\QuantCast
.\.venv\Scripts\activate

# Rebuild ARIMA public (optional but recommended)
python .\scripts\arima_one_step_public.py

# Build site
$env:TICKERS = "AAPL,MSFT,INTC,NVDA,META,AMZN,GOOGL,TSLA,BTC-USD,ETH-USD"
$env:PRICES_FEATURES_PATH = "data\processed\prices_features.csv"
$env:ARIMA_PUBLIC_PATH    = "data\processed\arima_results_public.csv"
python -m dashboard.dashboard

# Publish
git add .
git commit -m "Publish: refreshed predictions with ARIMA + classifier"
git push
```

---

## GitHub Pages — deployment modes

### A) GitHub Actions (recommended)
1. Ensure `.github/workflows/pages.yml` exists with trigger on `public/**` (see below).
2. Repo Settings → **Pages** → **Source = GitHub Actions**.
3. Any commit that changes `public/**` will deploy automatically.

Minimal `pages.yml`:
```yaml
name: Deploy static site (public/) to GitHub Pages
on:
  push:
    branches: [ main ]
    paths:
      - 'public/**'
      - '.github/workflows/pages.yml'
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: "pages"
  cancel-in-progress: false
jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: public
      - id: deployment
        uses: actions/deploy-pages@v4
```

### B) Branch mode (serve from `/docs`)
If your Pages is set to “Deploy from a branch”, mirror `public/` to `docs/`:
```powershell
robocopy public docs /MIR
git add docs
git commit -m "Publish: mirror public/ to docs/ for branch deploy"
git push
```
Then set **Settings → Pages → Deploy from a branch → main /docs**.

---

## Troubleshooting (most common)

- **I don’t see new tickers on the site**
  - Make sure the dashboard used your list: set `TICKERS` env var as above, or edit `dashboard\dashboard.py` → `DEFAULT_TICKERS` to include all 10.
  - Confirm locally:  
    ```powershell
    Import-Csv public\predictions.csv | Group-Object Ticker | Sort Name | ft Name,Count
    ```

- **ARIMA columns empty**
  - Ensure `ARIMA_PUBLIC_PATH` points to `data\processed\arima_results_public.csv`.
  - Rebuild it: `python .\scripts\arima_one_step_public.py`.

- **Publish didn’t update**
  - Always run **`git add .`** before committing.
  - If using Actions: check the Actions tab for the Pages workflow run.
  - If using branch mode: mirror to `/docs` as shown above.
  - Bust cache by appending a timestamp comment to `public\index.html`:
    ```powershell
    Add-Content .\public\index.html "`n<!-- build $(Get-Date -Format s) -->"
    git add .\public\index.html
    git commit -m "Publish: cache bust"
    git push
    ```

---

## Useful toggles

- **Watchlist**: `$env:TICKERS="AAPL,MSFT,INTC,NVDA,META,AMZN,GOOGL,TSLA,BTC-USD,ETH-USD"`  
- **Classifier threshold**: `$env:DIR_THRESHOLD="0.55"`  
- **Inputs for dashboard**:  
  - `$env:PRICES_FEATURES_PATH="data\processed\prices_features.csv"`  
  - `$env:ARIMA_PUBLIC_PATH="data\processed\arima_results_public.csv"`

---

## Verification snippet

```powershell
# Verify file the site will publish
Import-Csv .\public\predictions.csv | Group-Object Ticker | Sort Name | Format-Table Name,Count

# Verify last commit included public/
git log -1 --name-status | findstr /I public
```

---

**Remember:** the last step before commit is **`git add .`**.
