# 📈 QuantCast – Stock Price Forecasting with Hybrid Models (AWS + ML)

## 📌 Project Overview
**QuantCast** is an end-to-end pipeline that predicts **next-day returns** and **volatility** for a small watchlist of assets (e.g., AAPL, MSFT, BTC-USD).  
It combines **classical time-series models** (ARIMA/GARCH) with **deep learning (LSTM)** and is designed to be **AWS-native** for scalability.  

The project demonstrates:
- Data ingestion & feature engineering
- Hybrid quant + ML modeling
- Backtesting & performance evaluation
- (Planned) AWS deployment with SageMaker, Lambda, and QuickSight

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Statsmodels, PyTorch, Scikit-learn)
- **yfinance** – historical market data
- **Matplotlib** – local visualizations
- **AWS (Planned)**:
  - S3 (data lake)
  - Lambda + EventBridge (automation)
  - SageMaker (training & deployment)
  - DynamoDB (prediction storage)
  - QuickSight (dashboarding)

---

## 📂 Project Structure
QuantCast/
├── data/
│ ├── raw/ # raw market data from yfinance
│ └── processed/ # engineered features (Parquet/CSV)
│
├── etl/
│ ├── data_collection.py # fetch daily OHLCV data
│ └── feature_engineering.py# build features (returns, RSI, MA, volatility)
│
├── modeling/
│ ├── model_training.py # ARIMA + LSTM training & backtesting
│ └── deploy_model.py # package/export model artifacts
│
├── serving/
│ └── batch_infer.py # daily inference (planned for AWS Lambda)
│
├── dashboard/
│ └── dashboard.py # prototype dashboard (local/Matplotlib → QuickSight later)
│
├── infra/
│ └── README-aws-setup.md # AWS setup notes (S3, SageMaker, Lambda)
│
├── requirements.txt # dependencies
└── README.md # this file

---

## ⚙️ How It Works
1. **Data Collection**
   - Pulls historical OHLCV data from `yfinance`.
   - Saves files to `data/raw/`.

2. **Feature Engineering**
   - Creates technical indicators:
     - Log returns
     - Rolling mean/volatility
     - RSI
   - Saves to `data/processed/`.

3. **Model Training**
   - Classical: ARIMA for returns, GARCH for volatility.
   - Deep Learning: LSTM on feature windows.
   - Backtests with walk-forward validation.
   - Metrics: MAE, directional accuracy, Sharpe ratio.

4. **Serving (Planned)**
   - Deploys best model to **AWS SageMaker Endpoint**.
   - **Lambda** runs daily to fetch new data and store predictions.

5. **Dashboard (Planned)**
   - Local prototype in Matplotlib.
   - AWS QuickSight for live trader dashboard.

---

## 🚀 Getting Started

### 1. Clone Repo
```bash
git clone https://github.com/yourusername/QuantCast.git
cd QuantCast
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run Data Collection
```bash
python etl/data_collection.py
```
4. Run Feature Engineering
```bash
python etl/feature_engineering.py
```

5. Train Models
```bash
python modeling/model_training.py
```
📊 Example Output (local)

Features CSV/Parquet with returns, RSI, moving averages.

Plots:

Predicted vs actual returns.

Directional accuracy (up/down).

Strategy P&L curve.

🧠 Learning Outcomes

Build quant-style ML models for financial time series.

Apply walk-forward backtesting to avoid leakage.

Practice AWS ML pipeline design (S3 → SageMaker → Lambda → QuickSight).

Deploy ML models as daily prediction services.

🔮 Roadmap

 Local data collection & feature engineering

 Baseline ARIMA + LSTM training

 Backtesting & performance evaluation

 AWS S3 integration

 SageMaker training/deployment

 Lambda daily batch predictions

 QuickSight dashboard

📜 License

MIT License – feel free to use and adapt.


---

Do you want me to also include a **diagram (ASCII or simple image)** showing the pipeline flow (Data → Features → Models → Predictions → Dashboard) so it looks more “executive ready” for recruiters?