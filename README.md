# 📈 QuantPairs

QuantPairs is a **statistical arbitrage backtesting tool** built with
**Streamlit** and **Python**.\
It allows users to test simple **pairs trading strategies** using
cointegration between two stocks or ETFs, with metrics and
visualization.

------------------------------------------------------------------------

## 📸 Screenshots
> <img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/2b6ad5ee-b803-4bd8-a054-4c39616768fb" />
> <img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/0b021739-26ab-45fb-bb02-065b46c30720" />
> <img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/f00cf336-e945-4825-b22e-a8ca336f7056" />

------------------------------------------------------------------------

## 🚀 Features

-   Fetch historical price data from **Yahoo Finance** (`yfinance`).
-   Estimate **hedge ratios** using linear regression.
-   Run a **pairs trading backtest** with configurable parameters:
    -   Lookback window
    -   Entry/Exit/Stop Z-score thresholds
    -   Transaction costs (bps)
    -   Portfolio capital & leverage
-   Interactive **Streamlit dashboard** for backtest results.
-   Metrics: CAGR, Volatility, Sharpe Ratio, Max Drawdown, Win Rate,
    Turnover, Trade count.
-   Visualizations: Equity curve & Z-score plots.

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── app.py               # Streamlit frontend
    ├── pairs_backtest.py    # Core backtesting + utilities
    ├── requirements.txt     # Dependencies

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository and install dependencies:

``` bash
git clone https://github.com/yourusername/quantpairs.git
cd quantpairs
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Usage

Run the Streamlit app:

``` bash
streamlit run app.py
```

Then open the local URL shown in your terminal (default:
`http://localhost:8501`).

------------------------------------------------------------------------

## 📊 Example Workflow

1.  Enter two tickers (e.g., `NIFTYBEES.NS` and `BANKBEES.NS`).\
2.  Select a start date.\
3.  Click **Run Backtest**.\
4.  View performance metrics and plots:

-   **Equity Curve**\
-   **Z-score chart** (spread divergence & trading signals)

------------------------------------------------------------------------

## 🔮 Future Scope

-   Add support for **multiple pairs portfolio backtesting**.\
-   Integrate with **live data** for paper trading.\
-   Parameter optimization & walk-forward testing.\
-   Export results as **Excel/PDF reports**.\
-   Deploy as a **cloud app (Heroku/Streamlit Cloud)**.

------------------------------------------------------------------------

## 📦 Requirements

Listed in `requirements.txt`:

    streamlit
    yfinance
    pandas
    numpy
    statsmodels
    matplotlib

Install them with:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 👨‍💻 Author

Developed by **Aniket More**.
