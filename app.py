import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pairs_backtest import fetch_prices, estimate_hedge_ratio, backtest_pair, BacktestParams

st.title("📈 Pairs Trading Backtester")

st.markdown("Backtest a simple statistical arbitrage strategy between two tickers (e.g., ETFs or stocks).")

# User inputs
ticker1 = st.text_input("Enter ticker 1", "NIFTYBEES.NS")
ticker2 = st.text_input("Enter ticker 2", "BANKBEES.NS")
start_date = st.date_input("Start date", value=pd.to_datetime("2018-01-01"))

if st.button("Run Backtest"):
    try:
        st.write(f"Fetching data for {ticker1} and {ticker2}...")
        px = fetch_prices([ticker1, ticker2], str(start_date), None)

        if ticker1 not in px.columns or ticker2 not in px.columns:
            st.error("One of the tickers not found. Try again with valid NSE/BSE tickers.")
        else:
            hr = estimate_hedge_ratio(px[ticker1], px[ticker2])

            params = BacktestParams()
            result = backtest_pair(px[ticker1], px[ticker2], hr, params)

            # Show metrics
            st.subheader("Backtest Metrics")
            st.json(result.metrics)

            # Plot equity curve
            fig, ax = plt.subplots()
            ax.plot(result.equity.index, result.equity.values, label="Equity (₹)")
            ax.set_title(f"Equity Curve: {ticker1} / {ticker2}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
