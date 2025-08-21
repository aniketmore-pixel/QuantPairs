# Pairs Trading (Stat Arb) – cointegration scan + backtest
# Author: Aniket More
# --------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ---------- Utilities ----------

def annualize_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    if daily_returns.empty:
        return 0.0
    cum = (1 + daily_returns).prod()
    years = len(daily_returns) / trading_days
    if years <= 0:
        return 0.0
    return cum ** (1 / years) - 1

def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    if daily_returns.std(ddof=0) == 0:
        return 0.0
    return daily_returns.std(ddof=0) * math.sqrt(trading_days)

def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    # rf is daily risk-free (approx 0 for demo)
    excess = daily_returns - rf
    vol = annualize_vol(excess, trading_days)
    if vol == 0:
        return 0.0
    return (excess.mean() * trading_days) / vol

def max_drawdown(equity_curve: pd.Series) -> Tuple[float, float, float]:
    # returns (max_dd, peak_date, trough_date)
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    trough_idx = drawdown.idxmin()
    max_dd = drawdown.min()
    # find the last peak before trough
    peak_idx = equity_curve.loc[:trough_idx].idxmax()
    return float(max_dd), peak_idx, trough_idx

def zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=0)
    return (series - mean) / std

# ---------- Data ----------

def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetches adjusted close prices for given tickers from Yahoo Finance.
    """
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        px = data['Close'].copy()
    else:
        # when a single ticker returns a Series
        px = data.copy()
    return px.dropna(how="all")

# ---------- Cointegration Scan ----------

@dataclass
class CointResult:
    x: str
    y: str
    pvalue: float
    hedge_ratio: float

def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """
    Regress y on x -> y = a + b*x + e. Hedge ratio = b
    """
    x = sm.add_constant(x.values)
    model = sm.OLS(y.values, x).fit()
    return float(model.params[1])

def scan_cointegration(px: pd.DataFrame,
                       tickers: List[str],
                       max_pairs: int = 10,
                       min_history: int = 252) -> List[CointResult]:
    """
    Runs Engle–Granger cointegration test for each pair and returns top pairs by p-value.
    """
    results = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            a, b = tickers[i], tickers[j]
            y = px[a].dropna()
            x = px[b].dropna()
            df = pd.concat([y, x], axis=1).dropna()
            if len(df) < min_history:
                continue
            stat, pvalue, _ = coint(df.iloc[:,0], df.iloc[:,1])
            hr = estimate_hedge_ratio(df.iloc[:,0], df.iloc[:,1])
            results.append(CointResult(x=b, y=a, pvalue=float(pvalue), hedge_ratio=hr))
    # sort by p-value ascending (stronger cointegration)
    results.sort(key=lambda r: r.pvalue)
    return results[:max_pairs]

# ---------- Backtest ----------

@dataclass
class BacktestParams:
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    tc_bps: float = 1.0     # transaction cost (per leg) in basis points
    capital: float = 1_000_000.0
    gross_leverage: float = 1.0
    rebalance_every_n_days: int = 1  # 1=daily

@dataclass
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    zscores: pd.Series
    spread: pd.Series

def backtest_pair(y: pd.Series,
                  x: pd.Series,
                  hedge_ratio: float,
                  params: BacktestParams) -> BacktestResult:
    """
    Simple state-machine backtest for pairs trading on a spread = y - b*x.
    Positions are scaled to target gross leverage; costs applied on turnover.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().copy()
    # Spread & Z
    spread = df["y"] - hedge_ratio * df["x"]
    Z = zscore(spread, params.lookback)

    # States: 0 = flat, +1 = long spread (long y, short b*x), -1 = short spread
    state = 0
    pos_y = np.zeros(len(df))
    pos_x = np.zeros(len(df))

    last_pos_y = 0.0
    last_pos_x = 0.0

    rebal_counter = 0
    trades = []

    # Daily returns
    r_y = df["y"].pct_change().fillna(0.0)
    r_x = df["x"].pct_change().fillna(0.0)

    # Target raw weights before normalization: long spread => w_y=+1, w_x=-b
    def normalized_weights(sign: int) -> Tuple[float, float]:
        wy = 1.0 * sign
        wx = -hedge_ratio * sign
        gross = abs(wy) + abs(wx)
        wy /= gross
        wx /= gross
        # scale by desired gross leverage
        wy *= params.gross_leverage
        wx *= params.gross_leverage
        return wy, wx

    for t, (date, row) in enumerate(df.iterrows()):
        rebal_counter += 1
        z = Z.iloc[t]

        # Entry / Exit / Stop
        if state == 0:
            if z <= -params.entry_z:
                state = +1
            elif z >= params.entry_z:
                state = -1
        elif state == +1:
            if abs(z) <= params.exit_z:
                state = 0
            elif abs(z) >= params.stop_z:
                state = 0
        elif state == -1:
            if abs(z) <= params.exit_z:
                state = 0
            elif abs(z) >= params.stop_z:
                state = 0

        # Positions (rebalance on schedule, otherwise hold)
        if state == 0:
            wy, wx = 0.0, 0.0
        elif state == +1:
            wy, wx = normalized_weights(+1)
        else:  # -1
            wy, wx = normalized_weights(-1)

        if params.rebalance_every_n_days > 1:
            if (rebal_counter % params.rebalance_every_n_days) != 1:
                # carry previous positions if not rebalance day
                wy, wx = last_pos_y, last_pos_x
            else:
                rebal_counter = 1

        pos_y[t] = wy
        pos_x[t] = wx

        # Record trades when position changes
        if (wy != last_pos_y) or (wx != last_pos_x):
            trades.append({
                "date": date,
                "pos_y": wy,
                "pos_x": wx,
                "zscore": z,
                "state": state
            })

        last_pos_y, last_pos_x = wy, wx

    df["pos_y"] = pos_y
    df["pos_x"] = pos_x

    # Turnover & transaction costs (per-leg bps on notional change)
    turnover = (df["pos_y"].diff().abs().fillna(abs(df["pos_y"])) +
                df["pos_x"].diff().abs().fillna(abs(df["pos_x"])))
    # cost per day (bps -> fraction)
    daily_cost = turnover * (params.tc_bps / 10_000.0)

    # Strategy daily return
    strat_ret = df["pos_y"].shift(1).fillna(0.0) * r_y + df["pos_x"].shift(1).fillna(0.0) * r_x
    strat_ret_net = strat_ret - daily_cost

    equity = (1.0 + strat_ret_net).cumprod() * params.capital

    # Metrics
    ann_ret = annualize_return(strat_ret_net)
    ann_vol = annualize_vol(strat_ret_net)
    sharpe = sharpe_ratio(strat_ret_net)
    mdd, peak_dt, trough_dt = max_drawdown(equity)

    metrics = {
        "CAGR": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "PeakDate": str(peak_dt) if peak_dt is not None else "",
        "TroughDate": str(trough_dt) if trough_dt is not None else "",
        "WinRate": float((strat_ret_net > 0).mean()),
        "AvgDailyTurnover": float(turnover.mean()),
        "Trades": int(len(trades))
    }

    trades_df = pd.DataFrame(trades).set_index("date") if trades else pd.DataFrame(columns=["date","pos_y","pos_x","zscore","state"]).set_index("date")

    return BacktestResult(
        equity=equity,
        daily_returns=strat_ret_net,
        trades=trades_df,
        metrics=metrics,
        zscores=Z,
        spread=spread
    )

# ---------- Plotting ----------

def plot_equity(result: BacktestResult, title: str, fname: Optional[str] = None):
    plt.figure(figsize=(10, 5))
    plt.plot(result.equity.index, result.equity.values, label="Equity (₹)")
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if fname:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

def plot_zscore(result: BacktestResult, params: BacktestParams, title: str, fname: Optional[str] = None):
    plt.figure(figsize=(10, 4))
    plt.plot(result.zscores.index, result.zscores.values, label="Z-score")
    plt.axhline(params.entry_z, linestyle="--")
    plt.axhline(-params.entry_z, linestyle="--")
    plt.axhline(params.exit_z, linestyle=":")
    plt.axhline(-params.exit_z, linestyle=":")
    plt.axhline(params.stop_z, linestyle="-.")
    plt.axhline(-params.stop_z, linestyle="-.")
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Z")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if fname:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

# ---------- Example runner (can be called as a script) ----------

def run_demo():
    # 1) Choose tickers.
    # For India, start with ETFs (stable tickers): NIFTYBEES.NS, BANKBEES.NS, JUNIORBEES.NS, ITBEES.NS etc.
    tickers = ["NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "ITBEES.NS"]

    # 2) Fetch data (change dates as needed)
    px = fetch_prices(tickers, start="2016-01-01", end=None)
    px = px.dropna(how="any")

    # 3) Scan for cointegrated pairs
    coint_hits = scan_cointegration(px, tickers, max_pairs=5, min_history=500)
    print("Top cointegrated pairs (by p-value):")
    for r in coint_hits:
        print(f"{r.y} ~ {r.x} | p={r.pvalue:.4f} | hedge_ratio={r.hedge_ratio:.3f}")

    # 4) Pick the best one (lowest p-value)
    if not coint_hits:
        raise SystemExit("No cointegrated pairs found. Try different tickers or date range.")

    best = coint_hits[0]
    y = px[best.y]
    x = px[best.x]

    # 5) Train/Test split (walk-forward simple)
    split_date = y.index[int(len(y) * 0.7)]
    y_train, y_test = y.loc[:split_date], y.loc[split_date:]
    x_train, x_test = x.loc[:split_date], x.loc[split_date:]

    # Re-estimate hedge ratio on training only
    hr = estimate_hedge_ratio(y_train, x_train)

    params = BacktestParams(
        lookback=60,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        tc_bps=1.5,          # ~1.5 bps per leg
        capital=1_000_000.0,
        gross_leverage=1.0,
        rebalance_every_n_days=1
    )

    # 6) Backtest on test set
    result = backtest_pair(y_test, x_test, hr, params)

    # 7) Print metrics
    print("\n=== Backtest Metrics (Test) ===")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v:.4f}")
        else:
            print(f"{k:15s}: {v}")

    # 8) Plots
    import os
    os.makedirs("figures", exist_ok=True)

    plot_equity(result, f"Equity Curve: {best.y} / {best.x}", fname="figures/equity.png")
    plot_zscore(result, params, f"Z-Score: {best.y} / {best.x}", fname="figures/zscore.png")

if __name__ == "__main__":
    run_demo()