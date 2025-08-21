# pairs_backtest.py
# --------------------------------------------
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
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os

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
    excess = daily_returns - rf
    vol = annualize_vol(excess, trading_days)
    if vol == 0:
        return 0.0
    return (excess.mean() * trading_days) / vol

def max_drawdown(equity_curve: pd.Series) -> Tuple[float, float, float]:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    trough_idx = drawdown.idxmin()
    max_dd = drawdown.min()
    peak_idx = equity_curve.loc[:trough_idx].idxmax()
    return float(max_dd), peak_idx, trough_idx

def zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=0)
    return (series - mean) / std

# ---------- Data ----------

def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance.
    """
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        px = data['Close'].copy()
    else:
        px = data.copy()
    px = px.dropna(how="all")
    return px

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
    Handles missing values safely.
    """
    df = pd.concat([y, x], axis=1).dropna()
    if df.empty or df.shape[0] < 30:  # not enough data
        return 1.0
    y_clean = df.iloc[:, 0]
    x_clean = df.iloc[:, 1]
    x_with_const = sm.add_constant(x_clean.values)
    model = sm.OLS(y_clean.values, x_with_const, missing="drop").fit()
    return float(model.params[1])

def scan_cointegration(px: pd.DataFrame,
                       tickers: List[str],
                       max_pairs: int = 10,
                       min_history: int = 252) -> List[CointResult]:
    """
    Runs Engle–Granger cointegration test for each pair and returns top pairs.
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
            try:
                stat, pvalue, _ = coint(df.iloc[:,0], df.iloc[:,1])
                hr = estimate_hedge_ratio(df.iloc[:,0], df.iloc[:,1])
                results.append(CointResult(x=b, y=a, pvalue=float(pvalue), hedge_ratio=hr))
            except Exception:
                continue
    results.sort(key=lambda r: r.pvalue)
    return results[:max_pairs]

# ---------- Backtest ----------

@dataclass
class BacktestParams:
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    tc_bps: float = 1.0
    capital: float = 1_000_000.0
    gross_leverage: float = 1.0
    rebalance_every_n_days: int = 1

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
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().copy()
    if df.empty or len(df) < params.lookback + 5:
        raise ValueError("Not enough overlapping data for backtest.")

    # Spread & Z
    spread = df["y"] - hedge_ratio * df["x"]
    Z = zscore(spread, params.lookback)

    state = 0
    pos_y = np.zeros(len(df))
    pos_x = np.zeros(len(df))

    last_pos_y = 0.0
    last_pos_x = 0.0
    rebal_counter = 0
    trades = []

    r_y = df["y"].pct_change().fillna(0.0)
    r_x = df["x"].pct_change().fillna(0.0)

    def normalized_weights(sign: int) -> Tuple[float, float]:
        wy = 1.0 * sign
        wx = -hedge_ratio * sign
        gross = abs(wy) + abs(wx)
        wy /= gross
        wx /= gross
        wy *= params.gross_leverage
        wx *= params.gross_leverage
        return wy, wx

    for t, (date, row) in enumerate(df.iterrows()):
        rebal_counter += 1
        z = Z.iloc[t]

        if state == 0:
            if z <= -params.entry_z:
                state = +1
            elif z >= params.entry_z:
                state = -1
        elif state == +1:
            if abs(z) <= params.exit_z or abs(z) >= params.stop_z:
                state = 0
        elif state == -1:
            if abs(z) <= params.exit_z or abs(z) >= params.stop_z:
                state = 0

        if state == 0:
            wy, wx = 0.0, 0.0
        elif state == +1:
            wy, wx = normalized_weights(+1)
        else:
            wy, wx = normalized_weights(-1)

        if params.rebalance_every_n_days > 1:
            if (rebal_counter % params.rebalance_every_n_days) != 1:
                wy, wx = last_pos_y, last_pos_x
            else:
                rebal_counter = 1

        pos_y[t] = wy
        pos_x[t] = wx

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

    turnover = (df["pos_y"].diff().abs().fillna(abs(df["pos_y"])) +
                df["pos_x"].diff().abs().fillna(abs(df["pos_x"])))
    daily_cost = turnover * (params.tc_bps / 10_000.0)

    strat_ret = df["pos_y"].shift(1).fillna(0.0) * r_y + df["pos_x"].shift(1).fillna(0.0) * r_x
    strat_ret_net = strat_ret - daily_cost

    equity = (1.0 + strat_ret_net).cumprod() * params.capital

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

# ---------- Plotting (improved) ----------

def plot_equity(result: BacktestResult, title: str, fname: Optional[str] = None):
    plt.figure(figsize=(12, 6))
    plt.plot(result.equity.index, result.equity.values, label="Equity (₹)")
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Better date formatting
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    if fname:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

def plot_zscore(result: BacktestResult, params: BacktestParams, title: str, fname: Optional[str] = None):
    plt.figure(figsize=(12, 5))
    plt.plot(result.zscores.index, result.zscores.values, label="Z-score", color="blue")

    # Threshold lines
    plt.axhline(params.entry_z, linestyle="--", color="red", label="Entry")
    plt.axhline(-params.entry_z, linestyle="--", color="red")
    plt.axhline(params.exit_z, linestyle=":", color="green", label="Exit")
    plt.axhline(-params.exit_z, linestyle=":", color="green")
    plt.axhline(params.stop_z, linestyle="-.", color="black", label="Stop")
    plt.axhline(-params.stop_z, linestyle="-.", color="black")

    # Dynamic scaling for visibility
    zmin, zmax = result.zscores.min(), result.zscores.max()
    plt.ylim(zmin - 1, zmax + 1)

    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Z")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Better date formatting
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    if fname:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

# ---------- Example runner ----------

def run_demo():
    tickers = ["NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "ITBEES.NS"]
    px = fetch_prices(tickers, start="2016-01-01", end=None).dropna(how="any")

    coint_hits = scan_cointegration(px, tickers, max_pairs=5, min_history=500)
    print("Top cointegrated pairs (by p-value):")
    for r in coint_hits:
        print(f"{r.y} ~ {r.x} | p={r.pvalue:.4f} | hedge_ratio={r.hedge_ratio:.3f}")

    if not coint_hits:
        raise SystemExit("No cointegrated pairs found. Try different tickers or date range.")

    best = coint_hits[0]
    y = px[best.y]
    x = px[best.x]

    split_date = y.index[int(len(y) * 0.7)]
    y_train, y_test = y.loc[:split_date], y.loc[split_date:]
    x_train, x_test = x.loc[:split_date], x.loc[split_date:]

    hr = estimate_hedge_ratio(y_train, x_train)

    params = BacktestParams(
        lookback=60,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        tc_bps=1.5,
        capital=1_000_000.0,
        gross_leverage=1.0,
        rebalance_every_n_days=1
    )

    result = backtest_pair(y_test, x_test, hr, params)

    print("\n=== Backtest Metrics (Test) ===")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v:.4f}")
        else:
            print(f"{k:15s}: {v}")

    os.makedirs("figures", exist_ok=True)
    plot_equity(result, f"Equity Curve: {best.y} / {best.x}", fname="figures/equity.png")
    plot_zscore(result, params, f"Z-Score: {best.y} / {best.x}", fname="figures/zscore.png")

if __name__ == "__main__":
    run_demo()
