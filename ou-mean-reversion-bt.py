#!/usr/bin/env python3
"""
Ornstein-Uhlenbeck Mean Reversion Backtester
============================================
Usage:
  python ou-mean-reversion-bt.py QQQ 1yr
  python ou-mean-reversion-bt.py QQQ IWM 2yr --commission 5 --slippage 1
  python ou-mean-reversion-bt.py SPY 6mo --exit-mode hard --api-key YOUR_KEY
"""

# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 1]  Imports & Dependency Check
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import base64
import io
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import requests
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from scipy import stats
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
except ImportError as exc:
    sys.exit(
        f"\nMissing dependency: {exc.name}\n"
        "Install all required packages:\n"
        "  pip install requests pandas numpy scipy matplotlib rich python-dotenv\n"
    )

warnings.filterwarnings("ignore", category=RuntimeWarning)
console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 2]  Constants & Config Dataclass
# ══════════════════════════════════════════════════════════════════════════════

FMP_BASE_URL      = "https://financialmodelingprep.com/api/v3"
ROLLING_WINDOW    = 60        # days for OU estimation and Hurst
KELLY_LOOKBACK    = 30        # last N closed trades used for Kelly
KELLY_CAP         = 0.50      # half-Kelly safety cap
KELLY_DEFAULT     = 0.10      # allocation fraction before enough trade history
KELLY_MIN_TRADES  = 10        # minimum trades before Kelly activates
Z_ENTRY           = 2.0       # |z| to open a position
Z_EXIT_SOFT       = 1.0       # |z| to close (soft / default)
Z_EXIT_HARD       = 0.0       # |z| to close (hard / aggressive)
TRADING_DAYS_YEAR = 252
HURST_LAGS        = [2, 4, 8, 16, 32]
OUTPUT_DIR        = Path("./output")

TIMEFRAME_MAP = {
    "1yr": (1.00, "1 Year"),
    "2yr": (2.00, "2 Years"),
    "3yr": (3.00, "3 Years"),
    "5yr": (5.00, "5 Years"),
    "6mo": (0.50, "6 Months"),
    "3mo": (0.25, "3 Months"),
    "ytd": (None, "Year to Date"),
}


@dataclass
class Config:
    tickers:        list
    timeframe:      str
    commission_bps: float = 5.0
    slippage_bps:   float = 1.0
    api_key:        str   = ""
    exit_mode:      str   = "soft"
    start_date:     date  = field(default_factory=date.today)
    end_date:       date  = field(default_factory=date.today)
    output_dir:     Path  = field(default_factory=lambda: OUTPUT_DIR)

    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 3]  CLI / Config
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        prog="ou-mean-reversion-bt.py",
        description="Ornstein-Uhlenbeck Mean Reversion Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ou-mean-reversion-bt.py QQQ 1yr
  python ou-mean-reversion-bt.py QQQ IWM 2yr --commission 5 --slippage 1
  python ou-mean-reversion-bt.py SPY 6mo --exit-mode hard
        """,
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="One or more tickers followed by timeframe (e.g. QQQ IWM 1yr)",
    )
    parser.add_argument("--commission",  type=float, default=5.0,
                        help="Commission in basis points (default: 5)")
    parser.add_argument("--slippage",    type=float, default=1.0,
                        help="Slippage in basis points (default: 1)")
    parser.add_argument("--api-key",     type=str,   default=None,
                        help="FMP API key — overrides .env FMP_API_KEY")
    parser.add_argument("--exit-mode",   choices=["soft", "hard"], default="soft",
                        help="Exit at |z|=1 (soft) or |z|=0 (hard)")
    parser.add_argument("--output-dir",  type=Path,  default=OUTPUT_DIR,
                        help="Directory for charts and HTML report (default: ./output)")
    return parser.parse_args()


def resolve_date_range(timeframe: str):
    end = date.today()
    if timeframe == "ytd":
        start = date(end.year, 1, 1)
    else:
        years_back, _ = TIMEFRAME_MAP[timeframe]
        start = end - timedelta(days=int(years_back * 365.25))
    return start, end


def load_config() -> Config:
    load_dotenv()
    args = parse_args()

    tf_lookup = {k.lower(): k for k in TIMEFRAME_MAP}
    last_arg   = args.targets[-1].lower()

    if last_arg in tf_lookup:
        timeframe = tf_lookup[last_arg]
        tickers   = [t.upper() for t in args.targets[:-1]]
    else:
        sys.exit(
            f"\nERROR: '{args.targets[-1]}' is not a recognised timeframe.\n"
            f"Valid options: {list(TIMEFRAME_MAP.keys())}\n"
        )

    if not tickers:
        sys.exit("\nERROR: At least one ticker must be provided before the timeframe.\n")

    api_key = args.api_key or os.getenv("FMP_API_KEY", "")
    if not api_key:
        sys.exit(
            "\nERROR: FMP API key not found.\n"
            "Add FMP_API_KEY=your_key to a .env file in this directory, "
            "or pass --api-key YOUR_KEY\n"
        )

    start, end = resolve_date_range(timeframe)
    output_dir  = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        tickers=tickers,
        timeframe=timeframe,
        commission_bps=args.commission,
        slippage_bps=args.slippage,
        api_key=api_key,
        exit_mode=args.exit_mode,
        start_date=start,
        end_date=end,
        output_dir=output_dir,
    )


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 4]  FMP Data Fetcher
# ══════════════════════════════════════════════════════════════════════════════

def _parse_fmp_response(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index(ascending=True)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df   = df[keep].copy()
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])
    return df


def fetch_ohlcv(ticker: str, start: date, end: date, api_key: str) -> pd.DataFrame:
    url = (
        f"{FMP_BASE_URL}/historical-price-full/{ticker}"
        f"?from={start.isoformat()}&to={end.isoformat()}&apikey={api_key}"
    )
    console.print(f"  [cyan]Fetching {ticker}[/cyan] [{start} → {end}]...", end=" ")

    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                if attempt == 0:
                    console.print("[yellow]rate-limited, retrying in 3s...[/yellow]", end=" ")
                    time.sleep(3)
                    continue
                sys.exit("\nERROR: FMP API rate limit. Wait and retry.\n")
            resp.raise_for_status()
            break
        except requests.exceptions.ConnectionError:
            sys.exit("\nERROR: Cannot reach FMP API. Check your internet connection.\n")
        except requests.exceptions.Timeout:
            sys.exit(f"\nERROR: Request timed out fetching {ticker}.\n")
        except requests.exceptions.HTTPError as exc:
            sys.exit(f"\nHTTP error for {ticker}: {exc}\n")

    payload = resp.json()

    if isinstance(payload, dict) and "Error Message" in payload:
        sys.exit(f"\nFMP API error for {ticker}: {payload['Error Message']}\n")

    if not isinstance(payload, dict) or "historical" not in payload or not payload["historical"]:
        sys.exit(
            f"\nNo price data returned for {ticker}.\n"
            "Check: (1) ticker symbol, (2) date range, (3) your FMP plan tier.\n"
        )

    df = _parse_fmp_response(payload["historical"])
    console.print(f"[green]{len(df)} trading days[/green]")

    min_required = ROLLING_WINDOW + 1
    if len(df) < min_required:
        sys.exit(
            f"\nERROR: {ticker} returned only {len(df)} rows.\n"
            f"Need ≥{min_required} trading days for the rolling window.\n"
            "Try a longer timeframe (e.g. 2yr instead of 1yr).\n"
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 5]  OU Parameter Estimator  (rolling 60-day AR(1) OLS)
# ══════════════════════════════════════════════════════════════════════════════

_OU_NAN_KEYS = ["theta", "mu", "sigma_eq", "half_life", "z_score"]


def _ou_nan() -> dict:
    return {k: np.nan for k in _OU_NAN_KEYS}


def estimate_ou_params_window(prices: np.ndarray) -> dict:
    """
    Fit AR(1): X_t = a + b·X_{t-1} + ε  via OLS.
    Returns OU parameters for the last price in the window.
    """
    if len(prices) < 3:
        return _ou_nan()

    x_lag  = prices[:-1]
    x_curr = prices[1:]

    # scipy.stats.linregress: y = slope*x + intercept
    b, a, *_ = stats.linregress(x_lag, x_curr)

    # b must be in (0,1) for a mean-reverting stationary process
    if not (0.0 < b < 1.0):
        return _ou_nan()

    residuals = x_curr - (a + b * x_lag)
    dt        = 1.0 / TRADING_DAYS_YEAR

    theta    = -np.log(b) / dt                              # annualised reversion speed
    mu       = a / (1.0 - b)                               # long-run mean
    sigma_eq = np.std(residuals, ddof=1) / np.sqrt(1.0 - b ** 2)  # equilibrium std

    if sigma_eq < 1e-10 or not np.isfinite(sigma_eq):
        return _ou_nan()

    half_life = np.log(2.0) / theta * TRADING_DAYS_YEAR    # in trading days
    z_score   = (prices[-1] - mu) / sigma_eq

    return {
        "theta":    float(theta),
        "mu":       float(mu),
        "sigma_eq": float(sigma_eq),
        "half_life":float(half_life),
        "z_score":  float(z_score),
    }


def rolling_ou_params(close: pd.Series, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling AR(1) OU estimation — no look-ahead bias."""
    prices  = close.values
    n       = len(prices)
    results = [_ou_nan() for _ in range(n)]
    for i in range(window - 1, n):
        results[i] = estimate_ou_params_window(prices[i - window + 1 : i + 1])
    return pd.DataFrame(results, index=close.index)


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 6]  Z-Score Signal Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(ou_df: pd.DataFrame) -> pd.Series:
    """
    Raw entry signals based on z-score bands.
    +1 = oversold (long candidate), -1 = overbought (short candidate), 0 = no signal.
    These are end-of-day signals; execution happens on the next day's open.
    """
    z   = ou_df["z_score"]
    sig = pd.Series(0, index=ou_df.index, dtype=int)
    sig[z < -Z_ENTRY] = 1    # oversold → buy signal
    sig[z >  Z_ENTRY] = -1   # overbought → sell signal
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 7]  Hurst Exponent Calculator  (Rescaled Range / RS Analysis)
# ══════════════════════════════════════════════════════════════════════════════

def _hurst_rs(log_returns: np.ndarray) -> float:
    """Hurst exponent via RS analysis.  H<0.5=mean-rev, H>0.5=trending."""
    if len(log_returns) < max(HURST_LAGS) + 1:
        return np.nan

    valid_lags = []
    rs_vals    = []

    for lag in HURST_LAGS:
        if lag >= len(log_returns):
            continue
        n_chunks = len(log_returns) // lag
        if n_chunks < 2:
            continue
        chunk_rs = []
        for k in range(n_chunks):
            chunk    = log_returns[k * lag : (k + 1) * lag]
            mean_adj = chunk - chunk.mean()
            cumdev   = np.cumsum(mean_adj)
            R        = cumdev.max() - cumdev.min()
            S        = chunk.std(ddof=1)
            if S > 0:
                chunk_rs.append(R / S)
        if chunk_rs:
            rs_vals.append(np.mean(chunk_rs))
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return np.nan

    slope, *_ = stats.linregress(np.log(valid_lags), np.log(rs_vals))
    return float(slope)


def rolling_hurst(close: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    log_ret = np.log(close / close.shift(1)).fillna(0.0).values
    n       = len(log_ret)
    hurst   = np.full(n, np.nan)
    for i in range(window - 1, n):
        hurst[i] = _hurst_rs(log_ret[i - window + 1 : i + 1])
    return pd.Series(hurst, index=close.index, name="hurst")


def classify_regime(h: float) -> str:
    if not np.isfinite(h):
        return "random"
    if h < 0.45:
        return "mean-rev"
    if h > 0.55:
        return "trending"
    return "random"


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 8]  Kelly Sizer
# ══════════════════════════════════════════════════════════════════════════════

def compute_kelly_fraction(recent_pnl_pcts: list) -> float:
    """
    Kelly f* = W - (1-W)/R  where W=win_rate, R=avg_win/avg_loss.
    Capped at KELLY_CAP (half-Kelly) for safety.
    """
    if len(recent_pnl_pcts) < KELLY_MIN_TRADES:
        return KELLY_DEFAULT

    arr    = np.array(recent_pnl_pcts)
    wins   = arr[arr > 0]
    losses = arr[arr <= 0]

    if len(wins) == 0 or len(losses) == 0:
        return KELLY_DEFAULT

    W = len(wins) / len(arr)
    R = wins.mean() / abs(losses.mean())
    f = W - (1.0 - W) / R
    return float(np.clip(f, 0.0, KELLY_CAP))


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 9]  Backtester
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    ticker:          str,
    price_df:        pd.DataFrame,
    ou_df:           pd.DataFrame,
    hurst_s:         pd.Series,
    raw_signals:     pd.Series,
    cfg:             Config,
    initial_equity:  float = 100_000.0,
) -> tuple:
    """
    Simulate trade-by-trade.
    Signals generated from yesterday's close; execution at today's open.
    Returns (daily_df, trades_df).
    """
    z_exit   = Z_EXIT_SOFT if cfg.exit_mode == "soft" else Z_EXIT_HARD
    cost_bps = cfg.total_cost_bps

    n          = len(price_df)
    closes     = price_df["close"].values
    opens      = price_df["open"].values
    z_vals     = ou_df["z_score"].values
    hurst_vals = hurst_s.values
    sigs       = raw_signals.values   # sigs[i] → execute at opens[i+1]

    # Output arrays
    equity_arr    = np.full(n, np.nan)
    equity_arr[0] = initial_equity
    position_arr  = np.zeros(n, dtype=int)
    kelly_arr     = np.full(n, KELLY_DEFAULT)
    entry_px_arr  = np.full(n, np.nan)

    # State
    position_dir     = 0
    shares_held      = 0.0
    entry_price      = 0.0
    entry_idx        = 0
    kelly_frac       = KELLY_DEFAULT
    cur_equity       = initial_equity
    closed_pnl_pcts  = []
    trades           = []

    for i in range(1, n):
        open_i     = opens[i]
        close_i    = closes[i]
        close_prev = closes[i - 1]
        z_curr     = z_vals[i]
        sig        = sigs[i - 1]     # signal from yesterday's close
        day_delta  = 0.0

        # ── CLOSE EXISTING POSITION at today's open ──────────────────────────
        if position_dir != 0:
            if position_dir == 1:
                should_exit = (z_curr >= -z_exit) or (z_curr > Z_ENTRY)
            else:
                should_exit = (z_curr <= z_exit) or (z_curr < -Z_ENTRY)

            if should_exit:
                # MTM: yesterday close → today open
                day_delta += position_dir * shares_held * (open_i - close_prev)
                exit_cost  = shares_held * open_i * cost_bps / 10_000
                day_delta -= exit_cost

                # Trade record  (gross PnL vs entry price, both costs)
                gross    = position_dir * shares_held * (open_i - entry_price)
                ent_cost = shares_held * entry_price * cost_bps / 10_000
                net_pnl  = gross - ent_cost - exit_cost
                pnl_pct  = net_pnl / (shares_held * entry_price) if entry_price > 0 else 0.0

                trades.append({
                    "ticker":          ticker,
                    "entry_date":      price_df.index[entry_idx],
                    "exit_date":       price_df.index[i],
                    "direction":       "long" if position_dir == 1 else "short",
                    "entry_price":     round(entry_price, 4),
                    "exit_price":      round(open_i, 4),
                    "shares":          round(shares_held, 4),
                    "gross_pnl":       round(gross, 2),
                    "costs":           round(ent_cost + exit_cost, 2),
                    "net_pnl":         round(net_pnl, 2),
                    "pnl_pct":         round(pnl_pct, 6),
                    "holding_days":    i - entry_idx,
                    "entry_z":         round(float(z_vals[entry_idx - 1]), 4) if entry_idx > 0 else np.nan,
                    "exit_z":          round(float(z_curr), 4),
                    "regime_at_entry": classify_regime(hurst_vals[entry_idx]),
                })

                closed_pnl_pcts.append(pnl_pct)
                kelly_frac   = compute_kelly_fraction(closed_pnl_pcts[-KELLY_LOOKBACK:])
                position_dir = 0
                shares_held  = 0.0
                entry_price  = 0.0

            else:
                # Mark to market: yesterday close → today close
                day_delta += position_dir * shares_held * (close_i - close_prev)

        # ── OPEN NEW POSITION at today's open ────────────────────────────────
        if position_dir == 0 and sig != 0 and np.isfinite(z_vals[i - 1]):
            alloc = cur_equity * kelly_frac
            if alloc > 1.0 and open_i > 0:
                ent_cost     = alloc * cost_bps / 10_000
                shares_held  = (alloc - ent_cost) / open_i
                day_delta   -= ent_cost
                position_dir = int(sig)
                entry_price  = open_i
                entry_idx    = i
                # MTM: today open → today close (first partial day in position)
                day_delta   += position_dir * shares_held * (close_i - open_i)

        cur_equity      += day_delta
        equity_arr[i]    = cur_equity
        position_arr[i]  = position_dir
        kelly_arr[i]     = kelly_frac
        entry_px_arr[i]  = entry_price if position_dir != 0 else np.nan

    # ── Force-close any open position at final close ─────────────────────────
    if position_dir != 0:
        last_i   = n - 1
        exec_px  = closes[last_i]
        # Exit costs only (already MTM'd to close on last loop iteration)
        exit_cost = shares_held * exec_px * cost_bps / 10_000
        equity_arr[last_i] -= exit_cost

        gross    = position_dir * shares_held * (exec_px - entry_price)
        ent_cost = shares_held * entry_price * cost_bps / 10_000
        net_pnl  = gross - ent_cost - exit_cost
        pnl_pct  = net_pnl / (shares_held * entry_price) if entry_price > 0 else 0.0

        trades.append({
            "ticker":          ticker,
            "entry_date":      price_df.index[entry_idx],
            "exit_date":       price_df.index[last_i],
            "direction":       "long" if position_dir == 1 else "short",
            "entry_price":     round(entry_price, 4),
            "exit_price":      round(exec_px, 4),
            "shares":          round(shares_held, 4),
            "gross_pnl":       round(gross, 2),
            "costs":           round(ent_cost + exit_cost, 2),
            "net_pnl":         round(net_pnl, 2),
            "pnl_pct":         round(pnl_pct, 6),
            "holding_days":    last_i - entry_idx,
            "entry_z":         round(float(z_vals[entry_idx - 1]), 4) if entry_idx > 0 else np.nan,
            "exit_z":          round(float(z_vals[last_i]), 4),
            "regime_at_entry": classify_regime(hurst_vals[entry_idx]),
        })

    # ── Assemble daily DataFrame ──────────────────────────────────────────────
    df = price_df.copy()
    df["theta"]           = ou_df["theta"].values
    df["mu"]              = ou_df["mu"].values
    df["sigma_eq"]        = ou_df["sigma_eq"].values
    df["half_life"]       = ou_df["half_life"].values
    df["z_score"]         = z_vals
    df["hurst"]           = hurst_vals
    df["regime"]          = [classify_regime(h) for h in hurst_vals]
    df["signal"]          = sigs
    df["position"]        = position_arr
    df["entry_price_pos"] = entry_px_arr
    df["kelly_fraction"]  = kelly_arr
    df["strategy_equity"] = equity_arr

    strat_eq = pd.Series(equity_arr, index=df.index)
    df["strategy_return"]   = strat_eq.pct_change().fillna(0.0)
    df["benchmark_equity"]  = initial_equity * closes / closes[0]
    df["benchmark_return"]  = df["benchmark_equity"].pct_change().fillna(0.0)

    rolling_max  = strat_eq.cummax()
    df["drawdown"] = (strat_eq - rolling_max) / rolling_max

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return df, trades_df


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 10]  Performance Calculator
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_sharpe(returns: pd.Series, window: int = 30) -> pd.Series:
    mean = returns.rolling(window).mean()
    std  = returns.rolling(window).std(ddof=1).replace(0, np.nan)
    return (mean / std) * np.sqrt(TRADING_DAYS_YEAR)


def compute_metrics(daily_df: pd.DataFrame, trades_df: pd.DataFrame, ticker: str) -> dict:
    strat_ret = daily_df["strategy_return"]
    bench_ret = daily_df["benchmark_return"]
    strat_eq  = daily_df["strategy_equity"]
    bench_eq  = daily_df["benchmark_equity"]
    drawdown  = daily_df["drawdown"]

    n_days = len(daily_df)
    years  = n_days / TRADING_DAYS_YEAR

    # ── Returns & CAGR ────────────────────────────────────────────────────────
    total_ret        = strat_eq.iloc[-1] / strat_eq.iloc[0] - 1.0
    bench_total_ret  = bench_eq.iloc[-1] / bench_eq.iloc[0] - 1.0
    cagr       = (1 + total_ret)       ** (1 / years) - 1 if years > 0 else 0.0
    bench_cagr = (1 + bench_total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    # ── Volatility ────────────────────────────────────────────────────────────
    ann_vol = float(strat_ret.std(ddof=1)) * np.sqrt(TRADING_DAYS_YEAR)

    # ── Sharpe & Sortino ──────────────────────────────────────────────────────
    std_all  = strat_ret.std(ddof=1)
    sharpe   = (strat_ret.mean() / std_all) * np.sqrt(TRADING_DAYS_YEAR) if std_all > 0 else 0.0
    neg_ret  = strat_ret[strat_ret < 0]
    down_std = neg_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR) if len(neg_ret) > 0 else 1e-10
    sortino  = (strat_ret.mean() * TRADING_DAYS_YEAR) / down_std if down_std > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Max drawdown duration (consecutive days in drawdown)
    in_dd   = (drawdown < 0).values
    dd_dur  = max_dd_dur = cur = 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        max_dd_dur = max(max_dd_dur, cur)

    # ── VaR & CVaR (historical, 95%) ──────────────────────────────────────────
    clean = strat_ret.dropna()
    var_95  = float(np.percentile(clean, 5))
    cvar_95 = float(clean[clean <= var_95].mean()) if (clean <= var_95).any() else var_95

    # ── Alpha & Beta ──────────────────────────────────────────────────────────
    b_clean = bench_ret.fillna(0)
    s_clean = strat_ret.fillna(0)
    if b_clean.std() > 0 and len(s_clean) > 2:
        slope, intercept, *_ = stats.linregress(b_clean, s_clean)
        beta  = float(slope)
        alpha = float(intercept) * TRADING_DAYS_YEAR
    else:
        beta, alpha = 1.0, 0.0

    # ── Trade statistics ──────────────────────────────────────────────────────
    if trades_df is not None and len(trades_df) > 0:
        n_trades      = len(trades_df)
        wins          = trades_df[trades_df["net_pnl"] > 0]
        losses        = trades_df[trades_df["net_pnl"] <= 0]
        win_rate      = len(wins) / n_trades
        avg_win       = float(wins["pnl_pct"].mean())   if len(wins)   > 0 else 0.0
        avg_loss      = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
        pf_denom      = abs(losses["net_pnl"].sum())
        profit_factor = float(wins["net_pnl"].sum() / pf_denom) if pf_denom > 0 else np.inf
        expectancy    = float(trades_df["pnl_pct"].mean())
        avg_holding   = float(trades_df["holding_days"].mean())
    else:
        n_trades = win_rate = 0
        avg_win = avg_loss = profit_factor = expectancy = avg_holding = np.nan

    # ── Regime breakdown ──────────────────────────────────────────────────────
    regime_stats = {}
    for regime in ["mean-rev", "random", "trending"]:
        mask = daily_df["regime"] == regime
        if mask.sum() < 5:
            regime_stats[regime] = None
            continue
        r_ret = strat_ret[mask]
        r_std = r_ret.std(ddof=1)
        regime_stats[regime] = {
            "pct_time":   float(mask.sum() / n_days * 100),
            "ann_return": float(r_ret.mean() * TRADING_DAYS_YEAR),
            "ann_vol":    float(r_std * np.sqrt(TRADING_DAYS_YEAR)) if r_std > 0 else 0.0,
            "sharpe":     float((r_ret.mean() / r_std) * np.sqrt(TRADING_DAYS_YEAR)) if r_std > 0 else 0.0,
        }

    # Attach rolling Sharpe to daily_df for charts
    daily_df["rolling_sharpe"] = _rolling_sharpe(strat_ret)

    return {
        "ticker":           ticker,
        "n_days":           n_days,
        "years":            round(years, 2),
        "total_return":     total_ret,
        "bench_total_ret":  bench_total_ret,
        "cagr":             cagr,
        "bench_cagr":       bench_cagr,
        "ann_vol":          ann_vol,
        "sharpe":           float(sharpe),
        "sortino":          float(sortino),
        "calmar":           float(calmar),
        "max_dd":           max_dd,
        "max_dd_days":      max_dd_dur,
        "var_95":           var_95,
        "cvar_95":          cvar_95,
        "beta":             beta,
        "alpha":            alpha,
        "n_trades":         n_trades,
        "win_rate":         float(win_rate),
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "profit_factor":    profit_factor,
        "expectancy":       expectancy,
        "avg_holding_days": avg_holding,
        "regime_stats":     regime_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 11]  Rich Terminal Reporter
# ══════════════════════════════════════════════════════════════════════════════

def _fp(v, dec=2) -> str:
    """Format percentage."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    col = "green" if v >= 0 else "red"
    return f"[{col}]{v * 100:.{dec}f}%[/{col}]"


def _ff(v, dec=2) -> str:
    """Format float."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    col = "green" if v >= 0 else "red"
    return f"[{col}]{v:.{dec}f}[/{col}]"


def print_report(metrics_list: list, cfg: Config):
    console.print()
    console.print(Panel.fit(
        f"[bold white]OU Mean Reversion Backtest Report[/bold white]\n"
        f"[dim]{TIMEFRAME_MAP[cfg.timeframe][1]}  |  "
        f"{cfg.start_date} → {cfg.end_date}  |  "
        f"Commission {cfg.commission_bps}bps  |  Slippage {cfg.slippage_bps}bps  |  "
        f"Exit: {cfg.exit_mode}[/dim]",
        style="bold blue",
        padding=(0, 2),
    ))

    for m in metrics_list:
        ticker = m["ticker"]
        console.print(f"\n[bold yellow]{'─'*6} {ticker} {'─'*50}[/bold yellow]")

        # ── Performance vs Benchmark ──────────────────────────────────────────
        t1 = Table(title="Performance vs Benchmark", box=box.SIMPLE_HEAVY,
                   show_header=True, header_style="bold cyan", padding=(0, 1))
        t1.add_column("Metric",         style="dim",    min_width=28)
        t1.add_column("Strategy",       justify="right", min_width=14)
        t1.add_column("Buy & Hold",     justify="right", min_width=14)

        perf_rows = [
            ("Total Return",              _fp(m["total_return"]),   _fp(m["bench_total_ret"])),
            ("CAGR",                      _fp(m["cagr"]),           _fp(m["bench_cagr"])),
            ("Ann. Volatility",           _fp(m["ann_vol"]),        "—"),
            ("Sharpe Ratio (ann.)",       _ff(m["sharpe"]),         "—"),
            ("Sortino Ratio",             _ff(m["sortino"]),        "—"),
            ("Calmar Ratio",              _ff(m["calmar"]),         "—"),
            ("Max Drawdown",              _fp(m["max_dd"]),         "—"),
            ("Max DD Duration (days)",    str(m["max_dd_days"]),    "—"),
            ("VaR 95% (1-day)",           _fp(m["var_95"]),         "—"),
            ("CVaR 95% (1-day)",          _fp(m["cvar_95"]),        "—"),
            ("Alpha (ann.)",              _fp(m["alpha"]),          "—"),
            ("Beta",                      _ff(m["beta"]),           "—"),
        ]
        for r in perf_rows:
            t1.add_row(*r)
        console.print(t1)

        # ── Trade Statistics ──────────────────────────────────────────────────
        t2 = Table(title="Trade Statistics", box=box.SIMPLE_HEAVY,
                   show_header=True, header_style="bold cyan", padding=(0, 1))
        t2.add_column("Metric",    style="dim",     min_width=28)
        t2.add_column("Value",     justify="right", min_width=14)

        trade_rows = [
            ("# Trades",               str(m["n_trades"])),
            ("Win Rate",               _fp(m["win_rate"])),
            ("Avg Win",                _fp(m["avg_win_pct"])),
            ("Avg Loss",               _fp(m["avg_loss_pct"])),
            ("Profit Factor",          _ff(m["profit_factor"])),
            ("Expectancy (per trade)", _fp(m["expectancy"])),
            ("Avg Holding (days)",     _ff(m["avg_holding_days"], 1)),
        ]
        for r in trade_rows:
            t2.add_row(*r)
        console.print(t2)

        # ── Regime Breakdown ──────────────────────────────────────────────────
        rs = m.get("regime_stats", {})
        if any(v is not None for v in rs.values()):
            t3 = Table(title="Regime Breakdown (Hurst RS)", box=box.SIMPLE_HEAVY,
                       show_header=True, header_style="bold cyan", padding=(0, 1))
            t3.add_column("Regime",      min_width=14)
            t3.add_column("% Time",      justify="right", min_width=8)
            t3.add_column("Ann. Return", justify="right", min_width=12)
            t3.add_column("Ann. Vol",    justify="right", min_width=10)
            t3.add_column("Sharpe",      justify="right", min_width=10)

            regime_labels = {
                "mean-rev": "[green]Mean-Rev[/green] (H<0.45)",
                "random":   "[yellow]Random[/yellow]  (H≈0.5)",
                "trending": "[red]Trending[/red] (H>0.55)",
            }
            for regime, rd in rs.items():
                if rd is None:
                    continue
                t3.add_row(
                    regime_labels[regime],
                    f"{rd['pct_time']:.1f}%",
                    _fp(rd["ann_return"]),
                    _fp(rd["ann_vol"]),
                    _ff(rd["sharpe"]),
                )
            console.print(t3)
            console.print(
                "  [dim]Regime is identified via rolling 60-day Hurst exponent (RS analysis).\n"
                "  Mean-Rev periods are where this strategy is theoretically best suited.[/dim]"
            )


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 12]  Chart Generator  (6-panel matplotlib figure)
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_charts(
    ticker:    str,
    daily_df:  pd.DataFrame,
    trades_df: pd.DataFrame,
    cfg:       Config,
) -> str:
    """Generate 6-panel backtest chart. Returns base64-encoded PNG string."""
    ticker_dir = cfg.output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    BG  = "#0d1117"
    FG  = "#c9d1d9"
    ACC = "#58a6ff"

    fig = plt.figure(figsize=(16, 22), facecolor=BG)
    fig.suptitle(
        f"{ticker}  ·  OU Mean Reversion  ·  "
        f"{daily_df.index[0].strftime('%Y-%m-%d')} → {daily_df.index[-1].strftime('%Y-%m-%d')}",
        fontsize=13, fontweight="bold", color=FG, y=0.99,
    )

    gs   = GridSpec(6, 1, figure=fig, hspace=0.52,
                    top=0.97, bottom=0.03, left=0.07, right=0.97)
    dates = daily_df.index

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, fontsize=10, color=FG, pad=4, loc="left")
        ax.tick_params(colors=FG, labelsize=8)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, color="#30363d")
        ax.yaxis.label.set_color(FG)

    # ── Panel 1: Price + Signals ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, daily_df["close"], color=ACC, linewidth=1.2, label="Close", zorder=2)

    if trades_df is not None and len(trades_df) > 0:
        long_ent  = trades_df[trades_df["direction"] == "long"]
        short_ent = trades_df[trades_df["direction"] == "short"]

        def price_at(dt_series):
            return daily_df["close"].reindex(dt_series).values

        if len(long_ent):
            ax1.scatter(long_ent["entry_date"], price_at(long_ent["entry_date"].values),
                        marker="^", color="#2ecc71", s=70, zorder=5, label="Long entry")
        if len(short_ent):
            ax1.scatter(short_ent["entry_date"], price_at(short_ent["entry_date"].values),
                        marker="v", color="#e74c3c", s=70, zorder=5, label="Short entry")
        exits = pd.concat([long_ent, short_ent]) if len(long_ent) and len(short_ent) else (long_ent if len(long_ent) else short_ent)
        ax1.scatter(trades_df["exit_date"], price_at(trades_df["exit_date"].values),
                    marker="x", color="#f39c12", s=50, zorder=5, label="Exit", linewidths=1.2)

    ax1.legend(fontsize=7, loc="upper left", framealpha=0.3)
    ax1.set_ylabel("Price ($)", fontsize=8)
    style_ax(ax1, "Price  +  Trade Entries / Exits")

    # ── Panel 2: Z-Score ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, daily_df["z_score"], color="#a29bfe", linewidth=0.9, label="Z-score")
    ax2.fill_between(dates, daily_df["z_score"].clip(lower=-Z_ENTRY), -Z_ENTRY,
                     where=daily_df["z_score"] < -Z_ENTRY, color="#2ecc71", alpha=0.25)
    ax2.fill_between(dates, daily_df["z_score"].clip(upper=Z_ENTRY), Z_ENTRY,
                     where=daily_df["z_score"] > Z_ENTRY, color="#e74c3c", alpha=0.25)
    for level, col, lbl in [
        (Z_ENTRY,      "#e74c3c", f"+{Z_ENTRY}σ short entry"),
        (-Z_ENTRY,     "#2ecc71", f"-{Z_ENTRY}σ long entry"),
        (Z_EXIT_SOFT,  "#f39c12", f"±{Z_EXIT_SOFT}σ exit"),
        (-Z_EXIT_SOFT, "#f39c12", None),
        (0,            "#ffffff", None),
    ]:
        kw = {"label": lbl} if lbl else {}
        ax2.axhline(level, color=col, linestyle="--" if lbl else ":", linewidth=0.8, alpha=0.8, **kw)
    ax2.legend(fontsize=7, loc="upper left", framealpha=0.3)
    ax2.set_ylabel("Z-Score", fontsize=8)
    style_ax(ax2, "Z-Score  (Ornstein-Uhlenbeck, rolling 60-day AR(1))")

    # ── Panel 3: Equity Curve + regime shading ───────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dates, daily_df["strategy_equity"],  color="#f9ca24", linewidth=1.5, label="Strategy", zorder=3)
    ax3.plot(dates, daily_df["benchmark_equity"], color="#636e72", linewidth=1.0, label="Buy & Hold", alpha=0.7)

    # Regime shading spans
    reg_colours = {"mean-rev": "#2ecc71", "random": "#f39c12", "trending": "#e74c3c"}
    prev_reg, seg_start = None, dates[0]
    for dt, reg in zip(dates, daily_df["regime"]):
        if reg != prev_reg:
            if prev_reg is not None:
                ax3.axvspan(seg_start, dt, color=reg_colours.get(prev_reg, "#fff"),
                            alpha=0.12, zorder=1)
            seg_start, prev_reg = dt, reg
    if prev_reg:
        ax3.axvspan(seg_start, dates[-1], color=reg_colours.get(prev_reg, "#fff"), alpha=0.12)

    patches = [
        mpatches.Patch(color="#2ecc71", alpha=0.5, label="Mean-Rev (H<0.45)"),
        mpatches.Patch(color="#f39c12", alpha=0.5, label="Random (H≈0.5)"),
        mpatches.Patch(color="#e74c3c", alpha=0.5, label="Trending (H>0.55)"),
    ]
    h_, l_ = ax3.get_legend_handles_labels()
    ax3.legend(h_ + patches, l_ + [p.get_label() for p in patches],
               fontsize=7, loc="upper left", framealpha=0.3, ncol=2)
    ax3.set_ylabel("Equity ($)", fontsize=8)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    style_ax(ax3, "Equity Curve vs Buy & Hold  (regime-shaded background)")

    # ── Panel 4: Drawdown ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    dd_pct = daily_df["drawdown"] * 100
    ax4.fill_between(dates, dd_pct, 0, color="#e74c3c", alpha=0.6, label="Drawdown")
    ax4.plot(dates, dd_pct, color="#e74c3c", linewidth=0.6)
    ax4.set_ylabel("Drawdown (%)", fontsize=8)
    ax4.legend(fontsize=7, loc="lower left", framealpha=0.3)
    style_ax(ax4, "Rolling Drawdown")

    # ── Panel 5: Hurst Exponent ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    h_s = daily_df["hurst"].fillna(0.5)
    ax5.plot(dates, daily_df["hurst"], color="#fd79a8", linewidth=1.0, label="Hurst H")
    ax5.axhline(0.5,  color=FG,       linestyle="--", linewidth=0.8, alpha=0.6, label="H=0.5 (random walk)")
    ax5.axhline(0.45, color="#2ecc71", linestyle=":",  linewidth=0.7, alpha=0.5)
    ax5.axhline(0.55, color="#e74c3c", linestyle=":",  linewidth=0.7, alpha=0.5)
    ax5.fill_between(dates, h_s, 0.5, where=h_s < 0.5, color="#2ecc71", alpha=0.2, label="Mean-Rev")
    ax5.fill_between(dates, h_s, 0.5, where=h_s > 0.5, color="#e74c3c", alpha=0.2, label="Trending")
    ax5.set_ylim(0.1, 0.9)
    ax5.set_ylabel("Hurst H", fontsize=8)
    ax5.legend(fontsize=7, loc="upper right", framealpha=0.3)
    style_ax(ax5, "Rolling Hurst Exponent  (60-day RS analysis)")

    # ── Panel 6: Rolling Sharpe ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[5])
    rs_s = daily_df.get("rolling_sharpe", pd.Series(dtype=float)).fillna(0)
    ax6.plot(dates, rs_s, color="#00cec9", linewidth=0.9, label="30-day Sharpe")
    ax6.axhline(0, color=FG,       linestyle="-",  linewidth=0.4, alpha=0.3)
    ax6.axhline(1, color="#2ecc71", linestyle="--", linewidth=0.7, alpha=0.5, label="Sharpe=1")
    ax6.fill_between(dates, rs_s, 0, where=rs_s > 0, color="#2ecc71", alpha=0.15)
    ax6.fill_between(dates, rs_s, 0, where=rs_s < 0, color="#e74c3c", alpha=0.15)
    ax6.set_ylabel("Sharpe", fontsize=8)
    ax6.legend(fontsize=7, loc="upper left", framealpha=0.3)
    style_ax(ax6, "Rolling 30-Day Sharpe Ratio")

    out_path = ticker_dir / f"{ticker}_backtest.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=BG)
    console.print(f"  [green]Chart saved:[/green] {out_path}")

    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 13]  HTML Report Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_html_report(all_metrics: list, chart_b64_map: dict, cfg: Config) -> Path:
    today_str = date.today().isoformat()

    def pct(v, dec=2):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        cls = "pos" if v >= 0 else "neg"
        return f'<span class="{cls}">{v * 100:.{dec}f}%</span>'

    def flt(v, dec=2):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        cls = "pos" if v >= 0 else "neg"
        return f'<span class="{cls}">{v:.{dec}f}</span>'

    def mrow(label, strat, bench="—"):
        return f"<tr><td>{label}</td><td>{strat}</td><td>{bench}</td></tr>"

    def trow(label, val):
        return f"<tr><td>{label}</td><td>{val}</td></tr>"

    REGIME_COL = {"mean-rev": "#2ecc71", "random": "#f39c12", "trending": "#e74c3c"}

    ticker_sections = []
    for m in all_metrics:
        t      = m["ticker"]
        b64    = chart_b64_map.get(t, "")
        rs     = m.get("regime_stats", {})

        regime_rows_html = ""
        for reg, rd in rs.items():
            if rd is None:
                continue
            col = REGIME_COL.get(reg, "#fff")
            regime_rows_html += (
                f"<tr>"
                f"<td style='color:{col};font-weight:600'>{reg.title()}</td>"
                f"<td>{rd['pct_time']:.1f}%</td>"
                f"<td>{pct(rd['ann_return'])}</td>"
                f"<td>{pct(rd['ann_vol'])}</td>"
                f"<td>{flt(rd['sharpe'])}</td>"
                f"</tr>"
            )

        ticker_sections.append(f"""
    <section class="card">
      <h2>{t}</h2>
      <img src="data:image/png;base64,{b64}" alt="{t} chart">
      <div class="grid2">
        <div>
          <h3>Performance vs Benchmark</h3>
          <table>
            <thead><tr><th>Metric</th><th>Strategy</th><th>Buy &amp; Hold</th></tr></thead>
            <tbody>
              {mrow("Total Return",             pct(m["total_return"]),   pct(m["bench_total_ret"]))}
              {mrow("CAGR",                     pct(m["cagr"]),           pct(m["bench_cagr"]))}
              {mrow("Ann. Volatility",          pct(m["ann_vol"]))}
              {mrow("Sharpe Ratio",             flt(m["sharpe"]))}
              {mrow("Sortino Ratio",            flt(m["sortino"]))}
              {mrow("Calmar Ratio",             flt(m["calmar"]))}
              {mrow("Max Drawdown",             pct(m["max_dd"]))}
              {mrow("Max DD Duration (days)",   str(m["max_dd_days"]))}
              {mrow("VaR 95% (1-day)",          pct(m["var_95"]))}
              {mrow("CVaR 95% (1-day)",         pct(m["cvar_95"]))}
              {mrow("Alpha (annualised)",        pct(m["alpha"]))}
              {mrow("Beta",                     flt(m["beta"]))}
            </tbody>
          </table>
        </div>
        <div>
          <h3>Trade Statistics</h3>
          <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
              {trow("# Trades",               str(m["n_trades"]))}
              {trow("Win Rate",               pct(m["win_rate"]))}
              {trow("Avg Win",                pct(m["avg_win_pct"]))}
              {trow("Avg Loss",               pct(m["avg_loss_pct"]))}
              {trow("Profit Factor",          flt(m["profit_factor"]))}
              {trow("Expectancy (per trade)", pct(m["expectancy"]))}
              {trow("Avg Holding (days)",     flt(m["avg_holding_days"], 1))}
            </tbody>
          </table>
        </div>
      </div>
      <h3>Regime Breakdown <span class="sub">(Hurst RS Analysis, 60-day rolling)</span></h3>
      <p class="note">
        The rolling Hurst exponent classifies each day's market regime.
        Mean-Rev periods (H&lt;0.45) are where OU mean reversion is theoretically most effective.
        Compare strategy metrics across regimes to identify when the edge is concentrated.
      </p>
      <table>
        <thead><tr><th>Regime</th><th>% Time</th><th>Ann. Return</th><th>Ann. Vol</th><th>Sharpe</th></tr></thead>
        <tbody>{regime_rows_html}</tbody>
      </table>
    </section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OU Mean Reversion — {today_str}</title>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace;line-height:1.65;padding:2rem 2.5rem}}
    h1{{color:#58a6ff;font-size:1.75rem;margin-bottom:.25rem}}
    h2{{color:#f0a500;font-size:1.3rem;margin:1.8rem 0 .8rem;padding-bottom:.35rem;border-bottom:1px solid #30363d}}
    h3{{color:#79c0ff;font-size:.95rem;margin:1.1rem 0 .4rem}}
    .sub{{color:#8b949e;font-size:.82rem;font-weight:400}}
    .meta{{color:#8b949e;font-size:.83rem;margin-bottom:2rem}}
    .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.5rem;margin-bottom:2.5rem}}
    .card img{{max-width:100%;border-radius:6px;margin-bottom:1.2rem}}
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:1rem 0}}
    table{{width:100%;border-collapse:collapse;font-size:.85rem}}
    thead th{{background:#21262d;color:#58a6ff;padding:.45rem .75rem;text-align:left;font-weight:600}}
    tbody tr:nth-child(even){{background:#1c2128}}
    tbody td{{padding:.35rem .75rem;border-bottom:1px solid #21262d}}
    .pos{{color:#3fb950}}.neg{{color:#f85149}}
    .note{{font-size:.8rem;color:#8b949e;font-style:italic;margin-bottom:.7rem}}
    footer{{text-align:center;color:#30363d;font-size:.75rem;margin-top:3rem;padding-top:1rem;border-top:1px solid #21262d}}
    @media(max-width:768px){{.grid2{{grid-template-columns:1fr}}body{{padding:1rem}}}}
  </style>
</head>
<body>
  <h1>OU Mean Reversion Backtest Report</h1>
  <p class="meta">
    Generated: {today_str} &nbsp;|&nbsp;
    Tickers: <strong>{", ".join(cfg.tickers)}</strong> &nbsp;|&nbsp;
    {TIMEFRAME_MAP[cfg.timeframe][1]} ({cfg.start_date} → {cfg.end_date}) &nbsp;|&nbsp;
    Commission: {cfg.commission_bps}bps &nbsp;|&nbsp; Slippage: {cfg.slippage_bps}bps &nbsp;|&nbsp;
    Exit mode: {cfg.exit_mode} &nbsp;|&nbsp; OU window: 60-day rolling AR(1)
  </p>
  {"".join(ticker_sections)}
  <footer>
    Ornstein-Uhlenbeck Mean Reversion · Rolling AR(1) OU Estimation · Kelly Position Sizing · Hurst RS Regime Detection
  </footer>
</body>
</html>"""

    out_path = cfg.output_dir / f"report_{today_str}.html"
    out_path.write_text(html, encoding="utf-8")
    console.print(f"\n  [bold green]HTML report:[/bold green] {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# [SECTION 14]  Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config()

    console.print(Panel.fit(
        f"[bold]Ornstein-Uhlenbeck Mean Reversion Backtester[/bold]\n"
        f"Tickers   : [cyan]{', '.join(cfg.tickers)}[/cyan]\n"
        f"Timeframe : [yellow]{TIMEFRAME_MAP[cfg.timeframe][1]}[/yellow]"
        f"  ({cfg.start_date} → {cfg.end_date})\n"
        f"Costs     : {cfg.commission_bps}bps commission + {cfg.slippage_bps}bps slippage"
        f"  |  Exit: {cfg.exit_mode}  |  OU window: {ROLLING_WINDOW}-day\n"
        f"Output    : {cfg.output_dir.resolve()}",
        style="bold blue",
        padding=(0, 2),
    ))

    all_metrics:   list = []
    chart_b64_map: dict = {}

    for ticker in cfg.tickers:
        console.print(f"\n[bold white]{'━'*60}[/bold white]")
        console.print(f"[bold white]  {ticker}[/bold white]")
        console.print(f"[bold white]{'━'*60}[/bold white]")

        # 1. Fetch OHLCV
        price_df = fetch_ohlcv(ticker, cfg.start_date, cfg.end_date, cfg.api_key)

        # 2. OU parameter estimation
        console.print("  [dim]Estimating OU parameters (rolling 60-day AR1)...[/dim]", end=" ")
        ou_df = rolling_ou_params(price_df["close"])
        valid  = ou_df["z_score"].notna().sum()
        console.print(f"[green]done[/green]  ({valid} valid days)")

        # 3. Hurst exponent
        console.print("  [dim]Computing rolling Hurst exponent (RS analysis)...[/dim]", end=" ")
        hurst_s = rolling_hurst(price_df["close"])
        console.print("[green]done[/green]")

        # 4. Signals
        raw_signals = generate_signals(ou_df)
        n_sig = (raw_signals != 0).sum()
        console.print(f"  [dim]Signal events:[/dim] {n_sig} raw entry signals")

        # 5. Backtest
        console.print("  [dim]Running backtest simulation...[/dim]", end=" ")
        daily_df, trades_df = run_backtest(ticker, price_df, ou_df, hurst_s, raw_signals, cfg)
        n_trades = len(trades_df)
        console.print(f"[green]done[/green]  ({n_trades} trades closed)")

        # 6. Metrics
        metrics = compute_metrics(daily_df, trades_df, ticker)
        all_metrics.append(metrics)

        # 7. Charts
        console.print("  [dim]Generating charts...[/dim]")
        b64 = generate_charts(ticker, daily_df, trades_df, cfg)
        chart_b64_map[ticker] = b64

        # 8. Save trade log CSV
        if len(trades_df) > 0:
            csv_path = cfg.output_dir / ticker / f"{ticker}_trades.csv"
            trades_df.to_csv(csv_path, index=False)
            console.print(f"  [green]Trade log CSV:[/green] {csv_path}")

    # 9. Terminal report
    print_report(all_metrics, cfg)

    # 10. HTML report
    build_html_report(all_metrics, chart_b64_map, cfg)

    console.print(f"\n[bold green]✓  Backtest complete.[/bold green]  "
                  f"Output: [cyan]{cfg.output_dir.resolve()}[/cyan]\n")


if __name__ == "__main__":
    main()
