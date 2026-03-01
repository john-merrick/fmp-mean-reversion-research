# FMP Mean Reversion Research

A CLI backtesting tool for an **Ornstein-Uhlenbeck (OU) mean reversion** trading strategy. Feed it any FMP-listed ticker and a timeframe, and it produces a full performance report — terminal tables, charts, and a self-contained HTML report.

---

## Strategy Overview

The Ornstein-Uhlenbeck process models prices as tending to revert to a long-run mean. This script fits OU parameters to each asset using a **rolling 60-day AR(1) regression**, then trades when the price deviates significantly from its estimated mean.

| Signal | Condition | Action |
|---|---|---|
| Long entry | Z-score < −2σ | Price is oversold vs OU mean |
| Short entry | Z-score > +2σ | Price is overbought vs OU mean |
| Soft exit | \|Z-score\| < 1σ | Price has reverted sufficiently |
| Hard exit | \|Z-score\| < 0σ | Price has fully crossed the mean |

**Position sizing** uses a rolling Kelly criterion (capped at 50%) calibrated from the last 30 closed trades, falling back to 10% allocation until enough trade history accumulates.

**Regime detection** runs a rolling Hurst exponent (RS analysis) to classify each period:
- H < 0.45 → Mean-reverting (strategy-favourable)
- H ≈ 0.50 → Random walk
- H > 0.55 → Trending (strategy-unfavourable)

---

## Output

For each ticker the script produces:

- **Terminal report** — rich-formatted performance vs buy & hold, trade statistics, and a regime breakdown showing where the edge is concentrated
- **6-panel chart** — price + signals, z-score bands, equity curve with regime shading, drawdown, rolling Hurst, rolling Sharpe
- **HTML report** — self-contained file with embedded charts and all metrics tables
- **Trade log CSV** — every closed trade with entry/exit prices, P&L, holding days, and regime at entry

All files are saved to `./output/`.

---

## Requirements

- Python 3.10+
- [Financial Modeling Prep](https://financialmodelingprep.com) API key (free tier works for most tickers)

---

## Installation

```bash
git clone https://github.com/john-merrick/fmp-mean-reversion-research.git
cd fmp-mean-reversion-research

pip install requests pandas numpy scipy matplotlib rich python-dotenv
```

---

## Configuration

Copy the example env file and add your FMP API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
FMP_API_KEY=your_api_key_here
```

Your key is never committed — `.env` is listed in `.gitignore`.

---

## Usage

```bash
python ou-mean-reversion-bt.py <TICKER> [TICKER ...] <TIMEFRAME> [OPTIONS]
```

### Timeframes

| Argument | Period |
|---|---|
| `3mo` | 3 months |
| `6mo` | 6 months |
| `1yr` | 1 year |
| `2yr` | 2 years |
| `3yr` | 3 years |
| `5yr` | 5 years |
| `ytd` | Year to date |

### Options

| Flag | Default | Description |
|---|---|---|
| `--commission` | `5` | Commission in basis points |
| `--slippage` | `1` | Slippage in basis points |
| `--exit-mode` | `soft` | `soft` exits at \|z\|=1, `hard` exits at \|z\|=0 |
| `--api-key` | `.env` | Override the API key at runtime |
| `--output-dir` | `./output` | Directory for charts and reports |

### Examples

```bash
# Single ticker, 1 year
python ou-mean-reversion-bt.py QQQ 1yr

# Multiple tickers, 2 years, custom costs
python ou-mean-reversion-bt.py QQQ IWM 2yr --commission 5 --slippage 1

# Aggressive exit, 6 months
python ou-mean-reversion-bt.py SPY 6mo --exit-mode hard

# 5-year backtest with custom output directory
python ou-mean-reversion-bt.py AAPL 5yr --output-dir ./results
```

---

## Performance Metrics

| Category | Metrics |
|---|---|
| Returns | Total return, CAGR |
| Risk-adjusted | Sharpe ratio, Sortino ratio, Calmar ratio |
| Drawdown | Max drawdown, max drawdown duration |
| Tail risk | VaR 95%, CVaR 95% (1-day historical) |
| Benchmark | Alpha, Beta vs buy & hold |
| Trade stats | Win rate, avg win/loss, profit factor, expectancy, avg holding days |
| Regime | All metrics split by mean-reverting / random / trending period |

---

## Project Structure

```
fmp-mean-reversion-research/
├── ou-mean-reversion-bt.py   # Main backtester script
├── .env                      # API key (not committed)
├── .env.example              # Key template
├── .gitignore
├── skills.md
└── output/                   # Generated reports (not committed)
    └── {TICKER}/
        ├── {TICKER}_backtest.png
        └── {TICKER}_trades.csv
    └── report_{date}.html
```
