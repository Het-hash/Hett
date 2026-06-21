# Hett — Phase 1: NIFTY 50 Edge Discovery

A research-grade quantitative trading research system for discovering small,
repeatable statistical edges in the NIFTY 50 index using publicly available data.

---

## Research Objective

Phase 1 answers one question: **does a measurable, economically meaningful
trading edge exist in daily NIFTY 50 data that survives realistic costs and
out-of-sample validation?**

RL is intentionally excluded. RL is only considered after a robust edge is
confirmed through strict walk-forward and out-of-sample validation.

---

## ⚠ Critical Limitations

- **^NSEI cannot be traded directly.** This system uses it as a research proxy only.
- Backtest costs are **indicative scenario assumptions**, not real execution costs.
- Reported backtested performance does **not** represent achievable future returns.
- Any candidate must pass the **locked test period** before Phase 2 is considered.

---

## Architecture

```
hett/
├── config/
│   ├── base.yaml          # Data, splits, risk, regime config
│   ├── costs.yaml         # Cost profiles (zero / ETF / futures)
│   └── experiments.yaml   # Hypothesis library
│
├── src/
│   ├── data/              # Download → clean → validate → cache pipeline
│   ├── features/          # Causal feature families (returns, trend, MR, vol, breakouts, gaps, calendar)
│   ├── regimes/           # Bull/Bear/Neutral + Low/Normal/High vol + structure classifiers
│   ├── edge_discovery/    # Forward returns, excursions, stats, FDR, ranking
│   ├── backtesting/       # Engine, cost model, baselines, metrics
│   ├── validation/        # Splits, walk-forward, purging, sensitivity, stress tests
│   ├── risk/              # Position sizing, drawdown, limits
│   ├── reports/           # Charts, tables, HTML report builder
│   ├── experiments/       # Registry + runner
│   └── utils/             # Logging, paths, random state, serialization
│
├── scripts/
│   ├── seed_synthetic_data.py   # Generate offline test data (NOT real data)
│   ├── download_data.py         # Download from Yahoo Finance
│   ├── validate_data.py         # Data quality report
│   ├── run_edge_discovery.py    # Feature + regime + hypothesis evaluation
│   ├── run_backtests.py         # Baseline strategies
│   ├── generate_report.py       # HTML report
│   └── run_phase1.py            # ★ Master pipeline (all steps)
│
└── tests/                 # 53 automated tests
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+.

---

## Data Sources

**Default:** Yahoo Finance via `yfinance` (^NSEI symbol).

The index volume field for ^NSEI is unreliable. Volume-based signals are
disabled by default (`nsei_volume_reliable: false` in `config/base.yaml`).

**Offline / CI testing:**
```bash
python scripts/seed_synthetic_data.py
```
This generates a GBM synthetic price series calibrated to approximate
NIFTY 50 parameters. Results on synthetic data are for pipeline validation
only — not investment research.

---

## Configuration

All parameters are in `config/base.yaml`:

```yaml
data:
  symbol: "^NSEI"
  start_date: null       # null = maximum available history
  end_date: null

splits:
  development: 0.60
  validation:  0.20
  test:        0.20      # LOCKED — never used for selection

capital:
  initial: 1000000       # INR 10,00,000
  max_leverage: 1.0
```

Cost profiles are in `config/costs.yaml`. Hypothesis configs are in
`config/experiments.yaml`.

---

## Commands

```bash
# Download real data (requires Yahoo Finance access)
python scripts/download_data.py --config config/base.yaml

# OR: seed synthetic data for offline use
python scripts/seed_synthetic_data.py

# Validate data quality
python scripts/validate_data.py

# Run baseline backtests
python scripts/run_backtests.py

# Run edge discovery only
python scripts/run_edge_discovery.py

# Generate report from existing results
python scripts/generate_report.py

# ★ Full pipeline (recommended)
python scripts/run_phase1.py --config config/base.yaml

# Skip walk-forward for faster runs
python scripts/run_phase1.py --skip-wf

# Force data re-download
python scripts/run_phase1.py --force-refresh

# Run tests
python -m pytest tests/ -v
```

---

## Execution Convention

**Strict no-look-ahead rule:**

| Step | Time |
|------|------|
| Features calculated | Using close of day T and all prior data |
| Signal generated | At close of day T |
| Earliest execution | T+1 open |
| Return measurement | T+1 open → exit price |

A signal at T can **never** use the open, high, low, or close of T+1 or later.

---

## Validation Methodology

1. **Development (60%)** — regime fitting, feature development, hypothesis testing
2. **Validation (20%)** — out-of-sample evaluation of candidates from dev period
3. **Test (20%)** — **LOCKED** — never accessed during selection
4. **Walk-forward** — expanding window OOS evaluation within development period
5. **Stress tests** — cost sensitivity, delay sensitivity, crisis exclusion, bootstrap CI

An edge is only considered for Phase 2 if it passes ALL of:
- Positive OOS expected value (net of costs) on validation period
- Minimum 30 observations
- Hit rate ≥ 45%
- Survives walk-forward
- Survives 25bps cost stress
- Not concentrated in one crisis or rally period

---

## Cost Model

Three profiles available:

| Profile | Approx. Round-Trip |
|---------|--------------------|
| zero | 0 bps |
| etf_conservative | ~10–15 bps |
| etf_aggressive | ~20–30 bps |

Sensitivity sweeps are run at 0, 5, 10, 15, 25, 50 bps.

A candidate that disappears at 25 bps is not considered robust.

---

## Outputs

After running `run_phase1.py`:

```
outputs/
├── figures/           PNG charts (equity curves, drawdown, monthly heatmaps, etc.)
├── tables/            CSV metrics tables
├── experiments/       JSON experiment registry (every run logged)
└── reports/
    └── phase1_report.html   Complete Phase 1 research report
```

---

## Why RL is Excluded from Phase 1

RL is a tool for optimising decisions under uncertainty — not for discovering
whether information content exists. Using RL before establishing an underlying
statistical edge risks:

- Fitting to noise
- Obscuring the source of any apparent performance
- Making validation much harder

If Phase 1 confirms a robust edge, Phase 2 may use RL to improve:
- Signal selectivity
- Position sizing
- Exit timing
- Regime-conditional participation

A simple rule-based strategy remains preferred unless RL produces reproducible
improvement on genuinely untouched data.

---

## Known Limitations

1. **Index proxy** — ^NSEI is not directly tradable
2. **Synthetic test data** — pipeline CI runs on GBM data, not real NIFTY history
3. **Tax not modelled** — STCG/LTCG, securities transaction tax are approximated
4. **Gap risk** — overnight gaps may exceed modelled slippage
5. **Regime stationarity** — future regimes may not match historical distributions
6. **Multiple testing** — despite FDR correction, some findings may be spurious
7. **NIFTY 50 reconstitution** — composition changes affect long-term index level
8. **No intraday data** — daily resolution limits some strategies

---

## Test Suite

```
53 tests covering:
- Data schema and validation
- No look-ahead leakage (including deliberate leakage detection)
- Execution timing (signal at T → position at T+1)
- Cost calculations
- Forward return alignment
- Regime threshold fitting isolation
- Chronological split integrity
- Reproducibility
- Performance metrics
```

```bash
python -m pytest tests/ -v
```
