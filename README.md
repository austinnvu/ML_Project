# NBA Game Outcome Prediction & Contract Mispricing Detection

Predicting NBA game outcomes and identifying mispriced contracts on Kalshi and Polymarket using machine learning.

**Team:** Noah Arooji, Austin Vu, Azim Abdulmajeeth, Ciaran Jones

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
make install
```

### 2. Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Kalshi and Polymarket API keys
```

### 3. Fetch Data

```bash
make fetch
```

### 4. Train Models

```bash
make train
```

### 5. Run Backtest

```bash
make backtest
```

## Project Structure

```
├── config/                 # YAML configuration files
├── data/                   # Raw and processed datasets
│   ├── raw/               # Direct output from APIs
│   └── processed/         # Cleaned, feature-engineered data
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_backtesting.ipynb
├── src/                   # Main Python source code
│   ├── data/             # Data fetching and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Model implementations
│   ├── evaluation/       # Metrics and evaluation
│   ├── backtest/         # Trading simulation
│   └── utils/            # Logging and utilities
├── scripts/              # CLI entry points
│   ├── fetch_data.py
│   ├── train.py
│   └── backtest.py
├── tests/                # Unit tests with pytest
└── Makefile              # Common commands
```

## Models

1. **Logistic Regression** — Fast baseline with interpretable coefficients
2. **Random Forest** — Robust ensemble with feature importance
3. **XGBoost** — Advanced gradient boosting (expected best performer)

All models predict class probabilities for probability calibration.

## Evaluation Metrics

- **Accuracy** — Classification accuracy at 0.5 threshold
- **Log-Loss** — Cross-entropy penalty for confidence
- **Brier Score** — Measures probability calibration

## Trading Strategy

The backtest evaluates a simple threshold-based strategy:

- **Entry:** When `|model_probability - market_price| > threshold`
- **Exit:** At game resolution
- **Position Sizing:** Fixed or Kelly Criterion (configurable)
- **Output:** ROI, win rate, trade statistics

## Development

### Explore Data in Jupyter

```bash
make notebook
```

This launches Jupyter Lab. Start with `notebooks/01_eda.ipynb` to explore the data.

### Run Tests

```bash
make test
```

All tests are in the `tests/` directory using pytest.

### View Configuration

Edit `config/config.yaml` to adjust:
- Hyperparameters for each model
- Data date ranges and rolling windows
- Backtest parameters (threshold, bankroll, bet sizing)
- API endpoints

## References

- **Proposal:** See `PROPOSAL.md` for full project specification
- **nba_api:** https://github.com/swar/nba_api
- **Kalshi API:** https://docs.kalshi.com/

## TODO

- [ ] Implement `src/data/nba_fetcher.py` to fetch games and team stats
- [ ] Implement `src/data/kalshi_fetcher.py` to fetch contract prices
- [ ] Implement `src/data/preprocess.py` to merge data sources
- [ ] Implement feature engineering in `src/features/feature_builder.py`
- [ ] Implement backtest simulator in `src/backtest/simulator.py`
- [ ] Create exploratory notebooks for EDA
- [ ] Add more comprehensive tests
- [ ] Add CI/CD pipeline