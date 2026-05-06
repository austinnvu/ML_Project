# NBA Game Outcome Prediction & Contract Mispricing Detection

Predicting NBA game outcomes and identifying mispriced contracts on Kalshi using machine learning.

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

# Edit .env and add your Kalshi API keys
```

### 3. Fetch Data

```bash
make fetch          # NBA games + team stats via nba_api
make fetch-kalshi   # Kalshi closing prices for each game
make merge-kalshi   # Join Kalshi prices into the NBA training CSV
```

### 4. Train Models

```bash
make train          # Trains logistic regression, random forest, and XGBoost
```

### 5. Run Backtest

```bash
make backtest       # XGBoost vs. naive baselines, with threshold sweep
```

## Project Structure

```
├── config/                          # YAML configuration
│   └── config.yaml
├── data_warehouse/                  # Final merged datasets
│   ├── nba_training_data.csv                # 3,748 games with engineered features
│   ├── kalshi_game_prices.csv               # Kalshi closing prices per game
│   └── nba_training_data_with_kalshi.csv    # 882 games with both NBA + Kalshi data
├── artifacts/                       # Per-model outputs (metrics, plots, predictions)
│   ├── logistic_moneyline/
│   ├── random_forest/
│   ├── xgboost/
│   └── xgboost_tuning/                      # Hyperparameter sweep results
├── models/                          # Trained model pickles
│   ├── logistic_regression_moneyline.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── src/                             # Library code
│   ├── data/                        # Fetchers + preprocessing
│   ├── features/                    # Rolling-window feature builder
│   ├── models/                      # Model wrappers
│   ├── evaluation/                  # Metrics and calibration
│   ├── backtest/                    # Trading simulator
│   └── utils/                       # Logging helpers
├── scripts/                         # CLI entry points
│   ├── fetch_data.py
│   ├── fetch_kalshi_data.py
│   ├── merge_kalshi_with_nba.py
│   ├── train.py
│   ├── tune_xgboost.py
│   └── backtest.py
├── tests/                           # pytest suite
├── final_report.tex                 # Final write-up
├── PROPOSAL.md
└── Makefile
```

## Data

- **NBA training set:** 3,748 regular-season games pulled via `nba_api`, with rolling 5- and 10-game team features (offensive/defensive rating, pace, win rate), home/away split win percentages, and rest-days.
- **Kalshi-merged set:** 882 games where Kalshi closing prices were available. This is the dataset used for backtesting because it requires a market price per game.

## Models

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `C=1.0`, L2 penalty |
| Random Forest | `n_estimators=100`, `max_depth=15`, `min_samples_split=10`, `min_samples_leaf=4` |
| XGBoost | `max_depth=6`, `learning_rate=0.05`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=45` (selected via 100-trial random search on a held-out validation split) |

All models output calibrated win probabilities for the home team.

## Results

### Classification metrics (held-out test set)

| Model | Accuracy | ROC-AUC | Log-Loss | Brier |
|---|---|---|---|---|
| Logistic Regression | **0.723** | 0.760 | 0.580 | 0.197 |
| Random Forest | 0.706 | **0.776** | **0.574** | **0.194** |
| XGBoost | 0.678 | 0.737 | 0.602 | 0.207 |

Random Forest gives the best probability quality (lowest Brier, highest AUC); logistic regression edges it on raw accuracy. XGBoost trails on classification metrics — likely a sample-size effect — even though it is the configurable model used for the headline backtest.

### Top predictive features

The most informative signals across models are rolling 5-game offensive/defensive ratings, home/away split win percentage, and rolling 10-game win rate. See [artifacts/random_forest/feature_importances_top15.png](artifacts/random_forest/feature_importances_top15.png) and [artifacts/xgboost/feature_importances_top15.png](artifacts/xgboost/feature_importances_top15.png).

### Backtest (Kalshi-merged set, 177 test games, $10/bet, edge-threshold strategy)

| Strategy | Bets | Net P&L | ROI |
|---|---|---|---|
| Always bet home (naive) | 177 | $-189.51 | -10.71% |
| **Bet the Kalshi favorite (naive)** | **177** | **+$95.60** | **+5.40%** |
| Logistic Regression (edge > 0.05) | 148 | $-497.80 | -33.64% |
| Random Forest (edge > 0.05) | 147 | $-489.81 | -33.32% |
| XGBoost (edge > 0.05) | 146 | $-698.39 | -47.84% |

**Headline finding:** none of the trained models beat the market. The only profitable strategy in our backtest is the trivial "bet the Kalshi favorite" baseline, which suggests Kalshi's prices already incorporate most of the signal our features capture. Increasing the edge threshold (sweep at [artifacts/xgboost/roi_threshold_sweep.csv](artifacts/xgboost/roi_threshold_sweep.csv)) does not recover positive ROI — the bets that look most "mispriced" to the model are systematically the ones where the model is wrong.

See [final_report.tex](final_report.tex) for the full discussion.

## Evaluation Metrics

- **Accuracy** — Classification accuracy at 0.5 threshold
- **ROC-AUC** — Ranking quality across thresholds
- **Log-Loss** — Cross-entropy penalty for confidence
- **Brier Score** — Probability calibration

## Trading Strategy

The backtest evaluates an edge-threshold strategy:

- **Entry:** When `|model_probability − kalshi_price| > threshold`
- **Side:** YES if model > market, NO otherwise
- **Exit:** At game resolution
- **Position Sizing:** Fixed $10 per bet (configurable in `config/config.yaml`)
- **Output:** ROI, win rate, P&L curve, threshold sweep, baseline comparison

## Development

### Run Tests

```bash
make test
```

Tests cover the data fetchers, feature builder, model wrappers, and backtest simulator.

### View Configuration

Edit `config/config.yaml` to adjust:
- Per-model hyperparameters
- Data date ranges and rolling windows
- Backtest parameters (threshold, bankroll, bet sizing)
- API endpoints

### Re-run XGBoost Hyperparameter Sweep

```bash
python scripts/tune_xgboost.py
```

Results are written to `artifacts/xgboost_tuning/sweep_results.json`.

## References

- **nba_api:** https://github.com/swar/nba_api
- **Kalshi API:** https://docs.kalshi.com/
