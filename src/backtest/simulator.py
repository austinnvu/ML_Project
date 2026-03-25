"""
Backtesting simulator for trading strategy.

Evaluates trading performance against Kalshi contract prices using
a simple threshold-based strategy: trade when model probability diverges
from market price by more than a threshold.

TODO: Implement full simulator with position sizing, kelly criterion, PnL tracking.
"""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_backtest(model, games_df, market_df, threshold=0.05, bankroll=10000):
    """
    Run a simulated trading backtest.

    Strategy:
      - If |model_prob - market_prob| > threshold, take a trade
      - Bet a fixed amount (or kelly-sized) on the direction of divergence
      - Track cumulative PnL

    TODO: Implement logic for:
      - Computing expected value of divergence
      - Position sizing (fixed or kelly)
      - Trade execution and settlement
      - Commission/fees
      - Risk management

    Args:
        model: Trained model with predict_proba() method
        games_df (pd.DataFrame): Game data with dates and results
        market_df (pd.DataFrame): Market prices for games
        threshold (float): Divergence threshold to trigger trade (default 0.05)
        bankroll (float): Initial capital (default $10,000)

    Returns:
        dict: Results {
            'roi': float (return on investment as decimal),
            'n_trades': int (number of trades),
            'win_rate': float (% of profitable trades),
            'pnl_series': pd.Series (cumulative PnL over time),
            'details': pd.DataFrame (trade-level details)
        }
    """
    logger.info(f"Running backtest with threshold={threshold}, bankroll=${bankroll}...")

    # TODO: Implement backtest logic
    # 1. Align games_df with market_df
    # 2. Generate predictions for each game
    # 3. For each game, check if |pred - market_price| > threshold
    # 4. If triggered, record trade direction and outcome
    # 5. Compute PnL and aggregate statistics

    raise NotImplementedError("run_backtest() not yet implemented")
