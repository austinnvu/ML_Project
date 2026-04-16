"""
Run a flat-bet ROI simulation on moneyline logistic-regression predictions.
"""

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest.simulator import simulate_betting_roi
from src.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_PREDICTIONS_PATH = "artifacts/logistic_moneyline/test_predictions.csv"
DEFAULT_OUTPUT_PATH = "artifacts/logistic_moneyline/roi_bets.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ROI simulation on logistic regression predictions")
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=DEFAULT_PREDICTIONS_PATH,
        help="CSV produced by scripts/train.py containing predictions and Kalshi prices",
    )
    parser.add_argument(
        "--bet-amount",
        type=float,
        default=10.0,
        help="Flat dollar amount to wager on each qualifying trade",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Minimum probability edge needed to place a bet",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the trade-level ROI results CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading prediction file from {args.predictions_path}")
    predictions_df = pd.read_csv(args.predictions_path)

    required_columns = {"home_WIN", "model_prob_home_win", "kalshi_price_home_win"}
    missing_columns = required_columns.difference(predictions_df.columns)
    if missing_columns:
        raise ValueError(f"Predictions CSV is missing required columns: {sorted(missing_columns)}")

    results_df, summary = simulate_betting_roi(
        y_test=predictions_df["home_WIN"],
        y_prob_home=predictions_df["model_prob_home_win"],
        kalshi_prob_home=predictions_df["kalshi_price_home_win"],
        metadata_df=predictions_df[[col for col in predictions_df.columns if col not in required_columns]],
        bet_amount=args.bet_amount,
        edge_threshold=args.edge_threshold,
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(args.output_path, index=False)

    print("--- Betting Simulation Results ---")
    print(f"Total Games in Test Set: {summary['total_games']}")
    print(f"Total Bets Placed:       {summary['total_bets']} (Edge > {args.edge_threshold * 100:.1f}%)")
    print(f"Total Amount Wagered:    ${summary['total_wagered']:.2f}")
    print(f"Net Profit/Loss:         ${summary['net_profit']:.2f}")
    print(f"Return on Investment:    {summary['roi_percent']:.2f}%")

    logger.info(f"Saved ROI trade details to {args.output_path}")


if __name__ == "__main__":
    main()
