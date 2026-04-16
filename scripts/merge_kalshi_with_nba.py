"""
Merge Kalshi game-price CSV data into the NBA training data CSV.

Rows without a matched Kalshi home-win price are dropped from the final output.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Kalshi prices into NBA training data")
    parser.add_argument(
        "--nba-csv",
        type=str,
        default="data_warehouse/nba_training_data.csv",
        help="Path to the NBA training data CSV",
    )
    parser.add_argument(
        "--kalshi-csv",
        type=str,
        default="data_warehouse/kalshi_game_prices.csv",
        help="Path to the Kalshi game-price CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_warehouse/nba_training_data_with_kalshi.csv",
        help="Path to the merged output CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading NBA CSV from {args.nba_csv}")
    nba_df = pd.read_csv(args.nba_csv)
    logger.info(f"Loading Kalshi CSV from {args.kalshi_csv}")
    kalshi_df = pd.read_csv(args.kalshi_csv)

    join_keys = ["GAME_DATE", "home_TEAM_ABBREVIATION", "away_TEAM_ABBREVIATION"]
    for frame_name, frame in (("NBA", nba_df), ("Kalshi", kalshi_df)):
        missing = set(join_keys).difference(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} CSV is missing required columns: {sorted(missing)}")

    merged_df = nba_df.merge(kalshi_df, on=join_keys, how="left")
    merged_df = merged_df.dropna(subset=["kalshi_price_home_win"]).reset_index(drop=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    merged_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(merged_df)} merged rows with Kalshi prices to {args.output}")


if __name__ == "__main__":
    main()
