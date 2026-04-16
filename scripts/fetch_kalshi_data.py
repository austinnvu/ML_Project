"""
Fetch Kalshi NBA moneyline prices for the games present in the NBA training CSV.

This script builds a game-level Kalshi CSV that includes both historical and
current/recent markets by routing requests across Kalshi's historical and live
market endpoints.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.kalshi_fetcher import (
    KalshiClient,
    build_kalshi_game_prices,
    collect_nba_candidate_markets,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Kalshi NBA market prices")
    parser.add_argument(
        "--nba-csv",
        type=str,
        default="data_warehouse/nba_training_data.csv",
        help="Path to the NBA training data CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_warehouse/kalshi_game_prices.csv",
        help="Output path for the Kalshi game-price CSV",
    )
    parser.add_argument(
        "--series-tickers",
        type=str,
        default="",
        help="Optional comma-separated Kalshi series tickers to use instead of auto-discovery",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading NBA games from {args.nba_csv}")
    nba_df = pd.read_csv(args.nba_csv)

    required_columns = {"GAME_DATE", "home_TEAM_ABBREVIATION", "away_TEAM_ABBREVIATION"}
    missing = required_columns.difference(nba_df.columns)
    if missing:
        raise ValueError(f"NBA CSV is missing required columns: {sorted(missing)}")

    explicit_series = [item.strip() for item in args.series_tickers.split(",") if item.strip()]
    client = KalshiClient()

    logger.info("Fetching Kalshi candidate markets across historical and live endpoints")
    candidate_markets = collect_nba_candidate_markets(
        nba_df=nba_df,
        client=client,
        explicit_series_tickers=explicit_series or None,
    )

    logger.info("Matching Kalshi markets back to NBA games")
    kalshi_prices_df = build_kalshi_game_prices(nba_df=nba_df, candidate_markets=candidate_markets)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kalshi_prices_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(kalshi_prices_df)} Kalshi game-price rows to {args.output}")


if __name__ == "__main__":
    main()
