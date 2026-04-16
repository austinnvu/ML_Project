"""
Data fetching script.

Fetches NBA games, team stats, and Kalshi contract prices.
"""

import argparse
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.data.nba_fetcher import fetch_games, fetch_team_stats
from src.data.kalshi_fetcher import fetch_contracts
from src.data.preprocess import merge_and_clean, save_processed

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch NBA and market data")
    parser.add_argument("--season", type=int, default=2024, help="NBA season (e.g., 2024)")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()

    logger.info(f"Starting data fetch for season {args.season}...")

    try:
        # TODO: Fetch games and team stats from nba_api
        logger.info("Fetching NBA games...")
        games_df = fetch_games(args.season)

        logger.info("Fetching team stats...")
        stats_df = fetch_team_stats(args.season)

        # TODO: Fetch contract prices from Kalshi
        # For now, this will fail until kalshi_fetcher is implemented
        # logger.info("Fetching Kalshi contracts...")
        # contracts_df = fetch_contracts(...)

        # logger.info("Merging data...")
        # merged_df = merge_and_clean(games_df, stats_df, contracts_df)

        # logger.info(f"Saving to {args.output_dir}...")
        # save_processed(merged_df, f"{args.output_dir}/games_with_prices.csv")

    except NotImplementedError as e:
        logger.warning(f"Data fetch not yet fully implemented: {e}")


if __name__ == "__main__":
    main()
