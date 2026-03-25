"""
Data preprocessing and integration.

Combines game data, team stats, and contract prices into a single dataset.
"""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def merge_and_clean(games_df, stats_df, contracts_df):
    """
    Merge NBA games, team stats, and contract prices into a unified dataset.

    TODO: Implement logic for:
      - Joining games with home/away team stats
      - Aligning contract prices with game dates
      - Handling missing values
      - Filtering to games with both stats and contract data

    Args:
        games_df (pd.DataFrame): Games from fetch_games()
        stats_df (pd.DataFrame): Team stats from fetch_team_stats()
        contracts_df (pd.DataFrame): Contract prices from fetch_contracts()

    Returns:
        pd.DataFrame: Merged dataset ready for feature engineering
    """
    logger.info("Merging games, stats, and contracts...")
    raise NotImplementedError("merge_and_clean() not yet implemented")


def save_processed(df, path):
    """
    Save processed dataset to disk.

    Args:
        df (pd.DataFrame): Processed data
        path (str): Output file path (CSV)
    """
    logger.info(f"Saving processed data to {path}...")
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")
