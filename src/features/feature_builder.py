"""
Feature engineering for NBA game prediction.

TODO: Implement feature transformations including:
  - Rolling averages (e.g., last 10 games offensive rating)
  - Rest days (days since last game)
  - Home/away indicators
  - Season-to-date stats
  - Rest imbalance (difference between home and away team rest)
  - Momentum indicators
"""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_features(df):
    """
    Transform raw game data into features for modeling.

    Args:
        df (pd.DataFrame): Merged data from preprocess.merge_and_clean()

    Returns:
        pd.DataFrame: Dataset with engineered features ready for modeling
    """
    logger.info("Building features...")
    logger.info(f"Input shape: {df.shape}")

    # TODO: Implement feature engineering logic
    # Examples:
    #   - df['home_off_rating_ma'] = rolling_average(df['home_off_rating'], window=10)
    #   - df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']
    #   - One-hot encode for categorical features (teams, season, etc.)

    logger.info(f"Output shape: {df.shape}")
    return df
