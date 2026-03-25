"""
Fetcher for NBA game and team statistics via nba_api.

TODO: Implement data fetching from NBA.com via nba_api library.
This module should handle:
  - Fetching games (schedule, results, final scores)
  - Fetching team season statistics (offensive/defensive ratings, pace, etc.)
  - Rolling averages and aggregations
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_games(season, date_from=None, date_to=None):
    """
    Fetch NBA games for a given season.

    Args:
        season (int): NBA season year (e.g., 2024 for 2024-2025 season)
        date_from (str, optional): Start date in YYYY-MM-DD format
        date_to (str, optional): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: Games with columns [game_id, date, home_team, away_team, home_score, away_score, ...]
    """
    # TODO: Implement using nba_api.StaticData.teams() and appropriate endpoints
    logger.info(f"Fetching games for season {season}...")
    raise NotImplementedError("fetch_games() not yet implemented")


def fetch_team_stats(season):
    """
    Fetch season-long team statistics.

    Args:
        season (int): NBA season year

    Returns:
        pd.DataFrame: Team stats with columns [team_id, team_name, off_rating, def_rating, pace, ...]
    """
    # TODO: Implement using nba_api team stats endpoints
    logger.info(f"Fetching team stats for season {season}...")
    raise NotImplementedError("fetch_team_stats() not yet implemented")
