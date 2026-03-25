import pandas as pd
import time
from nba_api.stats.endpoints import teamgamelogs

# 1. Define the seasons we want
SEASONS = ['2023-24', '2024-25', '2025-26']

def fetch_nba_data(seasons):
    """Fetches and merges Base and Advanced team game logs."""
    all_games = []

    for season in seasons:
        print(f"Fetching data for {season}...")

        # Fetch Base stats (gets Matchup, W/L, PTS)
        base_log = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            measure_type_player_game_logs_nullable='Base'
        ).get_data_frames()[0]

        time.sleep(1) # Be polite to the NBA API

        # Fetch Advanced stats (gets OffRtg, DefRtg, NetRtg, Pace)
        adv_log = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            measure_type_player_game_logs_nullable='Advanced'
        ).get_data_frames()[0]

        time.sleep(1)

        # Merge on Game ID and Team ID
        cols_to_use = adv_log.columns.difference(base_log.columns).tolist() + ['GAME_ID', 'TEAM_ID']
        merged_season = pd.merge(base_log, adv_log[cols_to_use], on=['GAME_ID', 'TEAM_ID'])

        all_games.append(merged_season)

    df = pd.concat(all_games, ignore_index=True)

    # Convert dates to datetime objects
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.tz_localize(None)
    return df

def engineer_team_features(df):
    """Calculates rest days, rolling averages, and split win percentages."""
    # Sort chronologically for accurate rolling calculations
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    # Convert W/L to binary
    df['WIN'] = (df['WL'] == 'W').astype(int)

    # Home/Away Indicator (NBA matchup format: "BOS vs. LAL" is Home, "LAL @ BOS" is Away)
    df['IS_HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)

    # Group by team
    groupby_team = df.groupby('TEAM_ID')

    # 1. Calculate Rest Days
    df['REST_DAYS'] = groupby_team['GAME_DATE'].diff().dt.days - 1
    # Cap rest days at 7 to prevent massive numbers after offseasons/all-star breaks
    df['REST_DAYS'] = df['REST_DAYS'].fillna(7).clip(upper=7)

    # 2. Calculate Rolling Stats (Crucial: shift(1) to prevent data leakage from the current game)
    rolling_cols = ['WIN', 'OFF_RATING', 'DEF_RATING', 'PACE']
    for col in rolling_cols:
        # 5-game rolling
        df[f'ROLLING_5_{col}'] = groupby_team[col].transform(lambda x: x.shift(1).rolling(5).mean())
        # 10-game rolling
        df[f'ROLLING_10_{col}'] = groupby_team[col].transform(lambda x: x.shift(1).rolling(10).mean())

    # 3. Calculate Season-to-Date Home/Away Split Win %
    # Group by Team, Season, AND whether they are Home/Away
    split_group = df.groupby(['TEAM_ID', 'SEASON_YEAR', 'IS_HOME'])

    # Cumulative games played in this split prior to today
    df['SPLIT_GAMES'] = split_group.cumcount()
    # Cumulative wins in this split prior to today
    df['SPLIT_WINS'] = split_group['WIN'].transform(lambda x: x.shift(1).cumsum().fillna(0))

    df['SPLIT_WIN_PCT'] = df['SPLIT_WINS'] / df['SPLIT_GAMES']
    df['SPLIT_WIN_PCT'] = df['SPLIT_WIN_PCT'].fillna(0) # Handle 0 division on the first game of the split

    return df

def build_matchup_dataset(df):
    """Pivots the team-level data into a Game-level format (Home vs Away)."""
    # Separate home and away DataFrames
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()

    # Add prefixes to differentiate columns
    home_df = home_df.add_prefix('home_')
    away_df = away_df.add_prefix('away_')

    # Rename keys for merging
    home_df = home_df.rename(columns={'home_GAME_ID': 'GAME_ID', 'home_GAME_DATE': 'GAME_DATE'})
    away_df = away_df.rename(columns={'away_GAME_ID': 'GAME_ID'})

    # Merge into a single row per game
    matchups = pd.merge(home_df, away_df, on='GAME_ID')

    # Select our final feature columns
    final_cols = [
        'GAME_DATE',
        'home_TEAM_ABBREVIATION', 'away_TEAM_ABBREVIATION',
        'home_REST_DAYS', 'away_REST_DAYS',
        'home_SPLIT_WIN_PCT', 'away_SPLIT_WIN_PCT',
        'home_ROLLING_5_WIN', 'home_ROLLING_10_WIN',
        'away_ROLLING_5_WIN', 'away_ROLLING_10_WIN',
        'home_ROLLING_5_OFF_RATING', 'home_ROLLING_5_DEF_RATING', 'home_ROLLING_5_PACE',
        'away_ROLLING_5_OFF_RATING', 'away_ROLLING_5_DEF_RATING', 'away_ROLLING_5_PACE',
        'home_WIN' # This becomes our Target Variable (1 = Home Win)
    ]

    # Drop any rows with NaN values (which will mostly be the first 5-10 games of the 23-24 season due to rolling windows)
    matchups = matchups[final_cols].dropna().reset_index(drop=True)

    return matchups

# --- Execution ---
print("Starting pipeline...")
raw_data = fetch_nba_data(SEASONS)
engineered_data = engineer_team_features(raw_data)
final_dataset = build_matchup_dataset(engineered_data)
# Save the final dataset to a CSV file
final_dataset.to_csv('nba_training_data.csv', index=False)

print("\nPipeline Complete! Here is a preview of the training data:")
print(final_dataset.head())