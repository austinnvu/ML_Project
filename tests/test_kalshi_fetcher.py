"""
Tests for Kalshi market parsing and game-price matching.
"""

import pandas as pd

from src.data.kalshi_fetcher import build_kalshi_game_prices


def test_build_kalshi_game_prices_maps_yes_price_to_home_team():
    nba_df = pd.DataFrame(
        [
            {
                "GAME_DATE": "2024-12-10",
                "home_TEAM_ABBREVIATION": "BOS",
                "away_TEAM_ABBREVIATION": "MIL",
            }
        ]
    )

    candidate_markets = [
        {
            "ticker": "KXNBA-BOSMIL-20241210",
            "event_ticker": "KXNBA-BOSMIL",
            "title": "Will the Boston Celtics beat the Milwaukee Bucks?",
            "subtitle": "Boston Celtics vs Milwaukee Bucks",
            "yes_sub_title": "Boston Celtics",
            "no_sub_title": "Milwaukee Bucks",
            "status": "settled",
            "market_source": "historical",
            "last_price_dollars": "0.6400",
            "volume_fp": "1250.00",
            "close_time": "2024-12-10T23:30:00Z",
            "settlement_ts": "2024-12-11T03:00:00Z",
        }
    ]

    result = build_kalshi_game_prices(nba_df, candidate_markets)

    assert len(result) == 1
    assert result.loc[0, "kalshi_market_ticker"] == "KXNBA-BOSMIL-20241210"
    assert result.loc[0, "kalshi_yes_team"] == "BOS"
    assert result.loc[0, "kalshi_price_home_win"] == 0.64


def test_build_kalshi_game_prices_flips_yes_price_when_yes_team_is_away():
    nba_df = pd.DataFrame(
        [
            {
                "GAME_DATE": "2024-12-10",
                "home_TEAM_ABBREVIATION": "BOS",
                "away_TEAM_ABBREVIATION": "MIL",
            }
        ]
    )

    candidate_markets = [
        {
            "ticker": "KXNBA-MILBOS-20241210",
            "event_ticker": "KXNBA-MILBOS",
            "title": "Will the Milwaukee Bucks beat the Boston Celtics?",
            "subtitle": "Milwaukee Bucks vs Boston Celtics",
            "yes_sub_title": "Milwaukee Bucks",
            "no_sub_title": "Boston Celtics",
            "status": "settled",
            "market_source": "historical",
            "last_price_dollars": "0.4100",
            "volume_fp": "900.00",
            "close_time": "2024-12-10T23:30:00Z",
            "settlement_ts": "2024-12-11T03:00:00Z",
        }
    ]

    result = build_kalshi_game_prices(nba_df, candidate_markets)

    assert len(result) == 1
    assert result.loc[0, "kalshi_yes_team"] == "MIL"
    assert result.loc[0, "kalshi_price_home_win"] == 0.59
