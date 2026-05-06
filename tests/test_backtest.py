"""
Tests for backtesting simulator and Kalshi candlestick selection.
"""

import pandas as pd
import pytest

from src.backtest.simulator import (
    run_backtest,
    run_threshold_sweep,
    simulate_always_home,
    simulate_betting_roi,
    simulate_kalshi_favorite,
)
from src.data.kalshi_fetcher import _select_pregame_candle


def _make_predictions_df(rows):
    """Build a predictions DataFrame with the schema train.py emits."""
    base_cols = ("home_WIN", "model_prob_home_win", "kalshi_price_home_win")
    return pd.DataFrame(rows, columns=base_cols)


class TestSimulator:
    def test_imports(self):
        assert run_backtest is not None

    def test_simulate_betting_roi_known_inputs(self):
        # 4 games:
        #   row 0: model heavily favors home (edge +0.20), home wins -> profitable home bet at price 0.50
        #   row 1: model heavily favors away (edge +0.30 on away), away wins -> profitable away bet at price 1-0.30=0.70 -> wait
        #          model_home=0.10 -> model_away=0.90; kalshi_home=0.40 -> kalshi_away=0.60; away_edge=0.30; bet away at 0.60.
        #   row 2: edge below threshold -> no bet
        #   row 3: home edge but home loses -> -bet_amount
        df = _make_predictions_df([
            (1, 0.70, 0.50),
            (0, 0.10, 0.40),
            (1, 0.55, 0.52),
            (0, 0.80, 0.55),
        ])
        sim_df, summary = simulate_betting_roi(
            y_test=df["home_WIN"],
            y_prob_home=df["model_prob_home_win"],
            kalshi_prob_home=df["kalshi_price_home_win"],
            bet_amount=10.0,
            edge_threshold=0.05,
        )
        # 3 bets placed: rows 0, 1, 3. Row 2 has edge 0.03 < threshold.
        assert summary["total_bets"] == 3
        assert summary["total_wagered"] == pytest.approx(30.0)
        # Row 0 profit: 10 * (1/0.50) - 10 = 10
        # Row 1 profit: 10 * (1/0.60) - 10 = 6.6667
        # Row 3 profit: -10
        expected_profit = 10.0 + (10.0 / 0.60 - 10.0) + (-10.0)
        assert summary["net_profit"] == pytest.approx(expected_profit, rel=1e-3)

    def test_always_home_baseline_bets_every_game(self):
        df = _make_predictions_df([
            (1, 0.7, 0.5),
            (0, 0.4, 0.6),
            (1, 0.5, 0.55),
        ])
        _, summary = simulate_always_home(df, bet_amount=10.0)
        assert summary["total_bets"] == 3
        # Bets placed every row at home price; payout = -10 (loss) or 10*(1/p) - 10 (win)
        expected = (10.0 / 0.5 - 10.0) + (-10.0) + (10.0 / 0.55 - 10.0)
        assert summary["net_profit"] == pytest.approx(expected, rel=1e-3)

    def test_kalshi_favorite_skips_coin_flips(self):
        df = _make_predictions_df([
            (1, 0.5, 0.50),  # exact 0.5 -> no bet
            (1, 0.5, 0.70),  # home favorite, home wins -> bet home @ 0.70
            (0, 0.5, 0.30),  # away favorite (1-0.30=0.70), away wins -> bet away @ 0.70
        ])
        _, summary = simulate_kalshi_favorite(df, bet_amount=10.0)
        assert summary["total_bets"] == 2
        expected = (10.0 / 0.70 - 10.0) + (10.0 / 0.70 - 10.0)
        assert summary["net_profit"] == pytest.approx(expected, rel=1e-3)

    def test_threshold_sweep_monotonic_bet_count(self):
        df = _make_predictions_df([
            (1, 0.70, 0.50),
            (0, 0.10, 0.40),
            (1, 0.55, 0.52),
            (0, 0.80, 0.55),
            (1, 0.90, 0.40),
        ])
        sweep_df = run_threshold_sweep(df, thresholds=[0.0, 0.05, 0.10, 0.20], bet_amount=10.0)
        # As threshold increases, total_bets must be non-increasing.
        bet_counts = sweep_df["total_bets"].tolist()
        assert bet_counts == sorted(bet_counts, reverse=True)


class TestPregameCandleSelection:
    def test_picks_latest_candle_with_valid_quotes(self):
        candles = [
            # Earliest candle, has quotes
            {
                "end_period_ts": 1000,
                "yes_bid": {"close_dollars": "0.40"},
                "yes_ask": {"close_dollars": "0.42"},
                "volume_fp": "100",
            },
            # No quotes -- should be skipped
            {
                "end_period_ts": 1100,
                "yes_bid": {"close_dollars": None},
                "yes_ask": {"close_dollars": None},
                "volume_fp": "0",
            },
            # Latest valid candle
            {
                "end_period_ts": 1200,
                "yes_bid": {"close_dollars": "0.45"},
                "yes_ask": {"close_dollars": "0.46"},
                "volume_fp": "0",
            },
        ]
        chosen = _select_pregame_candle(candles)
        assert chosen is not None
        assert chosen["end_period_ts"] == 1200

    def test_returns_none_when_no_quotes_available(self):
        candles = [
            {
                "end_period_ts": 1000,
                "yes_bid": {"close_dollars": None},
                "yes_ask": {"close_dollars": None},
                "volume_fp": "0",
            }
        ]
        assert _select_pregame_candle(candles) is None
