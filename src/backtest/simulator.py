"""
Backtesting helpers for the moneyline progress-report workflow.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


PREDICTION_REQUIRED_COLUMNS = {"home_WIN", "model_prob_home_win", "kalshi_price_home_win"}


def _validate_predictions_df(df: pd.DataFrame) -> None:
    missing = PREDICTION_REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Predictions DataFrame is missing required columns: {sorted(missing)}"
        )


def simulate_betting_roi(
    y_test,
    y_prob_home,
    kalshi_prob_home,
    metadata_df=None,
    bet_amount=10.0,
    edge_threshold=0.05,
):
    """
    Simulate a flat betting strategy on a test set.

    Returns:
        tuple[pd.DataFrame, dict]: Trade-level results and a summary dict
    """

    sim_df = pd.DataFrame(
        {
            "actual_outcome": pd.Series(y_test).astype(int).reset_index(drop=True),
            "model_prob_home": pd.Series(y_prob_home, dtype=float).reset_index(drop=True),
            "kalshi_prob_home": pd.Series(kalshi_prob_home, dtype=float).reset_index(drop=True),
        }
    )

    if metadata_df is not None:
        metadata_df = metadata_df.reset_index(drop=True)
        sim_df = pd.concat([metadata_df, sim_df], axis=1)

    sim_df["model_prob_away"] = 1.0 - sim_df["model_prob_home"]
    sim_df["kalshi_prob_away"] = 1.0 - sim_df["kalshi_prob_home"]
    sim_df["home_edge"] = sim_df["model_prob_home"] - sim_df["kalshi_prob_home"]
    sim_df["away_edge"] = sim_df["model_prob_away"] - sim_df["kalshi_prob_away"]
    sim_df["bet_placed_on"] = pd.NA
    sim_df["profit_loss"] = 0.0

    for idx, row in sim_df.iterrows():
        if row["home_edge"] > edge_threshold:
            sim_df.at[idx, "bet_placed_on"] = "Home"
            if row["actual_outcome"] == 1:
                profit = bet_amount * (1.0 / row["kalshi_prob_home"]) - bet_amount
                sim_df.at[idx, "profit_loss"] = profit
            else:
                sim_df.at[idx, "profit_loss"] = -bet_amount
        elif row["away_edge"] > edge_threshold:
            sim_df.at[idx, "bet_placed_on"] = "Away"
            if row["actual_outcome"] == 0:
                profit = bet_amount * (1.0 / row["kalshi_prob_away"]) - bet_amount
                sim_df.at[idx, "profit_loss"] = profit
            else:
                sim_df.at[idx, "profit_loss"] = -bet_amount

    total_bets = int(sim_df["bet_placed_on"].notna().sum())
    total_wagered = float(total_bets * bet_amount)
    net_profit = float(sim_df["profit_loss"].sum())
    roi_percent = float((net_profit / total_wagered) * 100.0) if total_wagered > 0 else 0.0

    summary = {
        "total_games": int(len(sim_df)),
        "total_bets": total_bets,
        "total_wagered": total_wagered,
        "net_profit": net_profit,
        "roi_percent": roi_percent,
    }

    logger.info(
        "Simulated %s qualifying bets across %s games for %.2f%% ROI",
        total_bets,
        len(sim_df),
        roi_percent,
    )

    return sim_df, summary


def _summarize(sim_df: pd.DataFrame, bet_amount: float) -> dict:
    total_bets = int(sim_df["bet_placed_on"].notna().sum())
    total_wagered = float(total_bets * bet_amount)
    net_profit = float(sim_df["profit_loss"].sum())
    roi_percent = float((net_profit / total_wagered) * 100.0) if total_wagered > 0 else 0.0
    return {
        "total_games": int(len(sim_df)),
        "total_bets": total_bets,
        "total_wagered": total_wagered,
        "net_profit": net_profit,
        "roi_percent": roi_percent,
    }


def _payout(bet_amount: float, market_prob: float, won: bool) -> float:
    if not won:
        return -bet_amount
    if market_prob <= 0:
        return 0.0
    return bet_amount * (1.0 / market_prob) - bet_amount


def simulate_always_home(
    predictions_df: pd.DataFrame,
    bet_amount: float = 10.0,
) -> Tuple[pd.DataFrame, dict]:
    """Naive baseline from the proposal: bet `bet_amount` on home in every game."""
    _validate_predictions_df(predictions_df)

    sim_df = predictions_df.reset_index(drop=True).copy()
    sim_df["bet_placed_on"] = "Home"
    sim_df["profit_loss"] = [
        _payout(bet_amount, float(price), bool(actual == 1))
        for price, actual in zip(sim_df["kalshi_price_home_win"], sim_df["home_WIN"])
    ]

    summary = _summarize(sim_df, bet_amount)
    logger.info(
        "always-home baseline: %s bets, %.2f%% ROI",
        summary["total_bets"],
        summary["roi_percent"],
    )
    return sim_df, summary


def simulate_kalshi_favorite(
    predictions_df: pd.DataFrame,
    bet_amount: float = 10.0,
) -> Tuple[pd.DataFrame, dict]:
    """Baseline that mirrors the market: bet on whichever side Kalshi prices > 0.5."""
    _validate_predictions_df(predictions_df)

    sim_df = predictions_df.reset_index(drop=True).copy()
    sim_df["bet_placed_on"] = pd.NA
    sim_df["profit_loss"] = 0.0

    for idx, row in sim_df.iterrows():
        kalshi_home = float(row["kalshi_price_home_win"])
        actual_home_win = int(row["home_WIN"]) == 1
        if kalshi_home > 0.5:
            sim_df.at[idx, "bet_placed_on"] = "Home"
            sim_df.at[idx, "profit_loss"] = _payout(bet_amount, kalshi_home, actual_home_win)
        elif kalshi_home < 0.5:
            sim_df.at[idx, "bet_placed_on"] = "Away"
            sim_df.at[idx, "profit_loss"] = _payout(
                bet_amount, 1.0 - kalshi_home, not actual_home_win
            )
        # exactly 0.5 -> no bet (skip coin-flips)

    summary = _summarize(sim_df, bet_amount)
    logger.info(
        "kalshi-favorite baseline: %s bets, %.2f%% ROI",
        summary["total_bets"],
        summary["roi_percent"],
    )
    return sim_df, summary


def run_threshold_sweep(
    predictions_df: pd.DataFrame,
    thresholds: Iterable[float] = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15),
    bet_amount: float = 10.0,
) -> pd.DataFrame:
    """Run the model's edge-threshold simulator across a list of thresholds."""
    _validate_predictions_df(predictions_df)

    rows = []
    for threshold in thresholds:
        _, summary = simulate_betting_roi(
            y_test=predictions_df["home_WIN"],
            y_prob_home=predictions_df["model_prob_home_win"],
            kalshi_prob_home=predictions_df["kalshi_price_home_win"],
            metadata_df=None,
            bet_amount=bet_amount,
            edge_threshold=float(threshold),
        )
        rows.append({"threshold": float(threshold), **summary})

    return pd.DataFrame(rows)


def simulate_baselines(
    predictions_df: pd.DataFrame,
    bet_amount: float = 10.0,
    edge_threshold: float = 0.05,
    model_name: str = "model",
    other_predictions: Optional[dict] = None,
) -> pd.DataFrame:
    """Compare the model's strategy against the proposal's named baselines.

    `other_predictions` maps strategy_name -> predictions_df for additional
    model-vs-model comparisons (e.g. logistic regression). Each extra
    DataFrame must have the same prediction-column schema.
    """
    _validate_predictions_df(predictions_df)

    rows: list[dict] = []

    _, model_summary = simulate_betting_roi(
        y_test=predictions_df["home_WIN"],
        y_prob_home=predictions_df["model_prob_home_win"],
        kalshi_prob_home=predictions_df["kalshi_price_home_win"],
        bet_amount=bet_amount,
        edge_threshold=edge_threshold,
    )
    rows.append({"strategy": f"{model_name} (edge>{edge_threshold})", **model_summary})

    _, home_summary = simulate_always_home(predictions_df, bet_amount=bet_amount)
    rows.append({"strategy": "always_home (naive)", **home_summary})

    _, fav_summary = simulate_kalshi_favorite(predictions_df, bet_amount=bet_amount)
    rows.append({"strategy": "kalshi_favorite", **fav_summary})

    if other_predictions:
        for other_name, other_df in other_predictions.items():
            _validate_predictions_df(other_df)
            _, other_summary = simulate_betting_roi(
                y_test=other_df["home_WIN"],
                y_prob_home=other_df["model_prob_home_win"],
                kalshi_prob_home=other_df["kalshi_price_home_win"],
                bet_amount=bet_amount,
                edge_threshold=edge_threshold,
            )
            rows.append({"strategy": f"{other_name} (edge>{edge_threshold})", **other_summary})

    return pd.DataFrame(rows)


def load_predictions_for_baseline(model_dir: str) -> Optional[pd.DataFrame]:
    """Load test_predictions.csv from an artifacts/<model> dir if it exists.

    Used by the backtest CLI to opportunistically include peer models (e.g.
    logistic regression) in the baseline comparison.
    """
    path = os.path.join(model_dir, "test_predictions.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not PREDICTION_REQUIRED_COLUMNS.issubset(df.columns):
        return None
    return df


def run_backtest(model, games_df, market_df=None, threshold=0.05, bankroll=10000, bet_amount=10.0):
    """
    Compatibility wrapper around the flat ROI simulator.
    """

    if "kalshi_price_home_win" not in games_df.columns:
        if market_df is None or "kalshi_price_home_win" not in market_df.columns:
            raise ValueError("Kalshi home-win probability column is required for backtesting.")
        aligned_df = games_df.join(market_df[["kalshi_price_home_win"]])
    else:
        aligned_df = games_df.copy()

    if "home_WIN" not in aligned_df.columns:
        raise ValueError("Expected home_WIN column in games_df for backtesting.")

    feature_df = aligned_df.select_dtypes(include=["number", "bool"]).drop(
        columns=["home_WIN", "kalshi_price_home_win"],
        errors="ignore",
    )
    y_prob_home = model.predict_proba(feature_df)

    results_df, summary = simulate_betting_roi(
        y_test=aligned_df["home_WIN"],
        y_prob_home=y_prob_home,
        kalshi_prob_home=aligned_df["kalshi_price_home_win"],
        metadata_df=aligned_df[[col for col in aligned_df.columns if col not in {"home_WIN", "kalshi_price_home_win"}]],
        bet_amount=bet_amount,
        edge_threshold=threshold,
    )

    return {
        "roi": summary["roi_percent"] / 100.0,
        "n_trades": summary["total_bets"],
        "win_rate": float((results_df["profit_loss"] > 0).mean()) if len(results_df) else 0.0,
        "details": results_df,
        "bankroll": bankroll,
    }
