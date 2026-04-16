"""
Backtesting helpers for the moneyline progress-report workflow.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


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
