"""
Run an ROI simulation on saved model predictions.

Defaults to the XGBoost model. Supports threshold sweeps, baseline comparisons,
and PNG visualizations alongside the trade-level CSV.
"""

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest.simulator import (
    PREDICTION_REQUIRED_COLUMNS,
    load_predictions_for_baseline,
    run_threshold_sweep,
    simulate_baselines,
    simulate_betting_roi,
)
from src.backtest.visualizations import (
    plot_calibration,
    plot_cumulative_pnl,
    plot_roi_vs_threshold,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


MODEL_DIRS = {
    "logistic": "artifacts/logistic_moneyline",
    "random_forest": "artifacts/random_forest",
    "xgboost": "artifacts/xgboost",
}
MODEL_DISPLAY_NAMES = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
DEFAULT_THRESHOLDS = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ROI simulation on saved model predictions")
    parser.add_argument(
        "--model",
        type=str,
        choices=tuple(MODEL_DIRS.keys()),
        default="xgboost",
        help="Which model's predictions to backtest",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=None,
        help="Override path to test_predictions.csv (defaults to artifacts/<model>/test_predictions.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (defaults to artifacts/<model>)",
    )
    parser.add_argument("--bet-amount", type=float, default=10.0)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def warn_if_prices_concentrated(predictions_df: pd.DataFrame) -> None:
    """Print a warning if kalshi_price_home_win is suspiciously clustered.

    Guards against the upstream Kalshi pre-game-price fix being silently
    bypassed (the previous data had 96% of rows pinned to 0.995).
    """
    prices = predictions_df["kalshi_price_home_win"].dropna()
    if prices.empty:
        return
    most_common = prices.round(2).value_counts(normalize=True).iloc[0]
    if most_common > 0.5:
        top_value = prices.round(2).value_counts().index[0]
        logger.warning(
            "kalshi_price_home_win is %.0f%% concentrated at ~%.2f -- "
            "predictions likely use settlement prices, not pre-game prices. "
            "Re-run scripts/fetch_kalshi_data.py.",
            most_common * 100,
            top_value,
        )


def write_summary_json(output_dir: str, summary: dict) -> None:
    path = os.path.join(output_dir, "roi_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Saved ROI summary to {path}")


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or MODEL_DIRS[args.model]
    predictions_path = args.predictions_path or os.path.join(
        MODEL_DIRS[args.model], "test_predictions.csv"
    )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading {args.model} predictions from {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)

    missing = PREDICTION_REQUIRED_COLUMNS.difference(predictions_df.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {sorted(missing)}")

    warn_if_prices_concentrated(predictions_df)

    metadata_cols = [c for c in predictions_df.columns if c not in PREDICTION_REQUIRED_COLUMNS]
    sim_df, summary = simulate_betting_roi(
        y_test=predictions_df["home_WIN"],
        y_prob_home=predictions_df["model_prob_home_win"],
        kalshi_prob_home=predictions_df["kalshi_price_home_win"],
        metadata_df=predictions_df[metadata_cols],
        bet_amount=args.bet_amount,
        edge_threshold=args.edge_threshold,
    )

    bets_path = os.path.join(output_dir, "roi_bets.csv")
    sim_df.to_csv(bets_path, index=False)
    write_summary_json(output_dir, summary)

    print(f"--- {MODEL_DISPLAY_NAMES[args.model]} ROI Simulation ---")
    print(f"Total Games:          {summary['total_games']}")
    print(f"Total Bets Placed:    {summary['total_bets']} (edge > {args.edge_threshold:.2%})")
    print(f"Total Wagered:        ${summary['total_wagered']:.2f}")
    print(f"Net P&L:              ${summary['net_profit']:.2f}")
    print(f"ROI:                  {summary['roi_percent']:.2f}%")

    if args.threshold_sweep:
        sweep_df = run_threshold_sweep(
            predictions_df, thresholds=DEFAULT_THRESHOLDS, bet_amount=args.bet_amount
        )
        sweep_path = os.path.join(output_dir, "roi_threshold_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print("\n--- Threshold sweep ---")
        print(sweep_df.to_string(index=False))
        logger.info(f"Saved threshold sweep to {sweep_path}")

    if args.baselines:
        other_predictions = {}
        for peer in ("logistic", "random_forest", "xgboost"):
            if peer == args.model:
                continue
            peer_df = load_predictions_for_baseline(MODEL_DIRS[peer])
            if peer_df is not None:
                other_predictions[MODEL_DISPLAY_NAMES[peer]] = peer_df

        baseline_df = simulate_baselines(
            predictions_df,
            bet_amount=args.bet_amount,
            edge_threshold=args.edge_threshold,
            model_name=MODEL_DISPLAY_NAMES[args.model],
            other_predictions=other_predictions,
        )
        baseline_path = os.path.join(output_dir, "roi_baseline_comparison.csv")
        baseline_df.to_csv(baseline_path, index=False)
        print("\n--- Strategy comparison ---")
        print(baseline_df.to_string(index=False))
        logger.info(f"Saved baseline comparison to {baseline_path}")

    if not args.no_plots:
        plot_cumulative_pnl(sim_df, os.path.join(output_dir, "cumulative_pnl.png"))
        if args.threshold_sweep:
            plot_roi_vs_threshold(sweep_df, os.path.join(output_dir, "roi_vs_threshold.png"))
        plot_calibration(
            y_true=predictions_df["home_WIN"],
            y_prob_model=predictions_df["model_prob_home_win"],
            y_prob_kalshi=predictions_df["kalshi_price_home_win"],
            output_path=os.path.join(output_dir, "calibration.png"),
        )

    logger.info(f"Saved trade-level results to {bets_path}")


if __name__ == "__main__":
    main()
