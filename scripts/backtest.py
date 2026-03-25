"""
Backtesting script.

Runs a simulated trading strategy against historical market data.
"""

import argparse
from src.utils.logger import get_logger
from src.backtest.simulator import run_backtest

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtest simulation")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/games_with_prices.csv",
        help="Path to processed data"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/xgboost_model.pkl",
        help="Path to trained model"
    )
    parser.add_argument("--threshold", type=float, default=0.05, help="Divergence threshold")
    parser.add_argument("--bankroll", type=float, default=10000, help="Initial capital")

    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_path}...")
    logger.info(f"Loading model from {args.model_path}...")

    try:
        # TODO: Load data and model
        # df = pd.read_csv(args.data_path)
        # model = joblib.load(args.model_path)

        # results = run_backtest(model, df, threshold=args.threshold, bankroll=args.bankroll)

        logger.info("Backtest not yet fully implemented")
        logger.info(f"Threshold: {args.threshold}")
        logger.info(f"Bankroll: ${args.bankroll}")

        # logger.info(f"ROI: {results['roi']:.2%}")
        # logger.info(f"Number of trades: {results['n_trades']}")
        # logger.info(f"Win rate: {results['win_rate']:.2%}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")


if __name__ == "__main__":
    main()
