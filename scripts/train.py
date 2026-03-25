"""
Model training script.

Trains logistic regression, random forest, and XGBoost models.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.models.logistic_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train prediction models")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/games_with_prices.csv",
        help="Path to processed data"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["moneyline", "over_under"],
        default="moneyline",
        help="Prediction target"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")

    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_path}...")

    try:
        # TODO: Load processed data
        # df = pd.read_csv(args.data_path)
        # X = df.drop(columns=[args.target])
        # y = df[args.target]

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=args.test_size, random_state=42
        # )

        logger.info("Data loading not yet fully implemented")
        logger.info(f"Target: {args.target}")
        logger.info(f"Test size: {args.test_size}")

        # TODO: Train and evaluate each model
        # models = [
        #     LogisticRegressionModel(),
        #     RandomForestModel(),
        #     XGBoostModel(),
        # ]

        # for model in models:
        #     logger.info(f"Training {model.model_name}...")
        #     model.fit(X_train, y_train)
        #
        #     results = model.evaluate(X_test, y_test)
        #     logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()
