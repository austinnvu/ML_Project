"""
Train a moneyline logistic regression baseline on the merged NBA + Kalshi data.

Outputs:
  - Model artifact with fitted scaler and feature names
  - Test-set predictions with Kalshi price for ROI simulation
  - Metrics JSON
  - Coefficient CSV
  - Coefficient plot PNG
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_DATA_PATH = "data_warehouse/nba_training_data_with_kalshi.csv"
DEFAULT_OUTPUT_DIR = "artifacts/logistic_moneyline"
TARGET_COLUMN = "home_WIN"
MARKET_COLUMN = "kalshi_price_home_win"
METADATA_COLUMNS = [
    "GAME_DATE",
    "home_TEAM_ABBREVIATION",
    "away_TEAM_ABBREVIATION",
]


def load_and_prep_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


def prepare_train_test(df: pd.DataFrame, test_fraction: float):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if MARKET_COLUMN not in df.columns:
        raise ValueError(f"Missing Kalshi price column: {MARKET_COLUMN}")

    working_df = df.dropna(subset=[TARGET_COLUMN, MARKET_COLUMN]).copy()

    numeric_features = working_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_columns = [
        column
        for column in numeric_features
        if column not in {TARGET_COLUMN}
        and not column.startswith("kalshi_")
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns were found for model training.")

    model_df = working_df[METADATA_COLUMNS + feature_columns + [TARGET_COLUMN, MARKET_COLUMN]].dropna().copy()

    split_idx = int(len(model_df) * (1 - test_fraction))
    if split_idx <= 0 or split_idx >= len(model_df):
        raise ValueError(
            f"Chronological split produced an invalid split index ({split_idx}) "
            f"for {len(model_df)} rows. Adjust --test-size."
        )

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)

    return train_df, test_df, X_train, X_test, y_train, y_test, feature_columns


def train_logistic_model(X_train: pd.DataFrame, y_train: pd.Series):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_model(model, X_test_scaled, y_test: pd.Series):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
    }

    return metrics, y_pred, y_prob


def save_outputs(
    output_dir: str,
    model,
    scaler,
    feature_columns,
    metrics,
    test_df: pd.DataFrame,
    y_pred,
    y_prob,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    predictions_df = test_df[METADATA_COLUMNS + [TARGET_COLUMN, MARKET_COLUMN]].copy()
    predictions_df["predicted_home_win"] = y_pred
    predictions_df["model_prob_home_win"] = y_prob
    predictions_path = os.path.join(output_dir, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": model.coef_[0],
        }
    ).sort_values(by="coefficient", key=lambda series: series.abs(), ascending=False)
    coefficients_path = os.path.join(output_dir, "coefficients.csv")
    coefficients_df.to_csv(coefficients_path, index=False)

    plot_path = os.path.join(output_dir, "coefficients_top15.png")
    top_coefficients = coefficients_df.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_coefficients["feature"], top_coefficients["coefficient"])
    plt.title("Top 15 Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "market_column": MARKET_COLUMN,
    }
    model_path = os.path.join("models", "logistic_regression_moneyline.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(artifact, model_file)

    logger.info(f"Saved metrics to {metrics_path}")
    logger.info(f"Saved test predictions to {predictions_path}")
    logger.info(f"Saved coefficients to {coefficients_path}")
    logger.info(f"Saved coefficient plot to {plot_path}")
    logger.info(f"Saved model artifact to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a moneyline logistic regression baseline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to merged NBA + Kalshi CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of most recent games to use as the chronological test set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, predictions, and plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading merged dataset from {args.data_path}")
    df = load_and_prep_data(args.data_path)
    logger.info(f"Loaded {len(df)} rows")

    train_df, test_df, X_train, X_test, y_train, y_test, feature_columns = prepare_train_test(
        df,
        test_fraction=args.test_size,
    )

    logger.info(
        "Training logistic regression with %s training rows, %s test rows, and %s features",
        len(train_df),
        len(test_df),
        len(feature_columns),
    )

    model, scaler = train_logistic_model(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    metrics, y_pred, y_prob = evaluate_model(model, X_test_scaled, y_test)

    print("--- Logistic Regression Baseline ---")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")
    print(f"Features:   {len(feature_columns)}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"ROC AUC:    {metrics['roc_auc']:.4f}")
    print(f"Log Loss:   {metrics['log_loss']:.4f}")
    print(f"Brier:      {metrics['brier_score']:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    save_outputs(
        output_dir=args.output_dir,
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        metrics=metrics,
        test_df=test_df,
        y_pred=y_pred,
        y_prob=y_prob,
    )


if __name__ == "__main__":
    main()
