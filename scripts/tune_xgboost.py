"""
Tune XGBoost hyperparameters on a chronological validation slice.

Carves the last ~20% of the training set as a validation slice, runs a small
grid sweep with early stopping, and reports the best config by validation
log loss. The held-out test set is only scored once at the end for sanity.
"""

import argparse
import itertools
import json
import os
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    load_and_prep_data,
    prepare_train_test,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


PARAM_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

EARLY_STOPPING_ROUNDS = 25
MAX_BOOST_ROUNDS = 500


def split_train_val(train_df: pd.DataFrame, feature_columns, val_fraction: float):
    split_idx = int(len(train_df) * (1 - val_fraction))
    inner_df = train_df.iloc[:split_idx]
    val_df = train_df.iloc[split_idx:]
    X_inner = inner_df[feature_columns]
    y_inner = inner_df[TARGET_COLUMN].astype(int)
    X_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN].astype(int)
    return X_inner, y_inner, X_val, y_val


def score(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def fit_with_early_stopping(params, X_inner, y_inner, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=MAX_BOOST_ROUNDS,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **params,
    )
    model.fit(X_inner, y_inner, eval_set=[(X_val, y_val)], verbose=False)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search XGBoost hyperparameters")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for the final held-out test (matches train.py)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of the training set carved off chronologically for validation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/xgboost_tuning",
        help="Directory for sweep results JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_and_prep_data(args.data_path)
    train_df, test_df, X_train, X_test, y_train, y_test, feature_columns = prepare_train_test(
        df, test_fraction=args.test_size
    )
    X_inner, y_inner, X_val, y_val = split_train_val(train_df, feature_columns, args.val_size)

    logger.info(
        "Sweep splits — inner train: %s, val: %s, test (held out): %s, features: %s",
        len(X_inner),
        len(X_val),
        len(X_test),
        len(feature_columns),
    )

    grid_keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in grid_keys]))
    logger.info(f"Sweeping {len(combos)} combinations")

    results = []
    for combo in combos:
        params = dict(zip(grid_keys, combo))
        model = fit_with_early_stopping(params, X_inner, y_inner, X_val, y_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = score(y_val, val_prob)
        n_trees = int(model.best_iteration + 1) if model.best_iteration is not None else MAX_BOOST_ROUNDS
        results.append({
            "params": params,
            "n_estimators_used": n_trees,
            "val": val_metrics,
        })

    results.sort(key=lambda r: r["val"]["log_loss"])
    best = results[0]

    print("\n--- Top 5 by validation log loss ---")
    print(f"{'log_loss':>10} {'roc_auc':>9} {'acc':>7} {'n_est':>6}  params")
    for r in results[:5]:
        p = r["params"]
        print(
            f"{r['val']['log_loss']:>10.4f} {r['val']['roc_auc']:>9.4f} "
            f"{r['val']['accuracy']:>7.4f} {r['n_estimators_used']:>6}  {p}"
        )

    print("\n--- Best config ---")
    print(json.dumps(best, indent=2))

    print("\n--- Held-out test score for best config (sanity check) ---")
    best_model = fit_with_early_stopping(best["params"], X_inner, y_inner, X_val, y_val)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = score(y_test, test_prob)
    print(json.dumps(test_metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "splits": {
                "inner_train": len(X_inner),
                "val": len(X_val),
                "test": len(X_test),
            },
            "results": results,
            "best": best,
            "best_test_metrics": test_metrics,
        }, f, indent=2)
    logger.info(f"Saved sweep results to {summary_path}")


if __name__ == "__main__":
    main()
