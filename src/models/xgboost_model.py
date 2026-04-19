"""
XGBoost model for advanced ensemble prediction.
"""

import xgboost as xgb
from .base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost classifier (expected best performer for tabular data)."""

    def __init__(self, config_path="config/config.yaml"):
        super().__init__(model_name="xgboost", config_path=config_path)

        params = self.config.get("models", {}).get("xgboost", {})
        self.model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 6),
            min_child_weight=params.get("min_child_weight", 1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X, y):
        """Train XGBoost model."""
        logger.info(f"Training XGBoost on {len(X)} samples...")
        self.model.fit(X, y)
        logger.info("XGBoost training complete")

    def predict_proba(self, X):
        """Return probability of positive class."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        """
        Evaluate model on test data.

        Returns:
            dict: Accuracy, log-loss, brier score, feature importance, etc.
        """
        from src.evaluation.metrics import (
            accuracy, log_loss_score, brier_score
        )

        logger.info("Evaluating XGBoost...")
        proba = self.predict_proba(X)

        return {
            "accuracy": accuracy(y, proba),
            "log_loss": log_loss_score(y, proba),
            "brier_score": brier_score(y, proba),
            "feature_importance": dict(zip(
                range(X.shape[1]),
                self.model.feature_importances_
            ))
        }
