"""
Random Forest model for ensemble prediction.
"""

from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier."""

    def __init__(self, config_path="config/config.yaml"):
        super().__init__(model_name="random_forest", config_path=config_path)

        params = self.config.get("models", {}).get("random_forest", {})
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 15),
            min_samples_split=params.get("min_samples_split", 10),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        """Train random forest model."""
        logger.info(f"Training random forest on {len(X)} samples...")
        self.model.fit(X, y)
        logger.info("Random forest training complete")

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

        logger.info("Evaluating random forest...")
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
