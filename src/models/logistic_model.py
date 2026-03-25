"""
Logistic Regression model for baseline prediction.
"""

from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression with L2 regularization (Ridge).

    TODO: Load hyperparameters from config['models']['logistic_regression']
    """

    def __init__(self, config_path="config/config.yaml"):
        super().__init__(model_name="logistic_regression", config_path=config_path)

        # TODO: Extract from config
        # For now, use reasonable defaults
        self.model = LogisticRegression(
            C=1.0,  # Inverse regularization strength
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X, y):
        """Train logistic regression model."""
        logger.info(f"Training logistic regression on {len(X)} samples...")
        self.model.fit(X, y)
        logger.info("Logistic regression training complete")

    def predict_proba(self, X):
        """Return probability of positive class."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        """
        Evaluate model on test data.

        Returns:
            dict: Accuracy, log-loss, brier score, etc.
        """
        from src.evaluation.metrics import (
            accuracy, log_loss_score, brier_score
        )

        logger.info("Evaluating logistic regression...")
        proba = self.predict_proba(X)

        return {
            "accuracy": accuracy(y, proba),
            "log_loss": log_loss_score(y, proba),
            "brier_score": brier_score(y, proba),
        }
