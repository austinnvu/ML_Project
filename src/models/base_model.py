"""
Abstract base class for all prediction models.

All concrete models (LogisticRegression, RandomForest, XGBoost) should inherit
from BaseModel to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for ML models.

    Defines the interface all models must implement:
      - fit(X, y): Train the model
      - predict_proba(X): Return probability predictions
      - evaluate(X, y): Return metrics dict
    """

    def __init__(self, model_name, config_path="config/config.yaml"):
        """
        Initialize model with configuration.

        Args:
            model_name (str): Name of the model (e.g., 'logistic', 'random_forest', 'xgboost')
            config_path (str): Path to config YAML file
        """
        self.model_name = model_name
        self.config = self._load_config(config_path)
        self.model = None
        logger.info(f"Initialized {model_name} model")

    def _load_config(self, config_path):
        """Load configuration from YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. Using defaults.")
            return {}

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model on data.

        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (np.ndarray): Binary target (0 or 1)
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame or np.ndarray): Features

        Returns:
            np.ndarray: Probability of positive class (shape: (n_samples,))
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate model on data.

        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (np.ndarray): Binary target

        Returns:
            dict: Metrics (accuracy, log_loss, brier_score, etc.)
        """
        pass
