"""
Evaluation metrics for probability predictions.

Includes accuracy, log-loss, and Brier score for proper evaluation
of calibrated probability estimates.
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


def accuracy(y_true, y_pred_proba, threshold=0.5):
    """
    Compute accuracy using a probability threshold.

    Args:
        y_true (np.ndarray): Binary labels (0 or 1)
        y_pred_proba (np.ndarray): Predicted probabilities
        threshold (float): Classification threshold (default 0.5)

    Returns:
        float: Accuracy score [0, 1]
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return accuracy_score(y_true, y_pred)


def log_loss_score(y_true, y_pred_proba):
    """
    Compute log-loss (binary cross-entropy).

    Lower is better. Penalizes confident wrong predictions heavily.

    Args:
        y_true (np.ndarray): Binary labels
        y_pred_proba (np.ndarray): Predicted probabilities

    Returns:
        float: Log-loss score
    """
    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    return log_loss(y_true, y_pred_proba)


def brier_score(y_true, y_pred_proba):
    """
    Compute Brier score (mean squared error for probabilities).

    Lower is better. Measures calibration of probability estimates.

    Args:
        y_true (np.ndarray): Binary labels
        y_pred_proba (np.ndarray): Predicted probabilities

    Returns:
        float: Brier score [0, 1]
    """
    return brier_score_loss(y_true, y_pred_proba)


def classification_report_df(y_true, y_pred_proba, threshold=0.5):
    """
    Generate a detailed classification report.

    Args:
        y_true (np.ndarray): Binary labels
        y_pred_proba (np.ndarray): Predicted probabilities
        threshold (float): Classification threshold

    Returns:
        dict: Report with precision, recall, F1, support
    """
    from sklearn.metrics import classification_report

    y_pred = (y_pred_proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)
    return report
