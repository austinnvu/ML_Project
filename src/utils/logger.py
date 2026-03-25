"""
Centralized logging configuration for the project.
"""

import logging
import sys


def get_logger(name):
    """
    Returns a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger configured with console handler
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
