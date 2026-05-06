"""
Plotting helpers for backtest artifacts.

Each function takes data + an output path and writes a PNG to disk.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering for CI / scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_dir(output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_cumulative_pnl(sim_df: pd.DataFrame, output_path: str) -> None:
    """Plot running profit over chronological game dates."""
    _ensure_dir(output_path)

    df = sim_df.copy()
    if "GAME_DATE" not in df.columns:
        raise ValueError("sim_df must have a GAME_DATE column to plot cumulative P&L")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["cumulative_pnl"] = df["profit_loss"].fillna(0.0).cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(df["GAME_DATE"], df["cumulative_pnl"], linewidth=1.6)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Cumulative Profit/Loss Over Test Set")
    plt.xlabel("Game Date")
    plt.ylabel("Cumulative $ P&L")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved cumulative P&L plot to {output_path}")


def plot_roi_vs_threshold(sweep_df: pd.DataFrame, output_path: str) -> None:
    """Plot ROI% and bet count against edge threshold."""
    _ensure_dir(output_path)

    df = sweep_df.sort_values("threshold").reset_index(drop=True)

    fig, ax_roi = plt.subplots(figsize=(10, 5))
    ax_roi.plot(df["threshold"], df["roi_percent"], color="C0", marker="o", label="ROI %")
    ax_roi.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax_roi.set_xlabel("Edge threshold")
    ax_roi.set_ylabel("ROI (%)", color="C0")
    ax_roi.tick_params(axis="y", labelcolor="C0")

    ax_bets = ax_roi.twinx()
    ax_bets.bar(df["threshold"], df["total_bets"], color="C1", alpha=0.25, width=0.012, label="bets")
    ax_bets.set_ylabel("Total bets", color="C1")
    ax_bets.tick_params(axis="y", labelcolor="C1")

    plt.title("ROI vs Edge Threshold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved ROI-vs-threshold plot to {output_path}")


def plot_calibration(
    y_true: Sequence[int],
    y_prob_model: Sequence[float],
    y_prob_kalshi: Sequence[float],
    output_path: str,
    n_bins: int = 10,
) -> None:
    """Reliability diagram comparing model and Kalshi calibration against truth."""
    _ensure_dir(output_path)

    y_true = np.asarray(y_true).astype(int)
    y_prob_model = np.asarray(y_prob_model).astype(float)
    y_prob_kalshi = np.asarray(y_prob_kalshi).astype(float)

    frac_model, mean_model = calibration_curve(y_true, y_prob_model, n_bins=n_bins, strategy="quantile")
    frac_kalshi, mean_kalshi = calibration_curve(y_true, y_prob_kalshi, n_bins=n_bins, strategy="quantile")

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    plt.plot(mean_model, frac_model, marker="o", label="Model")
    plt.plot(mean_kalshi, frac_kalshi, marker="s", label="Kalshi (market)")
    plt.xlabel("Predicted probability of home win")
    plt.ylabel("Empirical frequency of home win")
    plt.title("Calibration: Model vs. Kalshi")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved calibration plot to {output_path}")
