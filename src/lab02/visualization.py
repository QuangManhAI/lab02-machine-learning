"""Visualization helpers from the notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt

from lab02.config import IMAGES_DIR


def configure_matplotlib() -> None:
    """Apply the plotting defaults used in the original notebook."""
    plt.rc("font", size=14)
    plt.rc("axes", labelsize=14, titlesize=14)
    plt.rc("legend", fontsize=14)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)


def save_fig(
    fig_id: str,
    tight_layout: bool = True,
    fig_extension: str = "png",
    resolution: int = 300,
) -> None:
    """Save the active matplotlib figure to the project image directory."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGES_DIR / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
