import os
import matplotlib.pyplot as plt
from typing import Sequence, Optional


def apply_default_style() -> None:
    """
    Apply the default plot style for consistent formatting across all plots.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 1,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.dpi': 300
    })


def plot_semilogy(
    y_values: Sequence[float],
    x_values: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
    marker: str = ',',
    linestyle: str = '-',
    xlabel: str = 'Iteration',
    ylabel: str = 'Residual Norm',
    title: Optional[str] = None,
    grid: bool = True,
    figsize: tuple = (8, 5)
) -> None:
    """
    Plot a semilog-y curve with default styling.

    Parameters
    ----------
    y_values : sequence of float
        Data for the y-axis.
    x_values : sequence of float, optional
        Data for the x-axis; defaults to range(len(y_values)).
    label : str, optional
        Legend label for the curve.
    marker : str
        Marker style.
    linestyle : str
        Line style.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str, optional
        Plot title.
    grid : bool
        Whether to show grid lines.
    figsize : tuple
        Figure size in inches.
    """
    apply_default_style()
    if x_values is None:
        x_values = list(range(len(y_values)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(x_values, y_values, marker=marker,
                linestyle=linestyle, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    if grid:
        ax.grid(True, which='both', linestyle='--')
    fig.tight_layout()


def save_plot(filename: str) -> None:
    """
    Save the current figure to the results folder with a given filename.

    Parameters
    ----------
    filename : str
        Name of the file to save inside the 'results' directory.
    """
    # Ensure the results directory exists
    results_dir = os.path.abspath(os.path.join(os.getcwd(), 'results'))
    os.makedirs(results_dir, exist_ok=True)
    # Build full path
    file_path = os.path.join(results_dir, filename)
    plt.savefig(file_path, dpi=300)
    plt.close()
