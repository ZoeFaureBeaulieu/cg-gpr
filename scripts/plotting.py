import numpy as np
import matplotlib
import pandas as pd
from typing import Tuple
from pathlib import Path
from .data import get_reference_structure, normalise_energies

# the parent directory
root_dir = Path(__file__).resolve().parent.parent
# directories to the data for plotting
grid_search_results = root_dir / "results/grid_search"
learning_curve_results = root_dir / "results/learning_curve"
gpr_with_cv_results = root_dir / "results/gpr"


def set_axes_labels(
    ax: matplotlib.axes.Axes,
    x_label: str,
    y_label: str,
) -> None:
    """Set the axes labels.

    Args:
        ax (matplotlib.axes.Axes): the axes object
        x_label (str): x label
        y_label (str): y label
    """
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)


def set_axes_scale(
    ax: matplotlib.axes.Axes, x_scale: str = None, y_scale: str = None
) -> None:
    """Set the axes scale.

    Args:
        ax (matplotlib.axes.Axes): the axes object
        x_scale (str, optional): x scale. Defaults to None.
        y_scale (str, optional): y scale. Defaults to None.
    """
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)


def set_axes_limits(
    ax: matplotlib.axes.Axes,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
) -> None:
    """Set the axes limits.

    Args:
        ax (matplotlib.axes.Axes): the axes object
        x_min (float, optional): the lower x limit. Defaults to None.
        x_max (float, optional): the upper x limit. Defaults to None.
        y_min (float, optional): the lower y limit. Defaults to None.
        y_max (float, optional): the upper y limit. Defaults to None.
    """
    if x_min is not None:
        ax.set_xlim(x_min, x_max)
    if y_min is not None:
        ax.set_ylim(y_min, y_max)


def set_axes_ticks(
    ax: matplotlib.axes.Axes,
    x_ticks: np.ndarray = None,
    y_ticks: np.ndarray = None,
    x_labels: np.ndarray = None,
    y_labels: np.ndarray = None,
    minor_ticks: bool = True,
) -> None:
    """Set the ticks and labels for the axes.

    Args:
        ax (matplotlib.axes.Axes): the axes object
        x_ticks (np.ndarray): the x ticks
        y_ticks (np.ndarray): the y ticks
        x_labels (np.ndarray): the x labels
        y_labels (np.ndarray): the y labels
        minor_ticks (bool, optional): whether to include minor ticks. Defaults to True.
    """
    ax.tick_params(axis="both", which="major", pad=8)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

    if minor_ticks == False:
        ax.minorticks_off()


def get_plot_colour(struct_type: str) -> str:
    """Get the plot colour for a given structure type.

    Args:
        struct_type (str): structure type: 'cg', 'A_cg', 'atomistic'

    Returns:
        str: the hex colour code
    """
    if struct_type == "cg":
        return "#76c8a5"
    elif struct_type == "A_cg":
        return "#d0384e"
    elif struct_type == "atomistic":
        return "#466eb1"


def log_spaced_levels(min: float, max: float, n: int) -> np.ndarray:
    """Get a set of n log-spaced levels between min and max.

    Args:
        min (float): the minimum value
        max (float): the maximum value
        n (int): the number of levels

    Returns:
        np.ndarray: the levels
    """
    return np.logspace(np.log10(min), np.log10(max), n)


def remove_plot_border(ax: matplotlib.axes.Axes) -> None:
    """Remove the border from a matplotlib plot.

    Args:
        ax (matplotlib.axes.Axes): the axes object
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def get_minima_labels(df: pd.DataFrame, sort_by: str) -> Tuple[float, float]:
    """Obtain the minimum x and y values from a dataframe according to a sorting criteria.

    Args:
        df (pd.DataFrame): the input data
        sort_by (str): the sorting criteria; e.g. 'result.av_test_rmse'

    Returns:
        Tuple[float, float]: the x and y labels of the minimum
    """
    best = df.sort_values(by=sort_by).iloc[0]
    x_label = best["config.sigma"]
    y_label = best["config.cutoff"]

    return x_label, y_label


def get_learning_curve_data(
    struct_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Obtain results from the learning curve experiments for a given structure type.

    Args:
        struct_type (str): the structure type: 'cg', 'A_cg', 'atomistic'

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: the x and y values (training set size and RMSE respectively); and the y ticks and labels.
    """
    df = pd.read_csv(learning_curve_results / f"lc_lMax8_{struct_type}.csv")

    index = df["numb_training_atoms"] > 5
    x = df["numb_training_atoms"][index]
    y = df["av_test_rmse"][index]

    ticks = [0.02, 0.04, 0.08, 0.16, 0.32]
    labels = ["0.02", "0.04", "0.08", "0.16", "0.32"]

    return x, y, ticks, labels


def get_grid_search_data(
    struct_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Obtain results from a grid search.

    Args:
        struct_type (str): the structure type: 'cg', 'A_cg', 'atomistic'

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]: x, y, z, levels, ticks, x_min, y_min.
    """
    df = pd.read_csv(grid_search_results / struct_type / "results.csv")

    x = list(df["config.sigma"])
    y = list(df["config.cutoff"])
    z = df["result.av_test_rmse"]

    # Find the minimum
    x_min, y_min = get_minima_labels(df, sort_by="result.av_test_rmse")

    return x, y, z, x_min, y_min


def get_gpr_data(
    struct_type: str,
    hypers_type: str,
    numb_train: int = 32000,
    energy_type: str = "local_energies",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Get the results from GPR with cross-validation.

    Args:
        struct_type (str): the structure type: 'cg', 'A_cg', 'atomistic'
        hypers_type (str): the type of optimised SOAP hyperparameters used: 'cg', 'A_cg', 'atomistic'
        numb_train (int, optional): the number of training atoms. Defaults to 32000 (as used in the paper).
        energy_type (str, optional): the type of energy used: 'local_energies' or 'zn_energies'. Defaults to 'local_energies' (ε(local) defined in the paper).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: the test set predictions and labels (energies) and the average test RMSE.
    """

    # Get the reference structure; this is the lowest energy structure
    # It is used to normalise the energies
    ref_struct = get_reference_structure()

    # Load the GPR results
    gpr_data = np.load(
        gpr_with_cv_results
        / f"{energy_type}/{hypers_type}_hypers/gpr_{struct_type}_ntrain{numb_train}.npy",
        allow_pickle=True,
    ).item()

    # Normalise the energies
    test_preds = normalise_energies(gpr_data["test_predictions"], ref_struct)
    test_labels = normalise_energies(gpr_data["test_labels"], ref_struct)
    rmse = gpr_data["av_test_rmse"]

    return test_preds, test_labels, rmse
