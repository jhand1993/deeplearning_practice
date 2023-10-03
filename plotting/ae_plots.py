""" Plotting for autoencoder encoding prediction output """
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
from sklearn.metrics import confusion_matrix


def plot_2Dencoding(
    encoding_vec: np.ndarray,
    figsize: tuple = (8, 8),
    **kwargs
) -> plt.Figure:
    """ Returns a plot of encodings with optional keyword arguments for
        for plotting.

    Args:
        encoding_vec (ndarray): Column vector of encoding predictions.
        figsize (Tuple[float or int]): Figure size.  Defaults to (8, 8).

    Returns:
        plt.Figure: Figure object
    """
    fig = plt.figure(figsize=figsize)
    plt.scatter(
        encoding_vec[:, 0], encoding_vec[:, 1],
        **kwargs
    )
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    return fig


def plot_3Dencoding(
    encoding_vec: np.ndarray,
    figsize: tuple = (8, 8),
    **kwargs
) -> plt.Figure:
    """ Returns a 3D plot of encodings with optional keyword arguments for
        for plotting.

    Args:
        encoding_vec (ndarray): Column vector of encoding predictions.
        figsize (Tuple[float or int]): Figure size.  Defaults to (8, 8).

    Returns:
        plt.Figure: Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        encoding_vec[:, 0], encoding_vec[:, 1], encoding_vec[:, 2],
        **kwargs
    )
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.set_zlabel('feature 3')

    return fig


def plot_2Dencodings_withclusters(
    encoding_vec: np.ndarray,
    label_vec: np.ndarray,
    title: str,
    n_labels: int,
    label_names: Iterable[int | str] = None,
    figsize: tuple = (8, 8),
    cmap_name: str = 'plasma',
    encoding_kwargs: dict = {},
    clusters_kwargs: dict = {}
) -> plt.Figure:
    """ Plots a bivariate encoding vector along with cluster coloring
        based on labels given.

    Args:
        encoding_vec (ndarray): Column vector of encoding predictions.
        label_vec (ndarray): Column vector of label predictions.
        title (str): Plot title.
        n_labels (int): Number of labels used in the clustering algorithm.
        label_names (Iterable[int | string], optional): Cluster names for
            legend.  Defaults to zero.
        figsize (tuple, optional): Figure size. Defaults to (8, 8).
        cmap_name (str, optional): Colormap for cluster coloring.
            Defaults to 'plasma'.
        encoding_kwargs (dict, optional): Keyword argument dictionary for the
            encoding scatterplot. Defaults to {}.
        clusters_kwargs (dict, optional): Keyword arugment dictionary for the
            cluster coloring scatterplots. Defaults to {}.

    Returns:
        plt.Figure: Figure object.
    """
    # Default labels are label index numbers.
    if label_names is None:
        label_names = list(range(n_labels))

    # Get colors for requested colormap name.
    colors = plt.get_cmap(cmap_name)(np.linspace(0.2, 0.8, n_labels))

    fig = plt.figure(figsize=figsize)

    # First, plot the bivariate encoding sample.
    plt.scatter(
        encoding_vec[:, 0], encoding_vec[:, 1],
        **encoding_kwargs
    )
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    # Second, loop through label index and plot clusters.
    for i in range(n_labels):
        mask = label_vec == i
        plt.scatter(
            encoding_vec[:, 0][mask], encoding_vec[:, 1][mask],
            color=colors[i], label=label_names[i],
            **clusters_kwargs
        )

    # Only have five label names at most per column.
    n_legend_col = n_labels // 5
    if n_labels % 5 > 0:
        n_legend_col += 1

    plt.legend(ncol=n_legend_col)
    plt.title(title)

    return fig


def plot_3Dencodings_withclusters(
    encoding_vec: np.ndarray,
    label_vec: np.ndarray,
    title: str,
    n_labels: int,
    label_names: Iterable[int | str] = None,
    figsize: tuple = (8, 8),
    cmap_name: str = 'plasma',
    encoding_kwargs: dict = {},
    clusters_kwargs: dict = {}
) -> plt.Figure:
    """ Plots a trivariate encoding vector along with cluster coloring
        based on labels given.

    Args:
        encoding_vec (ndarray): Column vector of encoding predictions.
        label_vec (ndarray): Column vector of label predictions.
        title (str): Plot title.
        n_labels (int): Number of labels used in the clustering algorithm.
        label_names (Iterable[int | string], optional): Cluster names for
            legend.  Defaults to zero.
        figsize (tuple, optional): Figure size. Defaults to (8, 8).
        cmap_name (str, optional): Colormap for cluster coloring.
            Defaults to 'plasma'.
        encoding_kwargs (dict, optional): Keyword argument dictionary for the
            encoding scatterplot. Defaults to {}.
        clusters_kwargs (dict, optional): Keyword arugment dictionary for the
            cluster coloring scatterplots. Defaults to {}.

    Returns:
        plt.Figure: Figure object.
    """
    # Default labels are label index numbers.
    if label_names is None:
        label_names = list(range(n_labels))

    # Get colors for requested colormap name.
    colors = plt.get_cmap(cmap_name)(np.linspace(0.2, 0.8, n_labels))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    # First, plot the bivariate encoding sample.
    ax.scatter(
        encoding_vec[:, 0],
        encoding_vec[:, 1],
        encoding_vec[:, 2],
        **encoding_kwargs
    )
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.set_zlabel('feature 3')

    # Second, loop through label index and plot clusters.
    for i in range(n_labels):
        mask = label_vec == i
        ax.scatter(
            encoding_vec[:, 0][mask],
            encoding_vec[:, 1][mask],
            encoding_vec[:, 2][mask],
            color=colors[i], label=label_names[i],
            **clusters_kwargs
        )

    # Only have five label names at most per column.
    n_legend_col = n_labels // 5
    if n_labels % 5 > 0:
        n_legend_col += 1

    ax.legend(ncol=n_legend_col)
    ax.set_title(title)

    return fig


def matching_matrix_heatmap(
    title: str,
    label_true: Iterable[int | float],
    label_pred: Iterable[float],
    figsize: tuple = (7, 7.5),
    cmap: str = 'plasma',
    density: bool = True,
    **kwargs
) -> (np.ndarray, plt.Figure):
    """ Produces a matching matrix from true and predicted
        labels, returning both the matrix and a heatmap figure.

    Args:
        title (str): Figure title.
        label_true (Iterable[int | float]): True labels.
        label_pred (Iterable[float]): Predicted and unmatched labels.
        figsize: tuple[int]: Figure size.  Defaults to (7, 7.5).
        cmap (str, optional): Color map for heat map. Defaults to
            'plasma'.
        density (bool): If true, the confusion matrix is normalized
            by the len(label_true)

    Returns:
        tuple: Matching matrix and corresponding heat map figure.
    """
    # Normalize if requested.
    if density:
        n = len(label_true.flatten())

    else:
        n = 1

    # Plot the confusion matrix.
    cmatrix = confusion_matrix(label_true, label_pred) / n
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    p = ax.imshow(cmatrix, cmap=cmap, **kwargs)
    ticks = np.arange(0, int(np.max(label_true)) + 1, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel('True Label')
    ax.set_ylabel('Clustering Label')
    ax.set_title(title)
    fig.colorbar(p, ax=ax)

    return cmatrix, fig
