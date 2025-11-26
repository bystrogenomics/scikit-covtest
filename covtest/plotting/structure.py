from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_eigen_scree(
    cov: np.ndarray,
    n_eigs: Optional[int] = None,
    show_mp: bool = True,
    sample_size: Optional[int] = None,
) -> None:
    """
    Plot a scree plot of the eigenvalues of a covariance matrix.

    Parameters
    ----------
    cov : ndarray of shape (p, p)
        Covariance or correlation matrix.
    n_eigs : int, optional
        Number of top eigenvalues to plot. Defaults to all.
    show_mp : bool, default=True
        Whether to overlay Marchenko–Pastur upper bound (requires sample_size).
    sample_size : int, optional
        Number of samples used to estimate the covariance.
        Needed if show_mp=True to compute the MP bulk.
    """
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]  # descending order

    if n_eigs is not None:
        eigvals = eigvals[:n_eigs]

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", linewidth=2)
    plt.xlabel("Eigenvalue index", fontsize=12)
    plt.ylabel("Eigenvalue magnitude", fontsize=12)
    plt.title("Eigenvalue Scree Plot", fontsize=14, weight="bold")

    # Overlay Marchenko–Pastur bulk
    if show_mp and sample_size is not None:
        p = cov.shape[0]
        q = p / sample_size
        if q < 1:
            lambda_plus = (1 + np.sqrt(q)) ** 2
            plt.axhline(y=lambda_plus, color="red", linestyle="--", linewidth=2)
            plt.text(
                len(eigvals) * 0.7,
                lambda_plus * 1.02,
                "MP upper bound",
                color="red",
            )

    # Style
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()


def plot_matrix_structure(
    mat: np.ndarray,
    cmap: str = "viridis",
    title: str = "Matrix Structure",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Visualize the structure of a covariance or correlation matrix.

    Parameters
    ----------
    mat : ndarray of shape (p, p)
        Covariance or correlation matrix.
    cmap : str, default='viridis'
        Colormap for the heatmap.
    title : str, default='Matrix Structure'
        Title for the plot.
    vmin, vmax : float, optional
        Color scale limits. Defaults to min/max of the matrix.
    """
    plt.figure(figsize=(6, 6))
    im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=14, weight="bold")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # Style
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()
