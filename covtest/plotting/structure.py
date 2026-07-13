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
        Whether to overlay Marchenko-Pastur upper bound (requires sample_size).
    sample_size : int, optional
        Number of samples used to estimate the covariance.
        Needed if show_mp=True to compute the MP bulk.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    The Marchenko-Pastur law describes the limiting distribution of
    eigenvalues for sample covariance matrices from white noise. The upper
    bound is given by (1 + sqrt(p/n))^2 when the population covariance is
    the identity matrix.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Generate a random covariance matrix
    >>> p = 50
    >>> X = np.random.randn(100, p)
    >>> cov = np.cov(X, rowvar=False)
    >>> plot_eigen_scree(cov, n_eigs=20, show_mp=True, sample_size=100)
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
                len(eigvals) * 0.7, lambda_plus * 1.02, "MP upper bound", color="red",
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

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    This function creates a heatmap visualization of the matrix, useful for
    identifying block structure, sparsity patterns, or correlation patterns.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a block diagonal correlation matrix
    >>> p = 20
    >>> block1 = np.ones((10, 10)) * 0.7 + np.eye(10) * 0.3
    >>> block2 = np.ones((10, 10)) * 0.5 + np.eye(10) * 0.5
    >>> mat = np.block([[block1, np.zeros((10, 10))],
    ...                 [np.zeros((10, 10)), block2]])
    >>> plot_matrix_structure(mat, title='Block Diagonal Structure')
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
