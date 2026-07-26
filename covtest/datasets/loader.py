"""
covtest/datasets/loader.py
==========================

Dataset loaders for bundled and remotely-cached datasets.

Iris (7 KB) is shipped inside the package.

MNIST (~19 MB) and TCGA (~74 MB) are downloaded on first use via ``pooch``
and cached in the user's OS cache directory (``~/.cache/scikit-covtest`` on
Linux/macOS, ``%LOCALAPPDATA%/scikit-covtest/Cache`` on Windows).

To enable remote datasets install the optional dependency::

    pip install scikit-covtest[datasets]   # adds pooch
    # or
    pip install pooch

Files are hosted on Zenodo (DOI: 10.5281/zenodo.21600332):
https://doi.org/10.5281/zenodo.21600332
"""

import importlib.util
import io
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Remote-fetch configuration
# ---------------------------------------------------------------------------

#: Base URL for the Zenodo record hosting the large dataset files.
_DATA_RELEASE_URL = "doi:10.5281/zenodo.21600332/"

#: SHA-256 hashes for integrity verification.
#: Computed from the canonical files at the time of the v0.0.1 release.
_REGISTRY = {
    "mnist.npz": (
        "sha256:"
        "98721cc021dc87a22b9a6c500ff19a4b0f6e3e7bf129043f5a9cd3d5c0084523"
    ),
    "tcga.npz": (
        "sha256:"
        "1ac1ff13dd9a83f5e304101721a42307b31ce59ffb4349f3164bd7e9bcb13e57"
    ),
}

#: Datasets that are still bundled inside the wheel (small enough to include).
_BUNDLED = {"iris_dataset.npz"}

_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_pooch():
    """Return the pooch module, or raise a helpful ImportError."""
    if importlib.util.find_spec("pooch") is None:
        raise ImportError(
            "Loading MNIST and TCGA requires the 'pooch' package, which is "
            "not installed by default to keep the core install lightweight.\n\n"
            "Install it with one of:\n"
            "    pip install scikit-covtest[datasets]\n"
            "    pip install pooch\n"
        )
    import pooch  # noqa: PLC0415

    return pooch


def _get_fetcher():
    """Build and return a configured pooch.Pooch instance."""
    pooch = _require_pooch()
    return pooch.create(
        path=pooch.os_cache("scikit-covtest"),
        base_url=_DATA_RELEASE_URL,
        registry=_REGISTRY,
    )


def _load_npz(filename: str) -> np.lib.npyio.NpzFile:
    """
    Load *filename* from the bundled data directory or from the pooch cache.

    Parameters
    ----------
    filename : str
        Bare filename, e.g. ``"mnist.npz"``.

    Returns
    -------
    np.lib.npyio.NpzFile
        An open NpzFile object.  Callers are responsible for indexing into it.
    """
    bundled_path = _DATA_DIR / filename

    if bundled_path.exists():
        # File is present on disk (bundled wheel, or developer install with
        # the raw repo checkout that still has the large files).
        with bundled_path.open("rb") as fh:
            return np.load(io.BytesIO(fh.read()), allow_pickle=True)

    # Not bundled – fetch via pooch (downloads once, then serves from cache).
    fetcher = _get_fetcher()
    local_path = fetcher.fetch(filename)
    return np.load(local_path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_mnist(
    split: str = "train", return_X_y: bool = True, normalize: bool = True
):
    """
    Load the MNIST dataset, fetching it from the internet on first use.

    The file is cached in the OS user-cache directory after the first
    download.  Subsequent calls are instant.

    Parameters
    ----------
    split : {"train", "test"}, default="train"
        Which split to return.
    return_X_y : bool, default=True
        If True, return (X, y) arrays.
        If False, return a dict with keys {"X_train", "y_train",
        "X_test", "y_test"}.
    normalize : bool, default=True
        If True, scale pixel values to [0, 1].

    Returns
    -------
    (X, y) : tuple of ndarray
        - X has shape (n_samples, 784)
        - y has shape (n_samples,)
    OR
    data : dict
        With arrays for both train and test splits.

    Notes
    -----
    Requires ``pooch`` when the file is not already present locally::

        pip install scikit-covtest[datasets]
    """
    data = _load_npz("mnist.npz")

    if not return_X_y:
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
        }

    if split == "train":
        X, y = data["X_train"], data["y_train"]
    elif split == "test":
        X, y = data["X_test"], data["y_test"]
    else:
        raise ValueError("split must be 'train' or 'test'")

    if normalize:
        X = X.astype(np.float32) / 255.0

    # Flatten to (n_samples, 784) if images are stored as 28x28
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    return X, y


def load_tcga(return_X_y: bool = True, return_names: bool = False):
    """
    Load the TCGA pan-cancer dataset, fetching it from the internet on first
    use.

    The file is cached in the OS user-cache directory after the first
    download.  Subsequent calls are instant.

    Parameters
    ----------
    return_X_y : bool, default=True
        If True, return (X, y). Otherwise return a dict with all arrays.
    return_names : bool, default=False
        If True *and* return_X_y is True, also return
        (X, y, feature_names, label_names, sample_ids).

    Returns
    -------
    (X, y) : tuple of ndarray
        Gene-expression matrix and cancer-type integer labels.
    OR
    (X, y, feature_names, label_names, sample_ids) : tuple
        Returned when return_names=True.
    OR
    data : dict
        Returned when return_X_y=False.

    Notes
    -----
    Requires ``pooch`` when the file is not already present locally::

        pip install scikit-covtest[datasets]
    """
    data = _load_npz("tcga.npz")

    if not return_X_y:
        return {
            "X": data["X"],
            "y": data["y"],
            "feature_names": data["feature_names"],
            "label_names": data["label_names"],
            "sample_ids": data["sample_ids"],
        }

    if return_names:
        return (
            data["X"],
            data["y"],
            data["feature_names"],
            data["label_names"],
            data["sample_ids"],
        )

    return data["X"], data["y"]


def load_iris(return_X_y: bool = True, return_names: bool = False):
    """
    Load the Iris dataset from the bundled NPZ file.

    Parameters
    ----------
    return_X_y : bool, default=True
        If True, returns (X, y). Otherwise returns a dictionary.
    return_names : bool, default=False
        If True and return_X_y is True, also returns
        (X, y, feature_names, target_names).

    Returns
    -------
    tuple or dict
        Depending on parameters, returns:
        - (X, y)
        - (X, y, feature_names, target_names)
        - or a dict with all four arrays.
    """
    data = _load_npz("iris_dataset.npz")

    if not return_X_y:
        return {
            "X": data["X"],
            "y": data["y"],
            "feature_names": data["feature_names"],
            "target_names": data["target_names"],
        }

    if return_names:
        return (
            data["X"],
            data["y"],
            data["feature_names"],
            data["target_names"],
        )

    return data["X"], data["y"]
