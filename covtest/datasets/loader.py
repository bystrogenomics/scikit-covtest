from pathlib import Path
import io

import numpy as np


def load_mnist(
    split: str = "train", return_X_y: bool = True, normalize: bool = True
):
    """
    Load the MNIST dataset from an .npz file stored in the package.

    Parameters
    ----------
    split : {"train", "test"}, default="train"
        Which split to return.
    return_X_y : bool, default=True
        If True, return (X, y) arrays.
        If False, return a dict with keys {"X_train", "y_train", "X_test", "y_test"}.
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
    """
    data_path = Path(__file__).parent / "data/mnist.npz"
    with data_path.open("rb") as f:
        buffer = io.BytesIO(f.read())
    data = np.load(buffer, allow_pickle=False)

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

    # Flatten to (n_samples, 784) if images are 28x28
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    return X, y


def load_tcga(return_X_y: bool = True, return_names: bool = False):
    data_path = Path(__file__).parent / "data/tcga.npz"
    with data_path.open("rb") as f:
        buffer = io.BytesIO(f.read())
    data = np.load(buffer, allow_pickle=True)

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
    Load the Iris dataset from a packaged NPZ file.

    Parameters
    ----------
    return_X_y : bool, default=True
        If True, returns (X, y). Otherwise returns a dictionary.

    return_names : bool, default=False
        If True and return_X_y is True, also returns (feature_names, target_names).

    Returns
    -------
    tuple or dict
        Depending on parameters, returns:
        - (X, y)
        - (X, y, feature_names, target_names)
        - or a dict with all four arrays.
    """
    data_path = Path(__file__).parent / "data/iris_dataset.npz"

    with data_path.open("rb") as f:
        buffer = io.BytesIO(f.read())
    data = np.load(buffer, allow_pickle=True)

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
