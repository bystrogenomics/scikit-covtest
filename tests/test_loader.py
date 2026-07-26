# tests/test_loader.py
#
# Test strategy for remote datasets (mnist, tcga):
#
#   * While the repo still contains the large .npz files (developer checkout),
#     _load_npz() finds them via the bundled-path branch and ALL tests run
#     without pooch installed.
#
#   * Once the files are removed from the repo (post PyPI release), the tests
#     that exercise mnist/tcga are skipped when pooch is absent, and download
#     + cache the files when pooch IS present (e.g. `pip install -e ".[dev]"`).
#
# This means CI passes in both configurations.

import numpy as np
import pytest

from covtest.datasets.loader import load_iris, load_mnist, load_tcga

# ---------------------------------------------------------------------------
# Iris – always bundled, no pooch needed
# ---------------------------------------------------------------------------


def test_load_iris_X_y_shape():
    X, y = load_iris(return_X_y=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_load_iris_with_names():
    X, y, feature_names, target_names = load_iris(
        return_X_y=True, return_names=True
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(feature_names, np.ndarray) or isinstance(
        feature_names, list
    )
    assert isinstance(target_names, np.ndarray) or isinstance(
        target_names, list
    )
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(feature_names) == 4
    assert len(target_names) == 3


def test_load_iris_dict_mode():
    data = load_iris(return_X_y=False)
    expected_keys = {"X", "y", "feature_names", "target_names"}
    assert set(data.keys()) == expected_keys
    assert data["X"].shape == (150, 4)
    assert data["y"].shape == (150,)
    assert len(data["feature_names"]) == 4
    assert len(data["target_names"]) == 3


# ---------------------------------------------------------------------------
# MNIST – bundled locally; fetched via pooch otherwise.
# ---------------------------------------------------------------------------

# Check whether the file is present on disk (developer checkout)
_MNIST_BUNDLED = (
    __import__("pathlib").Path(__file__).parent.parent
    / "covtest/datasets/data/mnist.npz"
).exists()

# Mark: skip if file is absent AND pooch is not installed
_mnist_available = pytest.mark.skipif(
    not _MNIST_BUNDLED
    and __import__("importlib").util.find_spec("pooch") is None,
    reason="mnist.npz not bundled and pooch not installed; "
    "install via `pip install scikit-covtest[datasets]`",
)


@_mnist_available
def test_load_mnist_train_shape():
    X, y = load_mnist(split="train", return_X_y=True, normalize=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (60000, 784)
    assert y.shape == (60000,)
    # Normalization check
    assert np.all(X >= 0.0) and np.all(X <= 1.0)


@_mnist_available
def test_load_mnist_test_shape():
    X, y = load_mnist(split="test", return_X_y=True, normalize=False)
    assert X.shape == (10000, 784)
    assert y.shape == (10000,)
    assert X.max() <= 255
    assert X.min() >= 0


@_mnist_available
def test_load_mnist_dict_mode():
    data = load_mnist(return_X_y=False, normalize=False)
    assert set(data.keys()) == {"X_train", "y_train", "X_test", "y_test"}
    assert data["X_train"].shape == (60000, 784)
    assert data["y_train"].shape == (60000,)
    assert data["X_test"].shape == (10000, 784)
    assert data["y_test"].shape == (10000,)


@_mnist_available
def test_invalid_split_raises():
    with pytest.raises(ValueError):
        load_mnist(split="validation")


def test_load_mnist_no_pooch_raises_helpful_error(monkeypatch, tmp_path):
    """When the file is absent and pooch is not installed, get a clear error."""
    import importlib
    import sys

    import covtest.datasets.loader as loader_mod

    # Pretend the bundled file doesn't exist
    monkeypatch.setattr(
        loader_mod, "_DATA_DIR", tmp_path  # empty dir → file not found
    )
    # Hide pooch from the import system
    monkeypatch.setitem(sys.modules, "pooch", None)
    # Invalidate the find_spec cache
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="pooch"):
        load_mnist()


# ---------------------------------------------------------------------------
# TCGA – bundled locally; fetched via pooch otherwise.
# ---------------------------------------------------------------------------

_TCGA_BUNDLED = (
    __import__("pathlib").Path(__file__).parent.parent
    / "covtest/datasets/data/tcga.npz"
).exists()

_tcga_available = pytest.mark.skipif(
    not _TCGA_BUNDLED
    and __import__("importlib").util.find_spec("pooch") is None,
    reason="tcga.npz not bundled and pooch not installed; "
    "install via `pip install scikit-covtest[datasets]`",
)


@_tcga_available
def test_load_tcga_basic():
    X, y = load_tcga()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]


@_tcga_available
def test_load_tcga_with_names():
    X, y, feat_names, label_names, sample_ids = load_tcga(return_names=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(feat_names, np.ndarray)
    assert isinstance(label_names, np.ndarray)
    assert isinstance(sample_ids, np.ndarray)
    assert feat_names.shape[0] == X.shape[1]
    assert y.shape[0] == sample_ids.shape[0]


@_tcga_available
def test_load_tcga_dict_mode():
    data = load_tcga(return_X_y=False)
    assert set(data.keys()) == {
        "X",
        "y",
        "feature_names",
        "label_names",
        "sample_ids",
    }
    assert data["X"].shape[0] == data["y"].shape[0]
    assert data["feature_names"].shape[0] == data["X"].shape[1]
    assert data["sample_ids"].shape[0] == data["y"].shape[0]
