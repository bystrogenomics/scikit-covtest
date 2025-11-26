# tests/test_loader.py
import numpy as np
import pytest

from covtest.datasets.loader import load_iris, load_mnist, load_tcga


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


def test_load_mnist_train_shape():
    X, y = load_mnist(split="train", return_X_y=True, normalize=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (60000, 784)
    assert y.shape == (60000,)
    # Normalization check
    assert np.all(X >= 0.0) and np.all(X <= 1.0)


def test_load_mnist_test_shape():
    X, y = load_mnist(split="test", return_X_y=True, normalize=False)
    assert X.shape == (10000, 784)
    assert y.shape == (10000,)
    # Pixel values should be in [0, 255] before normalization
    assert X.max() <= 255
    assert X.min() >= 0


def test_load_mnist_dict_mode():
    data = load_mnist(return_X_y=False, normalize=False)
    assert set(data.keys()) == {"X_train", "y_train", "X_test", "y_test"}
    assert data["X_train"].shape == (60000, 784)
    assert data["y_train"].shape == (60000,)
    assert data["X_test"].shape == (10000, 784)
    assert data["y_test"].shape == (10000,)


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        load_mnist(split="validation")


# -----------------------------
# New tests for load_tcga
# -----------------------------
def test_load_tcga_basic():
    X, y = load_tcga()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]  # should now match 801 samples


def test_load_tcga_with_names():
    X, y, feat_names, label_names, sample_ids = load_tcga(return_names=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(feat_names, np.ndarray)
    assert isinstance(label_names, np.ndarray)
    assert isinstance(sample_ids, np.ndarray)
    assert feat_names.shape[0] == X.shape[1]
    assert y.shape[0] == sample_ids.shape[0]


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
