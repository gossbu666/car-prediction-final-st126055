import numpy as np
import pytest

from model import LogisticRegression  # make sure class name/file match your A3/model.py

# --- Reproducible tiny dataset helper ---
def _make_data(n=100, d=10, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # create nontrivial labels via a random linear map
    W = rng.normal(size=(d, k))
    y = (X @ W).argmax(axis=1)
    return X, y


@pytest.fixture(scope="module")
def trained_model():
    """Train once and reuse across tests."""
    X, y = _make_data(n=120, d=10, k=4, seed=42)
    m = LogisticRegression(learning_rate=0.1, n_iterations=60, lmbda=0.01, batch_size=32)
    m.fit(X, y)
    return m


def test_model_accepts_expected_input(trained_model):
    """
    (1) The model takes the expected input: correct feature count -> no error.
    """
    rng = np.random.default_rng(1)
    X_ok = rng.normal(size=(5, 10))  # 10 features matches training
    # If this raises, pytest will fail the test automatically
    _ = trained_model.predict(X_ok)


def test_output_shape_is_1d_len_n(trained_model):
    """
    (2) The model's output has the expected shape: (n_samples,).
    """
    rng = np.random.default_rng(2)
    n_samples = 15
    X = rng.normal(size=(n_samples, 10))
    y_hat = trained_model.predict(X)
    assert isinstance(y_hat, np.ndarray)
    assert y_hat.shape == (n_samples,)