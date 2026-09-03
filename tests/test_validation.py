"""Tests for input validation in ``MIND._validation``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MIND import MIND
from MIND._validation import (
    validate_data_dict,
    validate_device,
    validate_hyperparameters,
)
from MIND.data import make_toy_dataset

# ---------------------------------------------------------------------------
# validate_data_dict
# ---------------------------------------------------------------------------


def test_validate_data_dict_accepts_valid_input(toy_data):
    valid = {k: v for k, v in toy_data.items() if k != "cls"}
    validate_data_dict(valid)


def test_validate_data_dict_rejects_non_mapping():
    with pytest.raises(TypeError):
        validate_data_dict([])


def test_validate_data_dict_rejects_empty_mapping():
    with pytest.raises(ValueError, match="at least one"):
        validate_data_dict({})


def test_validate_data_dict_rejects_non_dataframe_value():
    with pytest.raises(TypeError, match="DataFrame"):
        validate_data_dict({"x": np.zeros((3, 3))})


def test_validate_data_dict_rejects_inconsistent_row_counts():
    a = pd.DataFrame(np.zeros((4, 5)))
    b = pd.DataFrame(np.zeros((3, 5)))
    with pytest.raises(ValueError, match="same number of rows"):
        validate_data_dict({"a": a, "b": b})


def test_validate_data_dict_rejects_partially_missing_rows():
    arr = np.array([[1.0, 2.0, 3.0], [1.0, np.nan, 3.0]])
    df = pd.DataFrame(arr)
    with pytest.raises(ValueError, match="partially"):
        validate_data_dict({"a": df})


def test_validate_data_dict_rejects_zero_feature_modality():
    with pytest.raises(ValueError, match="zero feature"):
        validate_data_dict({"a": pd.DataFrame(np.zeros((3, 0)))})


# ---------------------------------------------------------------------------
# validate_hyperparameters
# ---------------------------------------------------------------------------


def test_validate_hyperparameters_accepts_defaults():
    validate_hyperparameters(emb_dim=8, alpha=0.05, beta=1.0, perp=30, n_modalities=3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"emb_dim": 0},
        {"emb_dim": -1},
        {"alpha": -0.1},
        {"beta": -0.1},
        {"perp": 0},
        {"perp": [10, 10]},  # length mismatch (n_modalities=3 in fixture)
    ],
)
def test_validate_hyperparameters_rejects_bad_inputs(kwargs):
    base = dict(emb_dim=8, alpha=0.05, beta=1.0, perp=30, n_modalities=3)
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        validate_hyperparameters(**base)


# ---------------------------------------------------------------------------
# validate_device
# ---------------------------------------------------------------------------


def test_validate_device_accepts_known_strings():
    validate_device("cpu")
    validate_device("cuda")
    validate_device("cuda:0")


def test_validate_device_rejects_unknown():
    with pytest.raises(ValueError, match="cpu"):
        validate_device("mps")
    with pytest.raises(TypeError):
        validate_device(0)


# ---------------------------------------------------------------------------
# End-to-end: MIND.__init__ surfaces validation errors
# ---------------------------------------------------------------------------


def test_mind_init_validates_dict():
    with pytest.raises(ValueError):
        MIND(data_dict={}, emb_dim=4)


def test_mind_init_validates_emb_dim():
    data = make_toy_dataset(n=20, n_modalities=2, n_features=4, seed=0)
    with pytest.raises(ValueError):
        MIND(
            data_dict={k: v for k, v in data.items() if k != "cls"},
            emb_dim=0,
        )
