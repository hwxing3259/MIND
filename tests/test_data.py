"""Tests for ``MIND.data.make_toy_dataset`` and basic data plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MIND.data import make_toy_dataset


def test_make_toy_dataset_shape():
    data = make_toy_dataset(n=50, n_modalities=2, n_features=8, seed=0)
    assert set(data.keys()) == {"modality_0", "modality_1", "cls"}
    assert data["modality_0"].shape == (50, 8)
    assert data["modality_1"].shape == (50, 8)
    assert data["cls"].shape == (50, 1)


def test_make_toy_dataset_seed_reproducibility():
    a = make_toy_dataset(n=30, n_modalities=2, seed=42)
    b = make_toy_dataset(n=30, n_modalities=2, seed=42)
    pd.testing.assert_frame_equal(a["modality_0"], b["modality_0"])
    pd.testing.assert_frame_equal(a["cls"], b["cls"])


def test_make_toy_dataset_no_fully_missing_patient():
    """Every patient must be observed in at least one modality."""
    data = make_toy_dataset(n=80, n_modalities=4, missing_rate=0.5, seed=1)
    masks = [df.isna().to_numpy().all(axis=1) for k, df in data.items() if k != "cls"]
    fully_missing = np.all(masks, axis=0)
    assert not fully_missing.any()


def test_make_toy_dataset_missing_rate_respected():
    data = make_toy_dataset(n=200, n_modalities=2, missing_rate=0.25, seed=2)
    # Each modality should have ~25% missing rows.
    for k, df in data.items():
        if k == "cls":
            continue
        frac_missing = df.isna().to_numpy().all(axis=1).mean()
        # Toy dataset corrects for fully-missing patients, so fraction may
        # be slightly less than the requested rate when n_modalities is small.
        assert 0.15 <= frac_missing <= 0.30


def test_make_toy_dataset_validates_inputs():
    with pytest.raises(ValueError):
        make_toy_dataset(n=0)
    with pytest.raises(ValueError):
        make_toy_dataset(n_modalities=0)
    with pytest.raises(ValueError):
        make_toy_dataset(missing_rate=1.0)
