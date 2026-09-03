"""Shared pytest fixtures.

The toy-dataset fixture is sized to keep the full test suite under ~30 s
CPU-only. Bumping ``n_features`` will dramatically slow training because
the per-modality MLPs scale with feature count.
"""

from __future__ import annotations

import pytest
import torch

from MIND import MIND
from MIND.data import make_toy_dataset


@pytest.fixture(scope="session")
def toy_data():
    """Small mixed-modality dataset used across tests."""
    return make_toy_dataset(
        n=60, n_modalities=3, n_features=10, n_clusters=3,
        missing_rate=0.15, seed=31415,
    )


@pytest.fixture()
def toy_model(toy_data):
    """Untrained MIND model on the toy dataset."""
    torch.manual_seed(0)
    return MIND(
        data_dict={k: v for k, v in toy_data.items() if k != "cls"},
        emb_dim=4,
        device="cpu",
    )
