"""Tests for the MIND model class and its building blocks."""

from __future__ import annotations

import torch

from MIND.layers import MLP

# ---------------------------------------------------------------------------
# MLP layer
# ---------------------------------------------------------------------------


def test_mlp_forward_shape():
    mlp = MLP(input_dim=10, inter_dims=[16, 8], output_dim=4)
    x = torch.randn(5, 10)
    assert mlp(x).shape == (5, 4)


def test_mlp_handles_nan_inputs():
    mlp = MLP(input_dim=4, inter_dims=[8], output_dim=2)
    x = torch.tensor([[float("nan"), 0.0, 1.0, 2.0]])
    out = mlp(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MIND construction + forward
# ---------------------------------------------------------------------------


def test_mind_construction(toy_model, toy_data):
    n = next(iter(toy_data.values())).shape[0]
    assert toy_model.N == n
    assert toy_model.emb_dim == 4
    # Number of modalities (cls is filtered out in the fixture).
    n_mod = sum(1 for k in toy_data if k != "cls")
    assert len(toy_model.encoder_list) == n_mod
    assert len(toy_model.decoder_list) == n_mod


def test_get_embedding_full_shape(toy_model):
    z_mean, z_log_std = toy_model.get_embedding()
    assert z_mean.shape == (toy_model.N, toy_model.emb_dim)
    assert z_log_std.shape == (toy_model.N, toy_model.emb_dim)
    assert torch.isfinite(z_mean).all()


def test_get_embedding_indexed_shape(toy_model):
    idx = torch.tensor([0, 1, 2, 3])
    z = toy_model.get_embedding(idx_list=idx)
    assert z.shape == (4, toy_model.emb_dim)


def test_predict_returns_one_per_modality(toy_model):
    recons = toy_model.predict()
    assert len(recons) == len(toy_model.data_list)
    for r, dat in zip(recons, toy_model.data_list, strict=True):
        assert r.shape == dat.shape


# ---------------------------------------------------------------------------
# Loss + gradients
# ---------------------------------------------------------------------------


def test_loss_is_finite(toy_model):
    idx = torch.arange(toy_model.N)
    # Pre-compute affinities (full-batch path is what `loss` expects when
    # idx_list covers all patients).
    from MIND._train import _precompute_data_similarities

    _precompute_data_similarities(toy_model)
    loss = toy_model.loss(idx)
    assert torch.isfinite(loss)


def test_backward_no_nan_grads(toy_model):
    from MIND._train import _precompute_data_similarities

    _precompute_data_similarities(toy_model)
    idx = torch.arange(toy_model.N)
    loss = toy_model.loss(idx)
    loss.backward()
    for name, p in toy_model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient on {name}"


# ---------------------------------------------------------------------------
# Validation: short training reduces loss
# ---------------------------------------------------------------------------


def test_short_training_reduces_loss(toy_model):
    """A few epochs of training should reduce the loss on the toy data."""
    from MIND._train import _precompute_data_similarities

    _precompute_data_similarities(toy_model)
    idx = torch.arange(toy_model.N)
    initial_loss = toy_model.loss(idx).item()

    optimizer = torch.optim.Adam(toy_model.parameters(), lr=1e-3)
    for _ in range(15):
        optimizer.zero_grad()
        toy_model.loss(idx).backward()
        optimizer.step()
    final_loss = toy_model.loss(idx).item()
    assert final_loss < initial_loss, (initial_loss, final_loss)


def test_my_train_runs_end_to_end(toy_model):
    """Smoke test: full-batch ``my_train`` completes and embeddings stay finite."""
    toy_model.my_train(n_epoch=3, lr=1e-3, verbose=False)
    z_mean, _ = toy_model.get_embedding()
    assert torch.isfinite(z_mean).all()
