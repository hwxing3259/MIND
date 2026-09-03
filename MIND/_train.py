"""Training loop for the MIND model.

Lives in its own module so that :class:`MIND.model.MIND` stays focused on
the model definition and the training loop can be unit-tested without a full
forward pass setup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from openTSNE import affinity

if TYPE_CHECKING:  # pragma: no cover - import-only for type checkers
    from .model import MIND


def _precompute_data_similarities(model: MIND) -> None:
    """Pre-compute per-modality NxN neighbourhood-affinity matrices.

    Used when training in full-batch mode (the affinity matrix can be
    computed once and reused). Matches the original implementation exactly.
    """
    if isinstance(model.perp, int):
        model.perp = [model.perp] * len(model.input_dim_list)
    if len(model.perp) != len(model.input_dim_list):
        raise ValueError(
            "perp must be an int or a sequence with one entry per modality."
        )
    for i in range(len(model.data_list)):
        nan_idx = ~model.data_list[i][:, 0].isnan()
        indices = nan_idx.nonzero(as_tuple=True)[0]
        temp = torch.zeros(
            (model.data_list[i].shape[0], model.data_list[i].shape[0]),
            device=model.device,
        )
        sim = torch.tensor(
            affinity.PerplexityBasedNN(
                model.data_list[i][nan_idx].cpu().numpy(),
                perplexity=model.perp[i],
                metric="euclidean",
            ).P.todense(),
            dtype=torch.float32,
            device=model.device,
        )
        temp[indices[:, None], indices] += sim
        model.data_similarities += [temp * 1.0]


def train_loop(
    model: MIND,
    n_epoch: int = 2000,
    lr: float = 1e-3,
    batch_size: int | None = None,
    verbose: bool = True,
) -> None:
    """Run the MIND optimisation loop in place.

    Parameters
    ----------
    model : MIND
        The model to train. Modified in place.
    n_epoch : int, default=2000
        Number of optimisation epochs.
    lr : float, default=1e-3
        Adam learning rate.
    batch_size : int, optional
        Mini-batch size. ``None`` means full-batch (``model.N``); the
        per-modality affinity matrices are pre-computed once.
    verbose : bool, default=True
        Print progress every 1000 epochs.
    """
    if batch_size is None:
        batch_size = model.N
        _precompute_data_similarities(model)

    model.data_idx_loader = torch.utils.data.DataLoader(
        torch.tensor(range(model.N), device=model.device),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    for ep in range(n_epoch):
        model.train(True)
        running_loss = 0.0
        for batch_id in model.data_idx_loader:
            optimizer.zero_grad()
            batch_loss = model.loss(batch_id)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.detach().cpu().item()
        if verbose and ep % 1000 == 0:
            print(f"Epoch={ep}")
