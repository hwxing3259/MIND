"""Main MIND model class.

Implements *Multimodal Integration with Neighbourhood-aware Distributions*
(Xing et al., bioRxiv 2025). MIND is a VAE-style model that learns a single
shared per-patient embedding from several partially-overlapping omics
modalities.

The training loop is delegated to :mod:`MIND._train`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from openTSNE import affinity

from ._train import train_loop
from ._validation import (
    validate_data_dict,
    validate_device,
    validate_hyperparameters,
)
from .layers import MLP


class MIND(nn.Module):
    """Multimodal Integration with Neighbourhood-aware Distributions.

    Trains a per-modality VAE encoder/decoder pair such that the shared
    latent distribution preserves the neighbourhood structure of each
    modality individually. Missing patients in a modality are encoded as
    rows of NaN; the model never queries the corresponding encoder for those
    patients.

    Parameters
    ----------
    data_dict : dict of {str: pandas.DataFrame}
        Mapping ``modality_name -> patients x features DataFrame``. Every
        DataFrame must have the same number of rows; rows for missing
        patients must be all-NaN (the package convention).
    emb_dim : int, default=128
        Dimensionality of the shared latent space.
    device : str, default='cpu'
        Either ``'cpu'`` or a CUDA device specifier (``'cuda'``, ``'cuda:0'``).
    alpha : float, default=0.05
        Regularisation strength for the modality-alignment penalty (the term
        that pulls per-modality embeddings of the same patient together).
    perp : int or sequence of int, default=30
        Perplexity used by openTSNE when computing the per-modality
        neighbourhood affinities. Pass a sequence to use a different value
        per modality.
    beta : float, default=1.0
        Strength of the t-SNE-style neighbourhood-preservation term in the
        ELBO.

    Attributes
    ----------
    encoder_list, decoder_list : nn.ModuleList
        Per-modality encoder/decoder networks (see :class:`MIND.layers.MLP`).
    noise_log_scales : nn.ParameterList
        Per-modality learned log-σ for the Gaussian likelihood.
    presence : list of torch.Tensor
        Boolean masks of shape ``(N,)`` marking which patients are observed
        for each modality.
    data_list : list of torch.Tensor
        Per-modality data, one ``(N, D_m)`` tensor each (NaNs preserved).

    Examples
    --------
    >>> from MIND import MIND
    >>> from MIND.data import make_toy_dataset
    >>> data = make_toy_dataset(n=80, n_modalities=3, seed=0)
    >>> model = MIND(data, emb_dim=8, device='cpu')
    >>> model.my_train(n_epoch=5)  # doctest: +SKIP
    >>> z_mean, z_log_std = model.get_embedding()
    >>> z_mean.shape
    torch.Size([80, 8])
    """

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        emb_dim: int = 128,
        device: str = "cpu",
        alpha: float = 5e-2,
        perp: int | Sequence[int] = 30,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        validate_data_dict(data_dict)
        validate_device(device)
        validate_hyperparameters(emb_dim, alpha, beta, perp, len(data_dict))

        presence = [
            torch.tensor(~_.isna().to_numpy().all(1)).to(device)
            for _ in list(data_dict.values())
        ]
        data_list = [
            torch.tensor(_.to_numpy(), dtype=torch.float32).to(device)
            for _ in list(data_dict.values())
        ]

        self.input_dim_list: list[int] = [_.shape[1] for _ in data_list]
        self.device: str = device
        self.P: float = float(np.mean(self.input_dim_list))
        self.N: int = data_list[0].shape[0]
        self.data_list: list[torch.Tensor] = data_list
        self.presence: list[torch.Tensor] = presence
        self.emb_dim: int = emb_dim
        self.alpha: float = alpha
        self.beta: float = beta
        self.perp: int | Sequence[int] = perp

        self.data_similarities: list[torch.Tensor] = []
        self.data_idx_loader: torch.utils.data.DataLoader | None = None

        self.decoder_list = nn.ModuleList(
            [
                MLP(
                    input_dim=self.emb_dim,
                    inter_dims=[
                        2 * self.emb_dim,
                        4 * self.emb_dim,
                        max(4 * self.emb_dim, self.input_dim_list[i] // 4),
                        max(4 * self.emb_dim, self.input_dim_list[i] // 2),
                    ],
                    output_dim=self.input_dim_list[i],
                )
                for i in range(len(self.data_list))
            ]
        )
        self.encoder_list = nn.ModuleList(
            [
                MLP(
                    input_dim=self.input_dim_list[i],
                    inter_dims=[
                        max(4 * self.emb_dim, self.input_dim_list[i] // 2),
                        max(4 * self.emb_dim, self.input_dim_list[i] // 4),
                        4 * self.emb_dim,
                        2 * self.emb_dim,
                    ],
                    output_dim=2 * self.emb_dim,
                )
                for i in range(len(self.data_list))
            ]
        )

        self.noise_log_scales = nn.ParameterList(
            [torch.zeros((1, _)) for _ in self.input_dim_list]
        )

        self.register_buffer("prior_mean", torch.zeros(self.emb_dim))
        self.register_buffer("prior_std", torch.ones(self.emb_dim))

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def mc_kl_term(
        self,
        emb: torch.Tensor,
        post_mean: torch.Tensor,
        post_log_std: torch.Tensor,
        idx_list: torch.Tensor,
    ) -> torch.Tensor:
        """Combined KL + neighbourhood preservation term.

        Computes the prior KL divergence on the latent posterior plus a
        cross-entropy term between the latent neighbourhood distribution
        ``Q`` and each modality's data-space neighbourhood distribution ``P``.

        Parameters
        ----------
        emb : torch.Tensor
            ``(N, emb_dim)`` shared embeddings; only ``emb[idx_list]`` is in
            the autograd graph during a batch update.
        post_mean : torch.Tensor
            ``(len(idx_list), emb_dim)`` posterior mean for the batch.
        post_log_std : torch.Tensor
            ``(len(idx_list), emb_dim)`` posterior log-std for the batch.
        idx_list : torch.Tensor
            1-D tensor of patient indices in the current batch.

        Returns
        -------
        torch.Tensor
            Scalar loss contribution.
        """
        loss_kl_1 = torch.distributions.kl.kl_divergence(
            torch.distributions.normal.Normal(
                loc=post_mean[:, None, :], scale=post_log_std[:, None, :].exp()
            ),
            torch.distributions.normal.Normal(
                loc=self.prior_mean, scale=self.prior_std
            ),
        ).sum()

        affinities = 1.0 / (1.0 + torch.cdist(emb[None], emb[None])[0] ** 2)
        affinities.fill_diagonal_(0.0)

        loss_kl_2: torch.Tensor = torch.tensor(0.0, device=self.device)
        for m in range(len(self.data_list)):
            available_id = self.presence[m][idx_list]
            overlap_m = idx_list[available_id]
            if len(idx_list) != self.N:
                P = torch.tensor(
                    affinity.PerplexityBasedNN(
                        self.data_list[m][overlap_m].cpu().numpy(),
                        perplexity=30,
                        metric="euclidean",
                    ).P.todense(),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                sub_data_similarity = self.data_similarities[m][overlap_m][
                    :, overlap_m
                ]
                P = sub_data_similarity / sub_data_similarity.sum()

            sub_affinity = affinities[available_id][:, available_id]
            Q = sub_affinity / (sub_affinity.sum() + 1e-8)

            loss_kl_2 = loss_kl_2 + -1.0 * self.beta * (P * torch.log(Q + 1e-8)).sum()

        return loss_kl_1 / (len(idx_list) * self.emb_dim) + loss_kl_2

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_embedding(
        self, idx_list: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Compute the shared latent embedding for some or all patients.

        Parameters
        ----------
        idx_list : torch.Tensor, optional
            1-D tensor of patient indices. If ``None``, returns the
            embedding for all patients.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            If ``idx_list is None``: a tuple ``(mean_z, log_std_z)`` of
            shape ``(N, emb_dim)`` each.
            Otherwise: a single sample ``mean_z + eps * exp(log_std_z)``
            of shape ``(len(idx_list), emb_dim)``.
        """
        if idx_list is None:
            emb_store = torch.zeros(
                (len(self.data_list), self.N, 2 * self.emb_dim),
                device=self.device,
            )
            appearance = torch.zeros(self.N, device=self.device)
            for m in range(len(self.data_list)):
                idx = torch.tensor(range(self.N), device=self.device)
                available_id = self.presence[m][idx]
                appearance[available_id] += 1.0
                overlap_m = idx[available_id]
                emb_store[m, available_id, :] = self.encoder_list[m](
                    self.data_list[m][overlap_m]
                )
            merged_z = emb_store.sum(0) / appearance[:, None]
            return merged_z[:, : self.emb_dim], merged_z[:, self.emb_dim :]

        emb_store = torch.zeros(
            (len(self.data_list), len(idx_list), 2 * self.emb_dim),
            device=self.device,
        )
        appearance = torch.zeros(len(idx_list), device=self.device)
        for m in range(len(self.data_list)):
            available_id = self.presence[m][idx_list]
            appearance[available_id] += 1.0
            overlap_m = idx_list[available_id]
            emb_store[m, available_id, :] = self.encoder_list[m](
                self.data_list[m][overlap_m]
            )
        merged_z = emb_store.sum(0) / appearance[:, None]
        mean_z, log_std_z = (
            merged_z[:, : self.emb_dim],
            merged_z[:, self.emb_dim :],
        )
        return mean_z + torch.randn_like(log_std_z) * log_std_z.exp()

    def predict(self) -> list[torch.Tensor]:
        """Reconstruct every modality from the posterior-mean embedding.

        Returns
        -------
        list of torch.Tensor
            One reconstruction per modality, shape matching the input data.
            Predictions are filled in for previously-missing rows as well.
        """
        reconstructed = [torch.zeros_like(_) * float("nan") for _ in self.data_list]
        with torch.no_grad():
            mean_z, _log_std_z = self.get_embedding()
            for m in range(len(self.data_list)):
                reconstructed[m] = self.decoder_list[m](mean_z)
        return reconstructed

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def loss(self, idx_list: torch.Tensor) -> torch.Tensor:
        """Compute the per-batch training loss.

        Parameters
        ----------
        idx_list : torch.Tensor
            1-D long tensor of patient indices in the current batch.

        Returns
        -------
        torch.Tensor
            Scalar loss = KL term + reconstruction NLL + α · alignment penalty.
        """
        emb_store = torch.zeros(
            (len(self.data_list), len(idx_list), 2 * self.emb_dim),
            device=self.device,
        )
        appearance = torch.zeros(len(idx_list), device=self.device)
        for m in range(len(self.data_list)):
            available_id = self.presence[m][idx_list]
            appearance[available_id] += 1.0
            overlap_m = idx_list[available_id]
            emb_store[m, available_id, :] = self.encoder_list[m](
                self.data_list[m][overlap_m]
            )
        merged_z = emb_store.sum(0) / appearance[:, None]
        mean_z, log_std_z = (
            merged_z[:, : self.emb_dim],
            merged_z[:, self.emb_dim :],
        )

        # Modality-alignment penalty: pull per-modality embeddings of the
        # same patient together (only counted for patients present in both).
        A_ = emb_store.permute(1, 0, 2)  # N x M x P
        with torch.no_grad():
            is_zero = (A_ == 0).all(dim=2)
            B = ~(is_zero.unsqueeze(2) | is_zero.unsqueeze(1)) * 1.0
            B.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        dist_penalty = ((torch.cdist(A_, A_) * B) ** 2).sum() / B.sum()

        sample_z = mean_z + torch.randn_like(log_std_z) * log_std_z.exp()

        loss_kl = self.mc_kl_term(sample_z, mean_z, log_std_z, idx_list)

        loss_recon: torch.Tensor = torch.tensor(0.0, device=self.device)
        for m in range(len(self.data_list)):
            available_id = self.presence[m][idx_list]
            m_recon = self.decoder_list[m](sample_z[available_id])
            overlap_m = idx_list[self.presence[m][idx_list]]
            nan_mask = ~torch.isnan(self.data_list[m][overlap_m])
            neg_gaussian_lkd = (
                self.noise_log_scales[m]
                + 0.5
                * (
                    (self.data_list[m][overlap_m] - m_recon)
                    / self.noise_log_scales[m].exp()
                )
                ** 2
            )
            loss_recon = loss_recon + neg_gaussian_lkd[nan_mask].mean() * len(
                idx_list
            ) + 1e-1 * (self.noise_log_scales[m] ** 2).mean()

        return (
            loss_kl
            + loss_recon / len(idx_list)
            + self.alpha * dist_penalty / self.emb_dim
        )

    def my_train(
        self,
        n_epoch: int = 2000,
        lr: float = 1e-3,
        batch_size: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Train the model in place.

        Parameters
        ----------
        n_epoch : int, default=2000
            Number of training epochs.
        lr : float, default=1e-3
            Adam learning rate.
        batch_size : int, optional
            Mini-batch size. If ``None``, full-batch training is used (this is
            what the paper does); per-modality affinity matrices are then
            pre-computed once for efficiency.
        verbose : bool, default=True
            If True, print progress every 1000 epochs.

        Notes
        -----
        Uses :func:`MIND._train.train_loop` internally so the loop can be
        unit-tested in isolation.
        """
        train_loop(
            model=self,
            n_epoch=n_epoch,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
        )
