"""Input validation helpers for the MIND package.

Centralised here so that error messages are consistent across the public API
and easy to update.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def validate_data_dict(data_dict: Mapping[str, pd.DataFrame]) -> None:
    """Validate the ``data_dict`` argument accepted by :class:`MIND.MIND`.

    Parameters
    ----------
    data_dict : Mapping[str, pandas.DataFrame]
        Mapping ``modality_name -> patient x feature DataFrame``.

    Raises
    ------
    TypeError
        If ``data_dict`` is not a mapping or any value is not a DataFrame.
    ValueError
        If the dictionary is empty, modality DataFrames have inconsistent
        row counts, or any row is partially observed (the package convention
        is that missing patients are encoded as a row of all-NaN).
    """
    if not isinstance(data_dict, Mapping):
        raise TypeError(
            f"data_dict must be a mapping (e.g. dict) of name -> DataFrame, "
            f"got {type(data_dict).__name__}."
        )
    if len(data_dict) == 0:
        raise ValueError("data_dict must contain at least one modality.")

    n_rows: int | None = None
    for name, df in data_dict.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"data_dict['{name}'] must be a pandas DataFrame, "
                f"got {type(df).__name__}."
            )
        if df.shape[1] == 0:
            raise ValueError(
                f"data_dict['{name}'] has zero feature columns."
            )
        if n_rows is None:
            n_rows = df.shape[0]
        elif df.shape[0] != n_rows:
            raise ValueError(
                f"All modalities must have the same number of rows. "
                f"data_dict['{name}'] has {df.shape[0]} rows, expected {n_rows}."
            )

        # Convention: a missing patient is encoded as a row of all-NaN.
        # Partially-observed rows are not supported (the model checks
        # column 0 to identify present rows).
        arr = df.to_numpy()
        nan_mask = np.isnan(arr)
        partially_missing = nan_mask.any(axis=1) & ~nan_mask.all(axis=1)
        if partially_missing.any():
            bad = int(partially_missing.sum())
            raise ValueError(
                f"data_dict['{name}'] has {bad} row(s) that are partially "
                f"missing (some NaN entries, others observed). MIND only "
                f"supports the all-NaN-row convention for missing patients. "
                f"Either impute these values or drop them from the modality."
            )


def validate_hyperparameters(
    emb_dim: int,
    alpha: float,
    beta: float,
    perp: int | Sequence[int],
    n_modalities: int,
) -> None:
    """Validate hyperparameters passed to :class:`MIND.MIND`.

    Parameters
    ----------
    emb_dim : int
        Embedding dimensionality. Must be a positive integer.
    alpha : float
        Modality-alignment regularisation strength. Must be non-negative.
    beta : float
        t-SNE tilting factor. Must be non-negative.
    perp : int or sequence of int
        Perplexity, either a single int applied to every modality or one int
        per modality.
    n_modalities : int
        Number of modalities in ``data_dict``.

    Raises
    ------
    ValueError
        If any value violates the documented constraints.
    """
    if not isinstance(emb_dim, int) or emb_dim <= 0:
        raise ValueError(f"emb_dim must be a positive integer, got {emb_dim!r}.")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha!r}.")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta!r}.")

    if isinstance(perp, int):
        if perp <= 0:
            raise ValueError(f"perp must be positive, got {perp!r}.")
    else:
        try:
            perp_list = list(perp)
        except TypeError as exc:
            raise TypeError(
                f"perp must be int or sequence of int, got {type(perp).__name__}."
            ) from exc
        if len(perp_list) != n_modalities:
            raise ValueError(
                f"perp sequence length ({len(perp_list)}) does not match "
                f"the number of modalities ({n_modalities})."
            )
        if any((not isinstance(p, int)) or p <= 0 for p in perp_list):
            raise ValueError(
                f"All entries of perp must be positive integers, got {perp_list!r}."
            )


def validate_device(device: str) -> None:
    """Validate the ``device`` string.

    Parameters
    ----------
    device : str
        Either ``'cpu'`` or a CUDA device specifier like ``'cuda'`` /
        ``'cuda:0'``. We do not check CUDA availability here so that users
        building a model on a CPU machine for later transfer get a clear
        error only at the point of use.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}.")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(
            f"device must be 'cpu' or start with 'cuda', got {device!r}."
        )
