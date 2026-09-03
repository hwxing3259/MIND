"""Neural-network building blocks used by :class:`MIND.MIND`.

The MIND model is built from a stack of identical MLPs (one encoder and one
decoder per modality). Keeping these helpers in their own module makes it
straightforward to test or swap them out without touching the model class.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def block(in_c: int, out_c: int) -> list[nn.Module]:
    """Return a ``[Linear, SiLU]`` block as a list of layers.

    Parameters
    ----------
    in_c : int
        Number of input features to the linear layer.
    out_c : int
        Number of output features.

    Returns
    -------
    list of nn.Module
        ``[nn.Linear(in_c, out_c), nn.SiLU()]``. A list (rather than a
        ``Sequential``) is returned so callers can splat-concatenate layers
        when assembling deeper stacks.
    """
    return [nn.Linear(in_c, out_c), nn.SiLU()]


class MLP(nn.Module):
    """A small fully-connected network with SiLU activations.

    The architecture is::

        Linear(input_dim, inter_dims[0]) -> SiLU
        Linear(inter_dims[0], inter_dims[1]) -> SiLU
        ...
        Linear(inter_dims[-1], output_dim)

    The forward pass replaces NaN inputs with 0 before applying the network,
    so encoders can be called on rows whose features include placeholders for
    "missing" entries without producing NaNs in the embeddings.

    Parameters
    ----------
    input_dim : int, default=784
        Dimensionality of the input vector.
    inter_dims : sequence of int, default=(500, 500, 300)
        Hidden-layer widths.
    output_dim : int, default=200
        Dimensionality of the output vector.

    Examples
    --------
    >>> import torch
    >>> mlp = MLP(input_dim=10, inter_dims=[16, 8], output_dim=4)
    >>> x = torch.randn(3, 10)
    >>> mlp(x).shape
    torch.Size([3, 4])
    """

    def __init__(
        self,
        input_dim: int = 784,
        inter_dims: Sequence[int] = (500, 500, 300),
        output_dim: int = 200,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        inter_dims = list(inter_dims)
        layers: list[nn.Module] = [*block(input_dim, inter_dims[0])]
        for i in range(len(inter_dims) - 1):
            layers += [*block(inter_dims[i], inter_dims[i + 1])]
        layers += [nn.Linear(inter_dims[-1], output_dim)]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        NaN entries in ``x`` are replaced with 0 before the linear stack.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, output_dim)``.
        """
        x = torch.nan_to_num(x, 0.0)
        return self.encoder(x)
