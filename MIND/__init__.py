"""MIND: Multimodal Integration with Neighbourhood-aware Distributions.

Public API (preserved from the 0.0.1 release):

    from MIND import MIND
    from MIND.data import (
        get_synthetic, load_synthetic,
        get_TCGA, load_TCGA,
        get_CCMA, load_CCMA,
        get_CCLE, load_CCLE,
    )

New in 0.1.0:

    from MIND.data import make_toy_dataset   # network-free synthetic data
    from MIND import MLP                     # was internal, now exported
"""

from .data import (
    download_all,
    get_CCLE,
    get_CCMA,
    get_synthetic,
    get_TCGA,
    load_CCLE,
    load_CCMA,
    load_synthetic,
    load_TCGA,
    make_toy_dataset,
)
from .layers import MLP
from .model import MIND

__version__ = "0.1.0"

__all__ = [
    "MIND",
    "MLP",
    "__version__",
    "download_all",
    "get_CCLE",
    "get_CCMA",
    "get_TCGA",
    "get_synthetic",
    "load_CCLE",
    "load_CCMA",
    "load_TCGA",
    "load_synthetic",
    "make_toy_dataset",
]
