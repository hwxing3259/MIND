#!/usr/bin/env python
"""Single-script reproduction of the MIND paper's numerical examples.

Usage::

    python reproduce_results.py --dataset synthetic
    python reproduce_results.py --dataset CCMA
    python reproduce_results.py --dataset CCLE
    python reproduce_results.py --dataset TCGA --cancer BRCA

The dataset is downloaded on first run (idempotent — re-running skips the
download). Trained embeddings and clinical metadata are written to
``--output_dir`` (default ``./MIND_outputs/<dataset>/``).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch

from MIND import MIND
from MIND.data import load_CCLE, load_CCMA, load_synthetic, load_TCGA


# Hyperparameters used in the MIND paper. Keep here so the script is the
# single source of truth for "how were the published numbers produced?".
PAPER_HPARAMS: Dict[str, dict] = {
    "synthetic": dict(emb_dim=64, lr=1e-4, n_epoch=5000),
    "CCMA": dict(emb_dim=64, lr=1e-4, n_epoch=5000),
    "CCLE": dict(emb_dim=64, lr=1e-4, n_epoch=5000),
    "TCGA": dict(emb_dim=64, lr=1e-4, n_epoch=5000),
}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset(dataset: str, cancer: str | None) -> Dict[str, pd.DataFrame]:
    if dataset == "synthetic":
        return load_synthetic(mode="high")
    if dataset == "CCMA":
        return load_CCMA(task="all")
    if dataset == "CCLE":
        return load_CCLE(task="all")
    if dataset == "TCGA":
        if cancer is None:
            raise SystemExit("--cancer is required when --dataset=TCGA")
        return load_TCGA(cancer_type=cancer, task="all")
    raise SystemExit(f"Unknown dataset {dataset!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        required=True,
        choices=["synthetic", "CCMA", "CCLE", "TCGA"],
    )
    p.add_argument(
        "--cancer",
        default=None,
        help="TCGA cancer-type code (only used for --dataset=TCGA).",
    )
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output_dir",
        default="./MIND_outputs",
        help="Directory to write embeddings.csv into.",
    )
    p.add_argument(
        "--n_epoch",
        type=int,
        default=None,
        help="Override the paper's number of epochs (useful for quick tests).",
    )
    args = p.parse_args()

    _set_seed(args.seed)
    raw = _load_dataset(args.dataset, args.cancer)

    # Pull off any non-modality entries (e.g. clinical labels) before
    # constructing MIND.
    extras: Dict[str, pd.DataFrame] = {}
    for key in ("clinical", "cls"):
        if key in raw:
            extras[key] = raw.pop(key)

    hp = PAPER_HPARAMS[args.dataset].copy()
    if args.n_epoch is not None:
        hp["n_epoch"] = args.n_epoch

    print(f"Building MIND on {args.dataset} ({len(raw)} modalities)...")
    model = MIND(data_dict=raw, emb_dim=hp["emb_dim"], device=args.device)
    print(f"Training for {hp['n_epoch']} epochs at lr={hp['lr']}...")
    model.my_train(n_epoch=hp["n_epoch"], lr=hp["lr"])

    with torch.no_grad():
        z_mean, _ = model.get_embedding()

    out_subdir = args.dataset if args.cancer is None else f"{args.dataset}_{args.cancer}"
    out_dir = os.path.join(args.output_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(z_mean.cpu().numpy()).to_csv(
        os.path.join(out_dir, "embeddings.csv")
    )
    for name, df in extras.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"))
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
