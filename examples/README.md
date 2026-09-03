# Examples

A tiered index of the example notebooks and scripts shipped with MIND.

## Tier 1 — start here

| File | Runtime | What it shows |
|---|---|---|
| [`quickstart_toy.ipynb`](quickstart_toy.ipynb) | <1 min on CPU, no download | The minimal end-to-end loop using `make_toy_dataset`. The recommended entry point for users applying MIND to their own data. |

## Tier 2 — single-script reproduction

| File | What it shows |
|---|---|
| [`reproduce_results.py`](reproduce_results.py) | One CLI that downloads the published datasets and reruns each paper experiment. Try `python reproduce_results.py --dataset synthetic`. |

## Tier 3 — extended notebook tutorials (paper datasets)

These reproduce the four numerical examples in the MIND paper. Each
downloads ~hundreds of MB and trains for thousands of epochs.

| File | Dataset | Notes |
|---|---|---|
| [`synthetic_example.ipynb`](synthetic_example.ipynb) | Paper synthetic | Best illustration of the neighbourhood-preservation idea. |
| [`CCMA_example.ipynb`](CCMA_example.ipynb) | CCMA (cancer-cell DNA-methylation atlas) | Multi-class cancer-type classification probe. |
| [`CCLE_example.ipynb`](CCLE_example.ipynb) | CCLE | Six modalities; uses `tcga_code` as the label. |
| [`TCGA_int_example.ipynb`](TCGA_int_example.ipynb) | TCGA per-cancer | Five modalities; survival benchmark. |
| [`MIND_ablation_studies_CCLE.ipynb`](MIND_ablation_studies_CCLE.ipynb) | CCLE | Ablation studies referenced in the paper. |

## Schematic figure

The package's pipeline diagram lives at
[`multiomics_integration_schematic.png`](multiomics_integration_schematic.png).
It is also embedded in the top-level README.
