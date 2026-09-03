# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-04

First semver release. Addresses the SMARTbiomed/software-review checklist
(Issue #2). All public API entry points from 0.0.1 are preserved.

### Added

- `MIND.data.make_toy_dataset` — pure-numpy synthetic generator that
  requires no network access; used in tests and the new quickstart tutorial.
- `examples/reproduce_results.py` — single-script reproduction CLI
  (`python reproduce_results.py --dataset {synthetic,CCMA,CCLE,TCGA}`).
- `examples/quickstart_toy.ipynb` — fast tutorial notebook driven by
  `make_toy_dataset`, geared at users applying MIND to their own data.
- `MIND._validation` — input-validation helpers; surfaced via `MIND.MIND.__init__`.
- Full `pytest` suite covering the model, data loaders, and validation.
- `.github/workflows/ci.yml` — automated CI on Linux + macOS, Python 3.10–3.12.
- `docs/` — Sphinx skeleton (`autodoc` + `napoleon`) plus a written user guide
  (`docs/user_guide.md`) covering input format and hyperparameter tuning.
- `.readthedocs.yaml`, `pyproject.toml`, `CITATION.cff`, `AUTHORS.md`,
  `CHANGELOG.md`.
- `download_if_missing` flag on every `load_*` helper — allows fail-fast
  behaviour for users who don't want network calls.

### Changed

- Package layout: `MIND_model.py` → `MIND/model.py` + `MIND/layers.py` +
  `MIND/_train.py`. `MIND_data.py` → `MIND/data.py`. The public re-exports
  in `MIND/__init__.py` are unchanged, so `from MIND import MIND` and the
  existing `from MIND import get_*, load_*` imports keep working.
- Type hints added to every public function, method and attribute.
- NumPy-style docstrings written for every public symbol.
- `MIND.data.get_data` now uses `os.path.join` (was string concatenation
  that assumed a trailing `/`).
- README rewritten: jargon-free summary, fixed typos ("Sofrware",
  "User needs ot provide"), added "Citing" section, comparison-to-alternatives
  table, and a network-free quickstart.

### Fixed

- Stale `device = 'cuda' if torch.cuda.is_available() else 'cpu'` line in
  `MIND_model.py` that ran before `import torch`.
- Toy-dataset construction no longer leaves any patient missing from every
  modality (previously caused a divide-by-zero in the `appearance` denominator).

## [0.0.1]

Initial release accompanying the bioRxiv preprint.
