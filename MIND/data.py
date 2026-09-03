"""Data download / loading helpers for the MIND package.

The package ships with helpers that:

- download the four pre-processed datasets used in the MIND paper from
  Figshare on demand (``get_synthetic``, ``get_TCGA``, ``get_CCMA``,
  ``get_CCLE``);
- load them into a ``{modality_name: pandas.DataFrame}`` dictionary in the
  exact shape :class:`MIND.MIND` expects (``load_synthetic``, ``load_TCGA``,
  ``load_CCMA``, ``load_CCLE``);
- generate a small fully-synthetic toy dataset that requires **no network
  access** (``make_toy_dataset``) — useful for tests, tutorials, and as a
  smoke check that the package is installed correctly.
"""

from __future__ import annotations

import os
import zipfile

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Toy dataset (no network)
# ---------------------------------------------------------------------------


def make_toy_dataset(
    n: int = 200,
    n_modalities: int = 3,
    n_features: int = 50,
    n_clusters: int = 3,
    missing_rate: float = 0.2,
    noise: float = 0.5,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Generate a small fully-synthetic multi-omics dataset.

    Patients are partitioned into ``n_clusters`` latent groups; each
    modality view is generated from the same latent code with its own random
    projection and additive Gaussian noise. A fraction ``missing_rate`` of
    rows in every modality is replaced with all-NaN, mirroring the barcode
    missingness pattern that real omics datasets exhibit.

    Designed to be **fast** (a few hundred patients × a few dozen features)
    and **dependency-free** (pure numpy + pandas).

    Parameters
    ----------
    n : int, default=200
        Number of patients.
    n_modalities : int, default=3
        Number of modalities to generate.
    n_features : int, default=50
        Number of features per modality.
    n_clusters : int, default=3
        Number of latent patient clusters (used as classification labels in
        the returned ``cls`` DataFrame).
    missing_rate : float, default=0.2
        Per-modality fraction of patients whose row is replaced with NaN.
    noise : float, default=0.5
        Standard deviation of the additive Gaussian noise.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Dictionary with keys ``'modality_0', 'modality_1', ...`` and an
        extra ``'cls'`` DataFrame holding the integer cluster label for
        each patient. Each modality DataFrame has shape ``(n, n_features)``;
        ``cls`` has shape ``(n, 1)``.

    Examples
    --------
    >>> data = make_toy_dataset(n=50, n_modalities=2, seed=0)
    >>> sorted(data.keys())
    ['cls', 'modality_0', 'modality_1']
    >>> data['modality_0'].shape
    (50, 50)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}.")
    if n_modalities <= 0:
        raise ValueError(f"n_modalities must be positive, got {n_modalities!r}.")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features!r}.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(
            f"missing_rate must be in [0, 1), got {missing_rate!r}."
        )

    rng = np.random.default_rng(seed)
    cluster_id = rng.integers(0, n_clusters, size=n)
    centers = rng.normal(size=(n_clusters, 8))
    z = centers[cluster_id] + rng.normal(scale=0.1, size=(n, 8))

    patient_index = pd.Index([f"patient_{i:04d}" for i in range(n)], name="patient")

    # Build the per-modality missing-row masks first, then guarantee no
    # patient is missing from every modality (otherwise the model's
    # `appearance` divisor is zero for that patient).
    missing_masks = []
    for _ in range(n_modalities):
        n_missing = int(round(missing_rate * n))
        mask = np.zeros(n, dtype=bool)
        if n_missing > 0:
            mask[rng.choice(n, size=n_missing, replace=False)] = True
        missing_masks.append(mask)
    fully_missing = np.all(missing_masks, axis=0)
    for patient in np.where(fully_missing)[0]:
        # Reveal one modality at random for these patients.
        keep = int(rng.integers(0, n_modalities))
        missing_masks[keep][patient] = False

    out: dict[str, pd.DataFrame] = {}
    for m in range(n_modalities):
        proj = rng.normal(size=(8, n_features))
        x = z @ proj + rng.normal(scale=noise, size=(n, n_features))
        x[missing_masks[m], :] = np.nan
        cols = [f"m{m}_feat_{j:03d}" for j in range(n_features)]
        out[f"modality_{m}"] = pd.DataFrame(x, index=patient_index, columns=cols)

    out["cls"] = pd.DataFrame(
        {"label": cluster_id}, index=patient_index
    )
    return out


# ---------------------------------------------------------------------------
# Figshare download utility
# ---------------------------------------------------------------------------


def get_data(url: str, destination_folder: str, data_name: str) -> None:
    """Download and unzip a dataset from a Figshare URL.

    The downloaded zip is extracted into ``destination_folder`` and then
    deleted. If the target zip is already present, the download is skipped
    and the existing file is unzipped again.

    Parameters
    ----------
    url : str
        Figshare download URL (e.g. ``"https://figshare.com/ndownloader/files/57981334"``).
    destination_folder : str
        Local directory to download into. Created if missing.
    data_name : str
        Stem used for the local zip filename (``<data_name>.zip``).
    """
    os.makedirs(destination_folder, exist_ok=True)

    local_filename = data_name + ".zip"
    local_path = os.path.join(destination_folder, local_filename)
    print(f"Downloading from {url}...")

    if not os.path.isfile(local_path):
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded and saved to {local_path}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return

    try:
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Successfully unzipped '{local_path}' into '{destination_folder}'.")
        os.remove(local_path)
    except zipfile.BadZipFile:
        print(f"Error: '{local_path}' is not a valid zip file or is corrupted.")
    except FileNotFoundError:
        print(f"Error: The file '{local_path}' was not found.")


# ---------------------------------------------------------------------------
# Per-dataset download + load helpers
# ---------------------------------------------------------------------------


_DEFAULT_DOWNLOAD_DIR = "./MIND_downloaded_data/"


def _ensure_local(get_fn, download_dir: str, marker_subdir: str) -> None:
    """Trigger a download if ``marker_subdir`` is not yet present."""
    if not os.path.isdir(os.path.join(download_dir, marker_subdir)):
        get_fn(download_dir)


def get_synthetic(download_dir: str = _DEFAULT_DOWNLOAD_DIR) -> None:
    """Download the synthetic dataset from the MIND paper.

    Parameters
    ----------
    download_dir : str
        Directory the data should be downloaded into. Created if missing.
    """
    get_data(
        "https://figshare.com/ndownloader/files/57981334",
        download_dir,
        "MIND_synthetic_data",
    )


def load_synthetic(
    download_dir: str = _DEFAULT_DOWNLOAD_DIR,
    mode: str = "high",
    download_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load the synthetic dataset.

    Parameters
    ----------
    download_dir : str
        Directory the data lives in (will be downloaded here on demand).
    mode : {'high', 'low'}, default='high'
        Noise regime to load.
    download_if_missing : bool, default=True
        If True (the default), download the dataset transparently when the
        local files are missing. Pass ``False`` to fail fast.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Dictionary with keys ``'RNA_expr'``, ``'Protein'``, ``'DNA_methyl'``
        (the omics modalities) and ``'cls'`` (cluster labels).
    """
    if mode not in {"high", "low"}:
        raise ValueError(f"mode must be 'high' or 'low', got {mode!r}.")
    if download_if_missing:
        _ensure_local(get_synthetic, download_dir, "synthetic_data")

    base = os.path.join(download_dir, "synthetic_data")
    sim_3 = pd.read_csv(os.path.join(base, f"sim_methyl_{mode}.csv"), index_col=0)
    sim_2 = pd.read_csv(os.path.join(base, f"sim_protein_{mode}.csv"), index_col=0)
    sim_1 = pd.read_csv(os.path.join(base, f"sim_expr_{mode}.csv"), index_col=0)
    sim_cls = pd.read_csv(os.path.join(base, f"sim_cls_{mode}.csv"), index_col=1)
    return {"RNA_expr": sim_1, "Protein": sim_2, "DNA_methyl": sim_3, "cls": sim_cls}


def get_TCGA(download_dir: str = _DEFAULT_DOWNLOAD_DIR) -> None:
    """Download the TCGA dataset from the MIND paper.

    Parameters
    ----------
    download_dir : str
        Directory the data should be downloaded into. Created if missing.
    """
    get_data(
        "https://figshare.com/ndownloader/files/57981316",
        download_dir,
        "MIND_TCGA_data",
    )


def load_TCGA(
    cancer_type: str,
    task: str = "all",
    download_dir: str = _DEFAULT_DOWNLOAD_DIR,
    download_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load a per-cancer slice of the TCGA dataset.

    Parameters
    ----------
    cancer_type : str
        TCGA cancer-type code (e.g. ``'BRCA'``, ``'LGG'``).
    task : {'all', 'recon_train', 'recon_test'}, default='all'
        ``'all'`` loads the full data; ``'recon_train'`` / ``'recon_test'``
        load the reconstruction-benchmark train/test split.
    download_dir : str
        Directory the data lives in (will be downloaded here on demand).
    download_if_missing : bool, default=True
        Whether to download the dataset if missing.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Dictionary keyed by modality name, plus a ``'clinical'`` entry with
        the per-patient phenotype labels.
    """
    if task not in {"all", "recon_train", "recon_test"}:
        raise ValueError(
            f"task must be one of 'all', 'recon_train', 'recon_test', got {task!r}."
        )
    if download_if_missing:
        _ensure_local(get_TCGA, download_dir, "TCGA_preprocessed")

    base = os.path.join(download_dir, "TCGA_preprocessed", cancer_type)
    mods = ["RNA", "methyl", "CNA", "miRNA", "RPPA"]
    ans: dict[str, pd.DataFrame] = {
        "clinical": pd.read_csv(
            os.path.join(base, "clinic_data.csv"), header=0, index_col=0
        )
    }
    if task == "recon_test":
        for mod in mods:
            path = os.path.join(base, f"{mod}_data_test.csv")
            if os.path.isfile(path):
                ans[mod] = pd.read_csv(path, header=0, index_col=0)
        return ans
    if task == "recon_train":
        for mod in mods:
            path = os.path.join(base, f"{mod}_data_train.csv")
            if os.path.isfile(path):
                ans[mod] = pd.read_csv(path, header=0, index_col=0)
        return ans

    # task == 'all'
    ans["RNA"] = pd.read_csv(os.path.join(base, "RNA_data.csv"), header=0, index_col=0)
    ans["methyl"] = pd.read_csv(
        os.path.join(base, "meth_data.csv"), header=0, index_col=0
    )
    ans["RPPA"] = pd.read_csv(
        os.path.join(base, "rppa_data_imp.csv"), header=0, index_col=0
    )
    ans["CNA"] = pd.read_csv(os.path.join(base, "cna_data.csv"), header=0, index_col=0)
    miRNA_path = os.path.join(base, "miRNA_data_imp.csv")
    if os.path.isfile(miRNA_path):
        ans["miRNA"] = pd.read_csv(miRNA_path, header=0, index_col=0)
    return ans


def get_CCMA(download_dir: str = _DEFAULT_DOWNLOAD_DIR) -> None:
    """Download the CCMA dataset from the MIND paper."""
    get_data(
        "https://figshare.com/ndownloader/files/57981304",
        download_dir,
        "MIND_CCMA_data",
    )


def load_CCMA(
    task: str = "all",
    download_dir: str = _DEFAULT_DOWNLOAD_DIR,
    download_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load the CCMA dataset.

    Parameters
    ----------
    task : {'all', 'recon_train', 'recon_test'}, default='all'
        Which split to load. ``'all'`` loads the full data; the other two
        load the held-out reconstruction split.
    download_dir : str
        Directory the data lives in (will be downloaded here on demand).
    download_if_missing : bool, default=True
        Whether to download the dataset if missing.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Keys: ``'RNA'``, ``'methyl'``, ``'CNV'``, ``'clinical'``.
    """
    if task not in {"all", "recon_train", "recon_test"}:
        raise ValueError(
            f"task must be one of 'all', 'recon_train', 'recon_test', got {task!r}."
        )
    if download_if_missing:
        _ensure_local(get_CCMA, download_dir, "CCMA_preprocessed")

    base = os.path.join(download_dir, "CCMA_preprocessed")
    ans: dict[str, pd.DataFrame] = {
        "clinical": pd.read_csv(
            os.path.join(base, "clinical.csv"), header=0, index_col=0
        )
    }
    suffix = {"all": "", "recon_train": "_train", "recon_test": "_test"}[task]
    ans["RNA"] = pd.read_csv(
        os.path.join(base, f"mRNA{suffix}.csv"), header=0, index_col=0
    )
    ans["methyl"] = pd.read_csv(
        os.path.join(base, f"meth{suffix}.csv"), header=0, index_col=0
    )
    ans["CNV"] = pd.read_csv(
        os.path.join(base, f"CNV{suffix}.csv"), header=0, index_col=0
    )
    return ans


def get_CCLE(download_dir: str = _DEFAULT_DOWNLOAD_DIR) -> None:
    """Download the CCLE dataset from the MIND paper."""
    get_data(
        "https://figshare.com/ndownloader/files/57981310",
        download_dir,
        "MIND_CCLE_data",
    )


def load_CCLE(
    task: str = "all",
    download_dir: str = _DEFAULT_DOWNLOAD_DIR,
    download_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load the CCLE dataset.

    Parameters
    ----------
    task : {'all', 'recon_train', 'recon_test'}, default='all'
        Which split to load.
    download_dir : str
        Directory the data lives in (will be downloaded here on demand).
    download_if_missing : bool, default=True
        Whether to download the dataset if missing.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Keys: ``'RNA'``, ``'meth'``, ``'cna'``, ``'metabolomics'``,
        ``'miRNA'``, ``'rppa'``, ``'clinical'``.
    """
    if task not in {"all", "recon_train", "recon_test"}:
        raise ValueError(
            f"task must be one of 'all', 'recon_train', 'recon_test', got {task!r}."
        )
    if download_if_missing:
        _ensure_local(get_CCLE, download_dir, "CCLE_preprocessed")

    base = os.path.join(download_dir, "CCLE_preprocessed")
    ans: dict[str, pd.DataFrame] = {
        "clinical": pd.read_csv(
            os.path.join(base, "clinic_data.csv"), header=0, index_col=0
        )
    }
    suffix = {"all": "", "recon_train": "_train", "recon_test": "_test"}[task]
    for key, stem in [
        ("RNA", "RNA_data"),
        ("meth", "meth_data"),
        ("cna", "cna_data"),
        ("metabolomics", "metabolomics_data"),
        ("miRNA", "miRNA_data"),
        ("rppa", "rppa_data"),
    ]:
        ans[key] = pd.read_csv(
            os.path.join(base, f"{stem}{suffix}.csv"), header=0, index_col=0
        )
    return ans


def download_all(download_dir: str = _DEFAULT_DOWNLOAD_DIR) -> None:
    """Convenience wrapper: download all four published MIND datasets."""
    get_synthetic(download_dir)
    get_TCGA(download_dir)
    get_CCLE(download_dir)
    get_CCMA(download_dir)
