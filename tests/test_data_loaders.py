"""Integration tests for the dataset download/load helpers.

These tests do **not** hit the network. They mock ``requests.get`` and
``zipfile.ZipFile`` to verify that the download plumbing (URL + path
construction, idempotent extraction) is wired correctly.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from MIND import data as data_mod


def _write_dummy_zip(path: Path, files: dict[str, str]) -> None:
    """Create a zip archive at ``path`` containing the given files."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, contents in files.items():
            zf.writestr(name, contents)


def test_get_data_skips_download_if_zip_present(tmp_path):
    """If the zip already exists, ``get_data`` should not call requests."""
    dest = tmp_path / "out"
    dest.mkdir()
    zip_path = dest / "MIND_synthetic_data.zip"
    _write_dummy_zip(zip_path, {"hello.txt": "world"})

    with mock.patch.object(data_mod.requests, "get") as mock_get:
        data_mod.get_data(
            "https://example.invalid/foo", str(dest), "MIND_synthetic_data"
        )
        mock_get.assert_not_called()
    assert (dest / "hello.txt").read_text() == "world"
    assert not zip_path.exists()  # cleaned up after extraction


def test_get_data_downloads_when_missing(tmp_path):
    """If the zip is missing, ``get_data`` should fetch and unpack it."""
    dest = tmp_path / "out"

    fake_zip_bytes = _make_fake_zip_bytes({"data.csv": "a,b\n1,2\n"})

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield fake_zip_bytes

    with mock.patch.object(data_mod.requests, "get", return_value=FakeResponse()) as mock_get:
        data_mod.get_data(
            "https://example.invalid/foo", str(dest), "MIND_synthetic_data"
        )
        mock_get.assert_called_once()
    assert (dest / "data.csv").read_text() == "a,b\n1,2\n"


def _make_fake_zip_bytes(files: dict[str, str]) -> bytes:
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, contents in files.items():
            zf.writestr(name, contents)
    return buf.getvalue()


def test_load_synthetic_validates_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        data_mod.load_synthetic(
            download_dir=str(tmp_path), mode="medium", download_if_missing=False
        )


def test_load_ccma_validates_task(tmp_path):
    with pytest.raises(ValueError, match="task"):
        data_mod.load_CCMA(
            task="bogus", download_dir=str(tmp_path), download_if_missing=False
        )


def test_load_ccle_validates_task(tmp_path):
    with pytest.raises(ValueError, match="task"):
        data_mod.load_CCLE(
            task="bogus", download_dir=str(tmp_path), download_if_missing=False
        )


def test_load_tcga_validates_task(tmp_path):
    with pytest.raises(ValueError, match="task"):
        data_mod.load_TCGA(
            cancer_type="BRCA",
            task="bogus",
            download_dir=str(tmp_path),
            download_if_missing=False,
        )


def test_load_ccma_reads_layout(tmp_path):
    """End-to-end: hand-craft the directory layout, verify keys + shapes."""
    base = tmp_path / "CCMA_preprocessed"
    base.mkdir(parents=True)
    pd.DataFrame({"label": [0, 1, 2]}, index=["a", "b", "c"]).to_csv(
        base / "clinical.csv"
    )
    for stem in ["mRNA", "meth", "CNV"]:
        pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            index=["a", "b", "c"],
            columns=["x", "y"],
        ).to_csv(base / f"{stem}.csv")

    out = data_mod.load_CCMA(
        task="all", download_dir=str(tmp_path) + os.sep, download_if_missing=False
    )
    assert set(out.keys()) == {"clinical", "RNA", "methyl", "CNV"}
    assert out["RNA"].shape == (3, 2)
