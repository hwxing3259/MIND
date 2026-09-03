"""Backwards-compatible setup.py.

The canonical metadata lives in ``pyproject.toml`` (PEP 621). This file is
kept so that ``pip install -e .`` continues to work in environments where
only setuptools is available.
"""

from setuptools import find_packages, setup

VERSION = "0.1.0"
DESCRIPTION = "Multimodal Integration with Neighbourhood-aware Distributions"
LONG_DESCRIPTION = (
    "MIND is a VAE-style model that learns a single shared per-patient "
    "embedding from several partially-overlapping omics modalities. "
    "See https://www.biorxiv.org/content/10.1101/2025.09.15.676314v1 for "
    "the methodology."
)

setup(
    name="MIND",
    version=VERSION,
    author="Hanwen Xing",
    author_email="hanwen.xing@wrh.ox.ac.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/plain",
    url="https://github.com/hwxing3259/MIND",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*", "docs", "examples")),
    python_requires=">=3.10",
    install_requires=[
        "pandas~=2.2",
        "numpy~=1.26",
        "matplotlib~=3.8",
        "torch~=2.2",
        "openTSNE~=1.0",
        "requests~=2.32",
    ],
    extras_require={
        "dev": ["pytest~=8.0", "pytest-cov~=5.0", "ruff~=0.5", "black~=24.0"],
        "docs": ["sphinx~=7.0", "sphinx-rtd-theme~=2.0", "myst-parser~=3.0"],
    },
    keywords=[
        "multi-omics", "vae", "deep learning", "bioinformatics", "integration",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
