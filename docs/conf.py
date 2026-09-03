"""Sphinx configuration for the MIND package."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "MIND"
author = "Hanwen Xing"
copyright = "2026, Hanwen Xing"

# Read the package version dynamically.
try:
    from MIND import __version__ as release  # type: ignore[import-not-found]
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # Don't pull in inherited methods from torch.nn.Module — their docstrings
    # are written for PyTorch's own Sphinx config and emit reST warnings here.
    "inherited-members": False,
}

# Suppress reST issues that originate inside third-party docstrings we don't
# control. This is robust against future PyTorch releases adding more methods.
suppress_warnings = ["docutils"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
