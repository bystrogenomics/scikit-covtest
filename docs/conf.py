import os
import sys

# -------------------------------------------------------------------
# Path setup: make your package importable
# -------------------------------------------------------------------
# If your package "covtest" lives at the repo root:
sys.path.insert(0, os.path.abspath(".."))

# If you use a src/ layout (src/covtest), instead do:
# sys.path.insert(0, os.path.abspath("../src"))

# -------------------------------------------------------------------
# Project information
# -------------------------------------------------------------------
project = "scikit-covtest"
author = "Your Name"

# Try to get the version from the package if available
try:
    import covtest
    release = covtest.__version__
except Exception:
    release = "0.0.0"

# -------------------------------------------------------------------
# General configuration
# -------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# Generate autosummary stub pages automatically
autosummary_generate = True

# Autodoc defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# If some imports are heavy / optional, you can mock them:
# autodoc_mock_imports = ["sklearn", "jax", "numpyro"]

# Napoleon: use NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Put type hints into the description, not in the signature
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -------------------------------------------------------------------
# HTML output
# -------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

