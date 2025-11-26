import os
import sys

# If conf.py is in docs/, repo root is one level up
sys.path.insert(0, os.path.abspath(".."))

project = "scikit-covtest"
author = "Your Name"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

