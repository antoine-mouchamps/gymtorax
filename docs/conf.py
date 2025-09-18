# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Gymtorax"
copyright = "2025, Mouchamps Antoine, Malherbe Arthur, Bolland Adrien, Ernst Damien"
author = "Mouchamps Antoine, Malherbe Arthur, Bolland Adrien, Ernst Damien"
release = "2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx_design",
]

# Napoleon settings (Google docstring style)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Intersphinx mapping for cross-references to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "torax": ("https://torax.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_member_order = "bysource"

# Mock imports for dependencies that might not be available during doc build
autodoc_mock_imports = [
    "torax",
    "jax",
    "jaxlib",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
numfig = True

html_theme_options = {
    "navigation_depth": 4,
}
