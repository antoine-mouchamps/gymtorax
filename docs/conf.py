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

project = "GymTORAX"
copyright = "2025, Mouchamps Antoine, Malherbe Arthur, Bolland Adrien, Ernst Damien"
author = "Mouchamps Antoine, Malherbe Arthur, Bolland Adrien, Ernst Damien"
release = "0.1.0a1"
version = "0.1.0a1"

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
napoleon_include_special_with_doc = True
napoleon_use_ivar = True

napoleon_custom_sections = [
    ('Class Attributes', 'params_style'),
    ('Action Parameters', 'params_style'),
]

# Intersphinx mapping for cross-references to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "torax": ("https://torax.readthedocs.io/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Autodoc settings for type hints
autodoc_typehints = "both"
autodoc_typehints_description_target = "all"
autodoc_typehints_format = "short"
autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__",
}

# Mock imports for dependencies that might not be available during doc build
autodoc_mock_imports = [
    "torax",
    "jax",
    "jaxlib",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
numfig = True

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "vcs_pageview_mode": "",
    'version_selector': True,
}

# Add navigation configuration for GitHub integration
html_context = {
    "display_github": True,
    "github_user": "antoine-mouchamps",
    "github_repo": "gymtorax",
    "github_version": "main",
    "conf_py_path": "/docs/",
}