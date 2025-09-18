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
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_custom_sections = ["Returns", "Yields", "Raises", "Warns", "See Also"]

# Intersphinx mapping for cross-references to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "torax": ("https://torax.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Autodoc settings for type hints
autodoc_typehints = "both"
autodoc_typehints_description_target = "all"
autodoc_typehints_format = "short"
autodoc_type_aliases = {}
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
html_static_path = ["_static"]
numfig = True

html_theme_options = {
    "navigation_depth": 4,
}