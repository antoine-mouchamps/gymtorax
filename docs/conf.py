# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "GymTORAX"
copyright = f"2024-{datetime.now().year}, Antoine Mouchamps, Arthur Malherbe, Adrien Bolland, Damien Ernst"
author = "Antoine Mouchamps, Arthur Malherbe, Adrien Bolland, Damien Ernst"

# The version info for the project
release = "0.1.1"
version = "0.1.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
]

# Napoleon settings (Google docstring style)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = True

napoleon_custom_sections = [
    ("Class Attributes", "params_style"),
    ("Action Parameters", "params_style"),
]

# Intersphinx mapping for cross-references to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "torax": ("https://torax.readthedocs.io/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# General settings
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".coverage"]
templates_path = ["_templates"]
language = "en"

# Math settings
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# -- Autodoc configuration ---------------------------------------------------

# Autodoc settings for type hints
autodoc_typehints = "both"
autodoc_typehints_description_target = "all"
autodoc_typehints_format = "short"
autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}

# Mock imports for dependencies that might not be available during doc build
autodoc_mock_imports = [
    "torax",
    "jax",
    "jaxlib",
]

# Remove warnings related to TORAX types that are not referenced in their docs
nitpick_ignore = [
    ("py:class", "FigureProperties"),
    ("py:class", "plotruns_lib.FigureProperties"),
    ("py:class", "torax._src.plotting.plotruns_lib.FigureProperties"),
    ("py:class", "plotruns_lib.PlotData"),
    ("py:class", "torax._src.plotting.plotruns_lib.PlotData"),
    ("py:class", "ToraxSimState"),
    ("py:class", "PostProcessedOutputs"),
    ("py:class", "StateHistory"),
]

# -- Copy button configuration -----------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "Images/logo.png"
html_favicon = "Images/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s'
}

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "vcs_pageview_mode": "",
    "version_selector": True,
}
html_context = {
    "display_github": True,
    "github_user": "antoine-mouchamps",
    "github_repo": "gymtorax",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
