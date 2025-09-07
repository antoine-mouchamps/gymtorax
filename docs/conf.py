# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Gymtorax'
copyright = '2025, Mouchamps Antoine, Malherbe Arthur, Bolland Adrien, Ernst Damien'
author = 'Mouchamps Antoine, Malherbe Arthur, Bolland Adrien, Ernst Damien'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Pour les formats Google/NumPy
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
numfig = True

html_theme_options = {
    'navigation_depth': 4,
}
