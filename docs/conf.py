# Configuration file for the Sphinx documentation builder.

project = 'MLC Lab Report'
copyright = '2026, Julian & Justin'
author = 'Julian, Justin'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster' # Standard theme, no external dependencies needed
html_static_path = ['_static']
