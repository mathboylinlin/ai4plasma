# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add parent directory to path for autodoc
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ai4plasma'
copyright = '2026, Linlin Zhong'
author = 'Linlin Zhong'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    "sphinx.ext.mathjax",
    # 'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# MyST parser configuration
myst_enable_extensions = [
    'dollarmath',  # Support for LaTeX math
    'amsmath',     # AMS math environments
    'tasklist',    # Task lists
    'colon_fence', # ::: fenced blocks
]

# Master document
master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Language and highlighting
language = 'en'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
}

# Sidebar configuration
# html_sidebars = {
#     '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html']
# }

# -- Options for autodoc -------------------------------------------------------
# autodoc_default_options = {
#     'members': True,
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': False,
#     'show-inheritance': True,
# }

# Napoleon configuration (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_attr_annotations = True
