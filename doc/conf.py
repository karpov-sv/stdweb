# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'STDWeb'
copyright = '2025, Sergey Karpov'
author = 'Sergey Karpov'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_parser',  # Markdown support
]

# Markdown support configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'fieldlist',
]

# Generate anchors for headings up to this level so that intra-page links in
# the included Markdown reference files (e.g. REST_API.md, TASK_CONFIG.md)
# resolve. Without this a -W build would fail on those cross-references.
myst_heading_anchors = 4

# Cosmetic Pygments highlighting failures (e.g. abbreviated JSON examples that
# use "..." placeholders) should not break a -W build; real problems such as
# broken cross-references and missing toctree entries still do.
suppress_warnings = ['misc.highlighting_failure']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
}
