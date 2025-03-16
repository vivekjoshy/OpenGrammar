"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from typing import List

import opengrammar

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project: str = "opengrammar"
copyright: str = "Copyright &copy 2025, Vivek Joshy"
author: str = "Vivek Joshy"
version: str = opengrammar.__version__
release: str = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: List[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "autoapi.extension",
    "sphinxcontrib.bibtex",
    "sphinx_favicon",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
    "sphinx_docsearch",
]

templates_path: List[str] = ["_templates"]
exclude_patterns: List[str] = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "OpenGrammar: Neural Language Model"
html_theme = "shibuya"
html_static_path = ["_static"]
html_theme_options = {
    "dark_code": True,
    "github_url": "https://github.com/vivekjoshy/OpenGrammar",
    "logo_target": "/",
    "light_logo": "_static/text_logo_light.png",
    "dark_logo": "_static/text_logo_dark.png",
}
html_css_files = [
    "custom.css",
]
html_context = {
    "source_type": "github",
    "source_user": "vivekjoshy",
    "source_repo": "OpenGrammar",
}

html_copy_source = False
html_show_sourcelink = False

# -- Options for AutoDoc output ---------------------------------------------'
autodoc_typehints = "description"
autodoc_preserve_defaults = True


# -- Options for AutoAPI output ---------------------------------------------
autoapi_root = "api"
autoapi_dirs = ["../../opengrammar"]
autoapi_template_dir = "_templates"
autoapi_options = [
    "members",
    "show-inheritance",
    "private-members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"

# -- Options for BibTeX output -----------------------------------------
bibtex_bibfiles = ["./bibliography.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# -- Options for MathJax output ---------------------------------------------
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"


# -- Options for OpenGraph output ---------------------------------------------
ogp_site_url = "https://opengrammar.readthedocs.io/en/stable/"
ogp_site_name = "OpenGrammar: Neural Language Model"
ogp_image = "https://i.imgur.com/bkfGSeF.png"
ogp_description_length = 200
ogp_type = "website"
ogp_enable_meta_description = True

# -- Options for Favicon output ---------------------------------------------
html_favicon = "_static/favicon.ico"
favicons = [
    {"href": "logo.svg"},
    {"href": "favicon-16x16.png"},
    {"href": "favicon-32x32.png"},
    {
        "rel": "apple-touch-icon",
        "href": "apple-touch-icon.png",
    },
    {
        "rel": "manifest",
        "href": "site.webmanifest",
    },
]
