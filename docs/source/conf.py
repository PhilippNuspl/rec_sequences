# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from codecs import open
from pathlib import Path

try:
    from sage.env import SAGE_DOC_SRC, SAGE_DOC, SAGE_SRC
    import sage.all
except ImportError:
    raise RuntimeError("to build the documentation you need to be inside a Sage shell (run first the command 'sage -sh' in a shell")

# Get information from separate files (README, VERSION)
def readfile(filename):
    with open(filename,  encoding='utf-8') as f:
        return f.read()

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'rec_sequences'
copyright = '2021, Philipp Nuspl'
author = 'Philipp Nuspl'

# The full version, including alpha/beta/rc tags
path = Path("../../", "VERSION")
release = readfile(str(path)).strip()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'autodocsumm'
]

# make summary appear on all autodoc pages
#autodoc_default_options = {
#    'autosummary': True,
#}

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'math'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []
#html_theme_path = [os.path.join(SAGE_DOC_SRC, 'common', 'themes')]
html_theme_path = [os.path.join(SAGE_DOC_SRC, 'common', 'themes', 'sage')]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autosummary_generate = True
