import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Add these extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser'
]

# Theme settings
html_theme = 'sphinx_rtd_theme'
