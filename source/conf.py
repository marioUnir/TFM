# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

project = 'credit-card'
copyright = '2025, Cristina Domínguez and Mario Río'
author = 'Cristina Domínguez and Mario Río'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',  # Para incluir enlaces al código fuente
    'sphinx.ext.mathjax',  # Necesario para fórmulas matemáticas
]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# Personalización del tema
html_theme_options = {
    'collapse_navigation': False,  # Mantener el menú de navegación expandido
    'style_external_links': True,  # Estilo para enlaces externos
    'navigation_depth': 4,         # Profundidad del menú de navegación
    'titles_only': False           # Mostrar títulos en el menú lateral
}

# Estilo propio
html_static_path = ['_static']

# Incluye un archivo de CSS personalizado
html_css_files = [
    'custom.css',  # Nuestro archivo personalizado
]

html_logo = "_static/unir_logo.png"  # Ruta al logo
html_favicon = "_static/unir_logo.png"  # Puedes usar el mismo logo como favicon

html_title = "TFM - UNIR | Documentación de la librería credit-card"