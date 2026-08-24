import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "AstroVOX"
author = "Arnab Lahiry"
copyright = "2026, Arnab Lahiry"
release = "1.0.1"

extensions = [
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

autoapi_dirs = ["../src/astrovox"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = True
autoapi_keep_files = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
}

html_theme = "sphinx_book_theme"
html_title = "AstroVOX"
html_static_path = ["_static"]
html_logo = "_static/volumetric_ico.png"
html_favicon = "_static/volumetric_ico.png"
html_baseurl = "https://arnablahiry.github.io/software/astrovox/"

html_theme_options = {
    "repository_url": "https://github.com/arnablahiry/AstroVOX",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": False,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "logo": {
        "image_light": "volumetric_ico.png",
        "image_dark": "volumetric_ico.png",
    },
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
