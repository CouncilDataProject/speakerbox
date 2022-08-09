# -*- coding: utf-8 -*-

"""Top-level package for speakerbox."""

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.0"


def get_module_version() -> str:
    return __version__


from .main import eval_model, train  # noqa: F401
