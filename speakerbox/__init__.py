# -*- coding: utf-8 -*-

"""Top-level package for speakerbox."""

__author__ = "Jackson Maxfield Brown"
__email__ = "jmaxfieldbrown@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.0"


def get_module_version() -> str:
    return __version__


from .ds.utils import expand_annotations_to_dataset  # noqa: F401
from .main import train  # noqa: F401
