# -*- coding: utf-8 -*-

"""Top-level package for speakerbox."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("speakerbox")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"

from .main import apply, eval_model, train

__all__ = ["apply", "eval_model", "train"]
