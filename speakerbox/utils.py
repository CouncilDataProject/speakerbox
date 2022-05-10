#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from typing import Union

###############################################################################


def _unpack_zip(
    zipfile: Union[str, Path],
    dest: Union[str, Path],
    clean: bool = False,
) -> Path:
    """
    Unzips the zipfile to the destination location.

    Parameters
    ----------
    zipfile: Union[str, Path]
        The zipfile to unpack.
    dest: Union[str, Path]
        The destination to unpack to.
    clean: bool
        If a directory already exists at the destination location, should the directory
        be removed entirely before unpacking again.
        Default: False

    Returns
    -------
    dataset_path: Path
        The path to the unpacked data.

    Raises
    ------
    NotADirectoryError
        A file exists at the specified destination.
    FileExistsError
        A directory exists at the specified destination and is not empty.
    """
    zipfile = Path(zipfile).resolve(strict=True)
    dest = Path(dest)

    # Check dest is a dir
    if dest.is_file():
        raise NotADirectoryError(dest)

    # Clean dest if required
    if dest.is_dir():
        dest_is_empty = not any(dest.iterdir())
        if not dest_is_empty:
            if clean:
                shutil.rmtree(dest)
            else:
                raise FileExistsError(
                    f"Files found in target destination directory ({dest}) "
                    f"but parameter `clean` set to False."
                )

    # Make new dir
    dest.mkdir(parents=True, exist_ok=True)

    # Extract
    shutil.unpack_archive(zipfile, dest)

    # Return extracted data dir
    return dest.resolve(strict=True)


def set_global_seed(seed: int) -> None:
    """
    Set the global RNG seed for torch, numpy, and Python.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
