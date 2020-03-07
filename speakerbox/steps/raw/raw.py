#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

import numpy as np
import pandas as pd
from datastep import Step, log_run_params
from PIL import Image

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Raw(Step):
    # You only need to have an __init__ if you aren't using the default values
    # In this case, we could get rid of it but for the purposes of this example
    # we will keep it.
    def __init__(self, direct_upstream_tasks=[], config=None):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(self, n: int = 10, **kwargs) -> List[Path]:
        """
        Generate N random images and save them to /images.

        Parameters
        ----------
        n: int
            Number of images to generate.

        Returns
        -------
        images: List[Path]
            A list of paths that point to the generated images.
        """

        # Empty manifest to fill in -- add more columns for e.g. labels, metadata, etc.
        self.manifest = pd.DataFrame(index=range(n), columns=["filepath"])

        # Subdirectory for the images
        imdir = self.step_local_staging_dir / Path("images")
        imdir.mkdir(parents=True, exist_ok=True)

        # Set seed for reproducible random images
        np.random.seed(seed=112358)

        # Create images, save them, and fill in dataframe
        images = []
        for i in tqdm(range(n), desc="Creating and saving images"):
            A = np.random.rand(128, 128, 4) * 255
            img = Image.fromarray(A.astype("uint8")).convert("RGBA")
            path = imdir / Path(f"image_{i}.png")
            img.save(path)
            self.manifest.at[i, "filepath"] = path
            images.append(path)

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )

        return images
