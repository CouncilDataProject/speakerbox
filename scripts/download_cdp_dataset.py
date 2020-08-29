#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script requires version 3.* of cdptools
# install by doing:
# git clone https://github.com/councildataproject/cdptools
# cd cdptools
# git checkout feature/update-index-pipeline
# pip install .[seattle]

import argparse
import logging
import sys
import traceback
from pathlib import Path

from cdptools import CDPInstance, configs
from cdptools.research_utils import generate_dataset

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s"
)
log = logging.getLogger(__name__)

###############################################################################
# Args


class Args(argparse.Namespace):

    def __init__(self):
        self.__parse()

    def __parse(self):
        # Setup parser
        p = argparse.ArgumentParser(
            prog="download_cdp_dataset",
            description="Download a dataset ready for training from a CDP instance."
        )

        # Arguments
        p.add_argument(
            "instance_name",
            help="Which CDP instance to retrieve the dataset from."
        )
        p.add_argument(
            "--save_dir",
            type=Path,
            default=Path("cdp_data"),
            help="Path to save the dataset to."
        )
        p.add_argument(
            "--overwrite",
            action="store_true",
            dest="overwrite",
            help="Overwrite existing files."
        )
        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help="Show traceback if the script were to fail."
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################
# Download and prepare CDP dataset

def download_cdp_dataset(args: Args):
    # Try running the download pipeline
    try:
        # Get instance config
        instance_config = getattr(configs, args.instance_name.upper())

        # Create connection to instance
        cdp_instance = CDPInstance(instance_config)

        # Get initial CDP dataframe
        data = generate_dataset(cdp_instance)

        # Store to CSV
        args.save_dir.mkdir(exist_ok=True, parents=True)
        data.to_csv(args.save_dir / "manifest.csv", index=False)

    # Catch any exception
    except Exception as e:
        log.error("=============================================")
        if args.debug:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)

###############################################################################
# Runner


def main():
    args = Args()
    download_cdp_dataset(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
