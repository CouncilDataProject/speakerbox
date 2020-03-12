#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback
from pathlib import Path

from speakerbox.annotation import AudioAnnotator

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
            prog="annotate_audio_dataset",
            description="Start an audio dataset annotation session."
        )

        # Arguments
        p.add_argument(
            "dataset",
            type=Path,
            help="Path to the audio dataset."
        )
        p.add_argument(
            "config",
            type=Path,
            help="Path to annotation options."
        )
        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help="Show traceback if something goes wrong."
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################
# Begin audio annotator

def annotate_audio_dataset(args: Args):
    # Run the annotator
    try:
        # Initialize object
        annotator = AudioAnnotator(dataset=args.dataset, config=args.config)
        annotator.run_session()

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
    annotate_audio_dataset(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
