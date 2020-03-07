#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will convert all the steps into CLI callables.

You should not edit this script.
"""

import inspect
import logging

import fire

from speakerbox import steps
from speakerbox.bin.all import All

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


def cli():
    step_map = {
        name.lower(): step
        for name, step in inspect.getmembers(steps)
        if inspect.isclass(step)
    }

    fire.Fire({**step_map, "all": All})
