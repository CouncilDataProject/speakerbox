#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run all tasks in a prefect Flow.

When you add steps to you step workflow be sure to add them to the step list
and configure their IO in the `run` function.
"""

import logging

from distributed import LocalCluster
from prefect import Flow
from prefect.engine.executors import DaskExecutor, LocalExecutor

from speakerbox import steps

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class All:
    def __init__(self):
        """
        Set all of your available steps here.
        This is only used for data logging operations, not computation purposes.
        """
        self.step_list = [steps.Raw()]

    def run(
        self,
        clean: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        """
        Run a flow with your steps.

        Parameters
        ----------
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Notes
        -----
        Documentation on prefect:
        https://docs.prefect.io/core/

        Basic prefect example:
        https://docs.prefect.io/core/
        """
        # Initalize steps
        raw = steps.Raw()

        # Choose executor
        if debug:
            exe = LocalExecutor()
        else:
            # Set up connection to computation cluster
            cluster = LocalCluster()

            # Inform of Dask UI
            log.info(f"Cluster dashboard available at: {cluster.dashboard_link}")

            # Create dask executor
            exe = DaskExecutor(cluster.scheduler_address)

        # Configure your flow
        with Flow("speakerbox") as flow:
            # If you want to clean the local staging directories pass clean
            # If you want to utilize some debugging functionality pass debug
            # If you don't utilize any of these, just pass the parameters you need.
            raw(
                clean=clean,
                debug=debug,
                **kwargs,  # Allows us to pass `--n {some integer}` or other params
            )

        # Run flow and get ending state
        state = flow.run(executor=exe)

        # Get and display any outputs you want to see on your local terminal
        log.info(raw.get_result(state, flow))

    def pull(self):
        """
        Pull all steps.
        """
        for step in self.step_list:
            step.pull()

    def checkout(self):
        """
        Checkout all steps.
        """
        for step in self.step_list:
            step.checkout()

    def push(self):
        """
        Push all steps.
        """
        for step in self.step_list:
            step.push()

    def clean(self):
        """
        Clean all steps.
        """
        for step in self.step_list:
            step.clean()
