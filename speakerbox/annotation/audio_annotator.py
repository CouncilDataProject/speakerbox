#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from pynput.keyboard import Key, Listener

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class AudioAnnotator:

    def __init__(
        self,
        dataset: Union[str, Path, pd.DataFrame],
        config: Union[str, Path, Dict]
    ):
        # Resolves paths
        if isinstance(dataset, (str, Path)):
            dataset = Path(dataset).expanduser().resolve(strict=True)

            # Read
            dataset = pd.read_csv(dataset)

        # Store for iterative processing
        self.dataset = dataset
        self.current = 0

        # Audio controls
        self.paused = False

    def _handle_key_press(self, key):
        # Handle escape
        if key is Key.esc:
            log.info("Ending annotation session")
            sys.stdin.flush()
            sys.exit()
        elif key is Key.left:
            log.info("Restarting audio clip")
        elif key is Key.right:
            log.info("Skipping audio clip")
        elif key is Key.space:
            if self.paused:
                log.info("Resuming audio playback")
                self.paused = False
            else:
                log.info("Pausing audio playback")
                self.paused = True
        else:
            print(self.dataset.head())

    def run_session(self):
        # Run until we hear an escape key press
        with Listener(
            on_press=lambda key: self._handle_key_press(key),
            on_release=lambda key: None
        ) as keyboard_listener:
            keyboard_listener.join()
