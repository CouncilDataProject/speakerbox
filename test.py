#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

from speakerbox.preprocessing import diarize_and_split_audio

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

test_file = Path(__file__).parent / "seattle-2021-proto/audio/5e881a137b6d.wav"

diarize_and_split_audio(test_file)
