#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from speakerbox.preprocessing import diarize_audio

###############################################################################

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

diarize_audio(
    "/Users/maxfield/Desktop/active/cdp/speakerbox/seattle-2021-proto/audio/7fe4c0d99b44.wav"
)
