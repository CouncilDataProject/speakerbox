#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from dataclasses_json import dataclass_json

###############################################################################


class GeckoAnnotationAndAudio(NamedTuple):
    annotation_file: Path
    audio_file: Path


@dataclass_json
@dataclass
class AnnotatedAudio:
    conversation_id: str
    label: str
    audio: str
    duration: float
