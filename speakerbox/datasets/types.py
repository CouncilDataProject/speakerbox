#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from dataclasses_json import dataclass_json

###############################################################################


class AnnotationAndAudio(NamedTuple):
    annotation_file: Path
    audio_file: Path


@dataclass
@dataclass_json
class AnnotatedAudio:
    speaker: str
    audio_file: str
    start_time: float
    end_time: float
