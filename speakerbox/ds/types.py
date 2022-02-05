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


@dataclass_json
@dataclass
class AnnotatedAudio:
    label: str
    audio: str
    duration: float
