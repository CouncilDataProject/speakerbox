#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union
from pathlib import Path
from pyannote.audio import Pipeline

###############################################################################


def diarize_audio(
    audio_file: Union[str, Path],
    storage_dir: Optional[Union[str, Path]] = None,
    diarization_pipeline: Optional[Pipeline] = None,
) -> Path:
    # Handle audio file
    if isinstance(audio_file, str):
        audio_file = Path(audio_file)

    # Ensure file exists
    audio_file = audio_file.resolve(strict=True)

    # Handle storage dir
    if isinstance(storage_dir, str):
        storage_dir = Path(storage_dir).resolve()
    elif storage_dir is None:
        # No storage dir provided
        # Make one using the audio file name
        audio_file_name = audio_file.with_suffix("").name
        storage_dir = Path(f"dia-{audio_file_name}").resolve()

    # Make storage dir
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    if diarization_pipeline is None:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # Diarize
    dia = diarization_pipeline(audio_file)

    # Iter labeled segments
    count = 0
    for turn, label, speaker in dia.itertracks(yield_label=True):
        print(label)
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        count += 1
        if count == 10:
            break
