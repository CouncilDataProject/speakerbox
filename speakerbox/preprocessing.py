#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, Union

from pyannote.audio import Pipeline
from pydub import AudioSegment

###############################################################################


def diarize_and_split_audio(
    audio_file: Union[str, Path],
    storage_dir: Optional[Union[str, Path]] = None,
    diarization_pipeline: Optional[Pipeline] = None,
) -> Path:
    """
    Diarize a single audio file and split the file into smaller chunks stored into
    directories with the unlabeled speaker annotation.

    Parameters
    ----------
    audio_file: Union[str, Path]
        The audio file to diarize and split.
    storage_dir: Optional[Union[str, Path]]
        A specific directory to store the produced chunks to.
        Default: None (use the audio file name to create a new directory)
    diarization_pipeline: Optional[Pipeline]
        A preloaded PyAnnote Pipeline.
        Default: None (load default)

    Returns
    -------
    storage_dir: Path
        The path to where all the chunked audio was stored.

    Notes
    -----
    The directory structure of the produced chunks will follow the pattern::

        {storage_dir}/
        ├── SPEAKER_00
        │   ├── {start_time_millis}-{start_end_millis}.wav
        │   └── {start_time_millis}-{start_end_millis}.wav
        ├── SPEAKER_01
        │   ├── {start_time_millis}-{start_end_millis}.wav
        │   └── {start_time_millis}-{start_end_millis}.wav
    """
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

    # Load audio once to pull and store chunks
    audio = AudioSegment.from_file(audio_file)

    # Iter labeled segments and store
    for turn, _, speaker in dia.itertracks(yield_label=True):
        # Make speaker dir if needed
        speaker_dir = storage_dir / speaker.replace(" ", "_")
        speaker_dir.mkdir(exist_ok=True)

        # Get chunk
        turn_start_millis = turn.start * 1000
        turn_end_millis = turn.end * 1000
        chunk = audio[turn_start_millis:turn_end_millis]

        # Save chunk
        chunk_save_path = (
            speaker_dir / f"{int(turn_start_millis)}-{int(turn_end_millis)}.wav"
        )
        chunk.export(chunk_save_path, format="wav")

    return storage_dir
