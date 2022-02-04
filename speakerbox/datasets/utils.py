#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import shutil
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pydub import AudioSegment

from .types import AnnotatedAudio, AnnotationAndAudio

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _unpack_zip(
    zipfile: Union[str, Path],
    dest: Union[str, Path],
    clean: bool = False,
) -> Path:
    """
    Unzips the zipfile to the destination location.

    Parameters
    ----------
    zipfile: Union[str, Path]
        The zipfile to unpack.
    dest: Union[str, Path]
        The destination to unpack to.
    clean: bool
        If a directory already exists at the destination location, should the directory
        be removed entirely before unpacking again.
        Default: False

    Returns
    -------
    dataset_path: Path
        The path to the unpacked data.

    Raises
    ------
    NotADirectoryError
        A file exists at the specified destination.
    FileExistsError
        A directory exists at the specified destination and is not empty.
    """
    zipfile = Path(zipfile).resolve(strict=True)
    dest = Path(dest)

    # Check dest is a dir
    if dest.is_file():
        raise NotADirectoryError(dest)

    # Clean dest if required
    if dest.is_dir():
        dest_is_empty = not any(dest.iterdir())
        if not dest_is_empty:
            if clean:
                shutil.rmtree(dest)
            else:
                raise FileExistsError(
                    f"Files found in target destination directory ({dest}) "
                    f"but parameter `clean` set to False."
                )

    # Make new dir
    dest.mkdir(parents=True, exist_ok=True)

    # Extract
    shutil.unpack_archive(zipfile, dest)

    # Return extracted data dir
    return dest.resolve(strict=True)


def expand_annotations_to_dataset(
    annotations_and_audios: List[AnnotationAndAudio],
    audio_output_dir: Union[str, Path] = Path("chunked-audio/"),
    overwrite: bool = False,
    max_audio_chunk_duration: int = 5,
) -> pd.DataFrame:
    """
    Expand a list of annotation and audio files into a full dataset to be used for
    training and testing a speaker classification model.

    Parameters
    ----------
    annotations_and_audios: List[AnnotationAndAudio]
        A list of annotation and their matching audio files to expand into a speaker,
        audio file path, start and end times.
    audio_output_dir: Union[str, Path]
        A directory path to store the chunked audio files in.
        Default: "chunked-audio" directory in the current working directory.
    overwrite: bool
        When writting out an audio chunk, should existing files be overwritten.
        Default: False (do not overwrite)
    max_audio_chunk_duration: int
        Length of the audio duration to split larger audio files into.
        Default: 5 seconds

    Returns
    -------
    dataset: pd.DataFrame
        The expanded dataset with columns: speaker, audio_file, start_time, end_time

    Raises
    ------
    NotADirectoryError
        A file exists at the specified destination.
    FileExistsError
        A file exists at the target chunk audio location but overwrite is False.
    """
    # Ensure dataset dir is valid
    audio_output_dir = Path(audio_output_dir).resolve()
    if audio_output_dir.is_file():
        raise NotADirectoryError(audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Iter annotations and audios
    annotated_audios: List[AnnotatedAudio] = []
    for aaa in annotations_and_audios:
        audio = AudioSegment.from_file(aaa.audio_file)
        with open(aaa.annotation_file, "r") as open_f:
            annotations = json.load(open_f)

        # Iter through each "monologue" section and add up times
        for i, monologue in enumerate(annotations["monologues"]):
            monologue_speaker = monologue["speaker"]["id"]
            monologue_start_time = monologue["start"]
            monologue_end_time = monologue["end"]

            # Split monologue into chunks
            for chunk_i, chunk_start_time in enumerate(
                np.arange(
                    monologue_start_time,
                    monologue_end_time,
                    max_audio_chunk_duration,
                )
            ):
                # Get chunk start and end in millis
                chunk_start_millis = chunk_start_time * 1000
                chunk_end = chunk_start_time + max_audio_chunk_duration

                # Only add chunks where the full chunk is available
                # (if there is more time available in the monologue)
                if monologue_end_time > chunk_end:
                    chunk_end_millis = chunk_end * 1000

                    # Get chunk
                    chunk = audio[chunk_start_millis:chunk_end_millis]

                    # Determine save pattern
                    chunk_save_path = (
                        audio_output_dir / f"{aaa.audio_file.with_suffix('').name}-"
                        f"monologue_{i}-{monologue_speaker}-chunk_{chunk_i}.wav"
                    )
                    if chunk_save_path.exists() and not overwrite:
                        raise FileExistsError(chunk_save_path)

                    # Save audio chunk
                    chunk.export(chunk_save_path, format="wav")

                    # Append new chunk to list
                    annotated_audios.append(
                        AnnotatedAudio(
                            label=monologue_speaker,
                            audio=str(chunk_save_path),
                            duration=(chunk_end_millis - chunk_start_millis) / 1000,
                        )
                    )

        log.info(f"Completed expansion for annotation file: {aaa.annotation_file}")

    return pd.DataFrame([aa.to_dict() for aa in annotated_audios])
