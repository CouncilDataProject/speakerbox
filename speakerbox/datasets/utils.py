#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from .types import AnnotatedAudio, AnnotationAndAudio

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


###############################################################################
# Dataset Summarization


@dataclass
class SpeakerTime:
    label: str
    duration: float
    percent_of_total: float


@dataclass
class DatasetSummary:
    n_speakers: int
    n_events: int
    speaker_times: List[SpeakerTime]
    mean_speaker_time: float
    min_speaker_time: SpeakerTime
    max_speaker_time: SpeakerTime
    std_speaker_time: float
    total_speaker_time: float


def summarize_annotation_statistics(
    annotations_dir: Union[str, Path],
) -> DatasetSummary:
    """
    Summarize all annotation files into a single pandas DataFrame.

    Parameters
    ----------
    annotations_dir: Union[str, Path]
        The path to the directory which contains the annotation files for a dataset.

    Returns
    -------
    summary: DatasetSummary
        The summary stats for the dataset.

    Examples
    --------
    >>> from speakerbox.datasets import (
    ...     unpack_seattle_2021_proto,
    ...     summarize_annotation_statistics,
    ... )
    ... ds_dir = unpack_seattle_2021_proto()
    ... summary_stats = summarize_annotation_statistics(ds_dir / "annotations")

    Raises
    ------
    FileNotFoundError
        The provided dataset directory was not found.
    FileNotFoundError
        The provided dataset directory did not contain any annotation files.
    NotADirectoryError
        THe provided dataset directory was a file and not a directory.
    """
    # Ensure dataset dir is valid
    annotations_dir = Path(annotations_dir).resolve(strict=True)
    if annotations_dir.is_file():
        raise NotADirectoryError(annotations_dir)

    # Add up the speaking time for all speakers found in every JSON
    # annotation file in the provided dataset directory
    speaker_lut: Dict[str, float] = {}
    n_events = 0
    for annotation_file in annotations_dir.glob("*.json"):
        # Open new annotation file
        with open(annotation_file, "r") as open_f:
            annotations = json.load(open_f)

        # Iter through each "monologue" section and add up times
        for monologue in annotations["monologues"]:
            speaker = monologue["speaker"]["id"]
            duration = monologue["end"] - monologue["start"]
            if speaker not in speaker_lut:
                speaker_lut[speaker] = duration
            else:
                speaker_lut[speaker] += duration

        n_events += 1

    # Compute summary stats
    total_speaker_time = sum([duration for speaker, duration in speaker_lut.items()])
    speaker_times: List[SpeakerTime] = [
        SpeakerTime(
            label=speaker,
            duration=duration,
            percent_of_total=duration / total_speaker_time,
        )
        for speaker, duration in speaker_lut.items()
    ]
    return DatasetSummary(
        n_speakers=len(speaker_times),
        n_events=n_events,
        speaker_times=speaker_times,
        mean_speaker_time=total_speaker_time / len(speaker_times),
        min_speaker_time=min(speaker_times, key=lambda st: st.duration),
        max_speaker_time=max(speaker_times, key=lambda st: st.duration),
        std_speaker_time=statistics.stdev([st.duration for st in speaker_times]),
        total_speaker_time=total_speaker_time,
    )


def expand_annotations_to_dataset(
    annotations_and_audios: List[AnnotationAndAudio],
    max_audio_chunk_duration: int = 5,
    audio_chunk_creation_scheme: str = "first-to-last",
) -> pd.DataFrame:
    """
    Expand a list of annotation and audio files into a full dataset to be used for
    training and testing a speaker classification model.

    Parameters
    ----------
    annotations_and_audios: List[AnnotationAndAudio]
        A list of annotation and their matching audio files to expand into a speaker,
        audio file path, start and end times.
    max_audio_chunk_duration: int
        Length of the audio duration to split larger audio files into.
        Default: 5 seconds
    audio_chunk_creation_scheme: str
        Function lookup for how to split audio files larger than the
        max_audio_chunk_duration.
        Default: "first-to-last"

    Returns
    -------
    dataset: pd.DataFrame
        The expanded dataset with columns: speaker, audio_file, start_time, end_time

    Notes
    -----
    This does not split larger audio files into smaller files.
    Rather, this function records the start and end times to split an audio
    file with during the training process. This is done as to not duplicate audio.
    """
    # Iter annotations and audios
    annotated_audios: List[AnnotatedAudio] = []
    for aaa in annotations_and_audios:
        # check length
        # construct start and end times
        # create aa for all start and end time pairs
        pass

    return pd.DataFrame([aa.to_dict() for aa in annotated_audios])
