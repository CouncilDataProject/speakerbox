#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

###############################################################################

DATASETS_DIR = (Path(__file__).parent / "data").resolve(strict=True)
SEATTLE_2021_PROTOTYPE_DATASET = "seattle-2021-proto.zip"

###############################################################################
# Dataset Utilities


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


def unpack_seattle_2021_proto(
    dest: Union[str, Path] = Path(SEATTLE_2021_PROTOTYPE_DATASET).with_suffix(""),
    clean: bool = False,
) -> Path:
    """
    Unpacks the Seattle 2021 Prototype Dataset from a zipfile to the provided
    destination location.

    Parameters
    ----------
    dest: Union[str, Path]
        The destination to unpack to.
        Default: A new directory in your current working directory with the standard
        Seattle 2021 Prototype Dataset name.
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
    return _unpack_zip(
        zipfile=DATASETS_DIR / SEATTLE_2021_PROTOTYPE_DATASET,
        dest=dest,
        clean=clean,
    )


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
    dataset: Union[str, Path],
) -> DatasetSummary:
    """
    Summarize all annotation files into a single pandas DataFrame.

    Parameters
    ----------
    dataset: Union[str, Path]
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
    dataset = Path(dataset).resolve(strict=True)
    if dataset.is_file():
        raise NotADirectoryError(dataset)

    # Add up the speaking time for all speakers found in every JSON
    # annotation file in the provided dataset directory
    speaker_lut: Dict[str, float] = {}
    n_events = 0
    for annotation_file in dataset.glob("*.json"):
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
