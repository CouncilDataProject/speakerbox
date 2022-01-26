#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import fireo
import pandas as pd
from cdp_backend.database import models as db_models
from gcsfs import GCSFileSystem
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client

###############################################################################

DATASETS_DIR = (Path(__file__).parent / "data").resolve(strict=True)
SEATTLE_2021_PROTOTYPE_DATASET = "seattle-2021-proto.zip"
SEATTLE_2021_PROTOTYPE_DATASET_DIR = Path(SEATTLE_2021_PROTOTYPE_DATASET).with_suffix(
    ""
)
SEATTLE_2021_PROTOTYPE_ANNOTATIONS_DIR = (
    SEATTLE_2021_PROTOTYPE_DATASET_DIR / "annotations"
)
SEATTLE_2021_PROTOTYPE_AUDIO_DIR = SEATTLE_2021_PROTOTYPE_DATASET_DIR / "audio"
SEATTLE_2021_INFRA_STR = "cdp-seattle-staging-dbengvtn"

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
    dest: Union[str, Path] = SEATTLE_2021_PROTOTYPE_DATASET_DIR,
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


class AnnotationAndAudio(NamedTuple):
    annotation_file: Path
    audio_file: Path


def _get_matching_audio_for_seattle_2021_proto_annotation_file(
    annotation_file: Path,
    audio_output_dir: Path,
    fs: GCSFileSystem,
    overwrite: bool,
) -> Optional[AnnotationAndAudio]:
    # Remove suffix and use just the file name for the event id
    event_id = annotation_file.with_suffix("").name

    # Pull sessions with matching event id
    matching_sessions = list(
        db_models.Session.collection.filter(
            "event_ref", "==", f"{db_models.Event.collection_name}/{event_id}"
        ).fetch()
    )

    # There should only be a single session so drop this event if more than one
    if len(matching_sessions) > 1:
        return None

    # Find any transcript name for this session and pull the content hash
    matching_session = matching_sessions[0]
    matching_transcript = db_models.Transcript.collection.filter(
        "session_ref", "==", matching_session.key
    ).get()
    matching_transcript_file = matching_transcript.file_ref.get()
    content_hash = matching_transcript_file.name.split("-")[0]

    # Check overwrite or skip
    audio_save_path = audio_output_dir / f"{event_id}.wav"
    if audio_save_path.exists():
        if overwrite:
            fs.get(
                f"gs://{SEATTLE_2021_INFRA_STR}.appspot.com/{content_hash}-audio.wav",
                str(audio_save_path),
            )
    else:
        fs.get(
            f"gs://{SEATTLE_2021_INFRA_STR}.appspot.com/{content_hash}-audio.wav",
            str(audio_save_path),
        )

    return AnnotationAndAudio(
        annotation_file=annotation_file,
        audio_file=audio_save_path,
    )


def pull_seattle_2021_proto_audio(
    annotations_dir: Union[str, Path] = SEATTLE_2021_PROTOTYPE_ANNOTATIONS_DIR,
    audio_output_dir: Union[str, Path] = SEATTLE_2021_PROTOTYPE_AUDIO_DIR,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Using the filenames for the files found in the annotations directory as event
    ids, pull the matching audio files and store them in the dataset directory.

    Parameters
    ----------
    annotations_dir: Union[str, Path]
        The path to the directory which contains the annotation files for a dataset.
        Default: Default Seattle 2021 Prototype Dataset Annotations Directory
    audio_output_dir: Union[str, Path]
        The path to where to store the pulled audio files
        Default: Default Seattle 2021 Prototype Dataset Audio Directory
    overwrite: bool
        If an audio file with a filename matching the event id already exists in the
        output directory exists, should it be overwritten (safer) or skipped (faster).
        Default: False (skipped)

    Returns
    -------
    full_dataset: pd.DataFrame
        A minimal dataframe with columns for the annotation file and the
        matching audio file.

    Raises
    ------
    NotADirectoryError
        Provided annotations_dir is not a directory.
    NotADirectoryError
        A file exists at the specified audio output directory.

    Examples
    --------
    >>> from speakerbox.datasets import (
    ...     unpack_seattle_2021_proto,
    ...     pull_seattle_2021_proto_audio,
    ... )
    ... ds_dir = unpack_seattle_2021_proto()
    ... ds = pull_seattle_2021_proto_audio()
    """
    # Ensure dataset dir is valid
    annotations_dir = Path(annotations_dir).resolve(strict=True)
    audio_output_dir = Path(audio_output_dir).resolve()
    if annotations_dir.is_file():
        raise NotADirectoryError(annotations_dir)
    if audio_output_dir.exists() and audio_output_dir.is_file():
        raise NotADirectoryError(audio_output_dir)

    # Make output dir
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Create connection to DB and FS
    fireo.connection(
        client=Client(
            project=SEATTLE_2021_INFRA_STR, credentials=AnonymousCredentials()
        )
    )
    fs = GCSFileSystem(project=SEATTLE_2021_INFRA_STR, token="anon")

    # Create partial function for threading
    downloader = partial(
        _get_matching_audio_for_seattle_2021_proto_annotation_file,
        audio_output_dir=audio_output_dir,
        fs=fs,
        overwrite=overwrite,
    )

    # Thread audio file downloading
    with ThreadPoolExecutor() as exe:
        downloads = list(exe.map(downloader, annotations_dir.glob("*.json")))

    # Unpack results
    results: List[Dict[str, str]] = []
    for annotation_and_audio in downloads:
        if annotation_and_audio is not None:
            results.append(
                {
                    "annotations": str(annotation_and_audio.annotation_file),
                    "audio": str(annotation_and_audio.audio_file),
                }
            )

    return pd.DataFrame(results)


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
