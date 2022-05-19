#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import fireo
from cdp_backend.database import models as db_models
from gcsfs import GCSFileSystem
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client
from tqdm import tqdm

from ..types import GeckoAnnotationAndAudio
from ..utils import _unpack_zip

###############################################################################

DATASETS_DIR = (Path(__file__).parent / "data").resolve(strict=True)
SEATTLE_2021_PROTOTYPE_DATASET = "seattle-2021-proto.zip"
SEATTLE_2021_PROTOTYPE_DATASET_DIR = Path(SEATTLE_2021_PROTOTYPE_DATASET).with_suffix(
    ""
)
SEATTLE_2021_PROTOTYPE_ANNOTATIONS_DIR = (
    SEATTLE_2021_PROTOTYPE_DATASET_DIR / "annotations"
)
SEATTLE_2021_PROTOTYPE_TRANSCRIPT_DIR = (
    SEATTLE_2021_PROTOTYPE_DATASET_DIR / "unlabeled_transcripts"
)
SEATTLE_2021_PROTOTYPE_AUDIO_DIR = SEATTLE_2021_PROTOTYPE_DATASET_DIR / "audio"
SEATTLE_2021_INFRA_STR = "cdp-seattle-staging-dbengvtn"

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def unpack(
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


def _get_matching_unlabelled_transcript_and_audio(
    annotation_file: Path,
    transcript_output_dir: Path,
    audio_output_dir: Path,
    fs: GCSFileSystem,
    overwrite: bool,
    progress_bar: Optional[tqdm] = None,
) -> Optional[GeckoAnnotationAndAudio]:
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
        log.debug(
            f"Multiple sessions found for event: {event_id}, "
            f"dropping annotations from dataset."
        )
        return None

    # Find any transcript name for this session and pull the content hash
    matching_session = matching_sessions[0]
    matching_transcript = db_models.Transcript.collection.filter(
        "session_ref", "==", matching_session.key
    ).get()
    matching_transcript_file = matching_transcript.file_ref.get()
    content_hash = matching_transcript_file.name.split("-")[0]

    # Check overwrite or skip
    transcript_save_path = transcript_output_dir / f"{event_id}.json"
    audio_save_path = audio_output_dir / f"{event_id}.wav"
    remote_audio_uri = (
        f"gs://{SEATTLE_2021_INFRA_STR}.appspot.com/{content_hash}-audio.wav"
    )

    # Transcript
    if transcript_save_path.exists():
        if overwrite:
            fs.get(matching_transcript_file.uri, str(transcript_save_path))
        else:
            log.debug(
                f"Using existing transcript file found for event: {event_id}, "
                f"({transcript_save_path})."
            )
    else:
        fs.get(matching_transcript_file.uri, str(transcript_save_path))

    # Audio
    if audio_save_path.exists():
        if overwrite:
            fs.get(remote_audio_uri, str(audio_save_path))
        else:
            log.debug(
                f"Using existing audio file found for event: {event_id}, "
                f"({audio_save_path})."
            )
    else:
        fs.get(remote_audio_uri, str(audio_save_path))

    # Update progress
    if progress_bar:
        progress_bar.update()

    return GeckoAnnotationAndAudio(
        annotation_file=annotation_file,
        audio_file=audio_save_path,
    )


def pull_all_files(
    annotations_dir: Union[str, Path] = SEATTLE_2021_PROTOTYPE_ANNOTATIONS_DIR,
    transcript_output_dir: Union[str, Path] = SEATTLE_2021_PROTOTYPE_TRANSCRIPT_DIR,
    audio_output_dir: Union[str, Path] = SEATTLE_2021_PROTOTYPE_AUDIO_DIR,
    overwrite: bool = False,
) -> List[GeckoAnnotationAndAudio]:
    """
    Using the filenames for the files found in the annotations directory as event
    ids, pull the matching audio files and store them in the dataset directory.

    Parameters
    ----------
    annotations_dir: Union[str, Path]
        The path to the directory which contains the annotation files for a dataset.
        Default: Default Seattle 2021 Prototype Dataset Annotations Directory
    transcript_output_dir: Union[str, Path]
        The path to where to store the pulled unlabeled transcript files.
        Default: Default Seattle 2021 Prototype Dataset Transcript Directory
    audio_output_dir: Union[str, Path]
        The path to where to store the pulled audio files.
        Default: Default Seattle 2021 Prototype Dataset Audio Directory
    overwrite: bool
        If an audio file with a filename matching the event id already exists in the
        output directory exists, should it be overwritten (safer) or skipped (faster).
        Default: False (skipped)

    Returns
    -------
    annotations_and_audios: List[GeckoAnnotationAndAudio]
        A list of matching GeckoAnnotationAndAudio objects.

    Raises
    ------
    NotADirectoryError
        Provided annotations directory is not a directory.
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
    transcript_output_dir = Path(transcript_output_dir).resolve()
    audio_output_dir = Path(audio_output_dir).resolve()
    if annotations_dir.is_file():
        raise NotADirectoryError(annotations_dir)
    if transcript_output_dir.exists() and transcript_output_dir.is_file():
        raise NotADirectoryError(transcript_output_dir)
    if audio_output_dir.exists() and audio_output_dir.is_file():
        raise NotADirectoryError(audio_output_dir)

    # Make output dirs
    transcript_output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Create connection to DB and FS
    fireo.connection(
        client=Client(
            project=SEATTLE_2021_INFRA_STR, credentials=AnonymousCredentials()
        )
    )
    fs = GCSFileSystem(project=SEATTLE_2021_INFRA_STR, token="anon")

    # Get filepaths used for pulling extra files
    annotation_files = list(annotations_dir.glob("*.json"))

    # Create progress bar
    pbar = tqdm(total=len(annotation_files))

    # Create partial function for threading
    downloader = partial(
        _get_matching_unlabelled_transcript_and_audio,
        transcript_output_dir=transcript_output_dir,
        audio_output_dir=audio_output_dir,
        fs=fs,
        overwrite=overwrite,
        progress_bar=pbar,
    )

    # Thread audio file downloading
    with ThreadPoolExecutor() as exe:
        results = list(exe.map(downloader, annotation_files))

    # Filter out bad events
    return [r for r in results if isinstance(r, GeckoAnnotationAndAudio)]
