#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from pyannote.audio import Pipeline
from pydub import AudioSegment
from sklearn.model_selection import train_test_split

from .types import AnnotatedAudio, GeckoAnnotationAndAudio

###############################################################################

log = logging.getLogger(__name__)

###############################################################################
# Annotation


def diarize_and_split_audio(
    audio_file: Union[str, Path],
    storage_dir: Optional[Union[str, Path]] = None,
    min_audio_chunk_duration: float = 0.5,
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
    min_audio_chunk_duration: float
        Length of the minimum audio duration to allow through after chunking.
        default: 0.5 seconds
    diarization_pipeline: Optional[Pipeline]
        A preloaded PyAnnote Pipeline.
        Default: None (load default)

    Returns
    -------
    storage_dir: Path
        The path to where all the chunked audio was stored.

    See Also
    --------
    expand_labeled_diarized_audio_dir_to_dataset
        After labeling the audio in the produced diarized audio directory, expand
        the labeled data into a dataset ready for training.

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

        # Only store chunk if duration is longer than param
        if (turn.end - turn.start) >= min_audio_chunk_duration:
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


###############################################################################
# Dataset expansions


def _chunk_audio(
    monologue_start_time: float,
    monologue_end_time: float,
    max_audio_chunk_duration: float,
    min_audio_chunk_duration: float,
    audio: AudioSegment,
    audio_output_dir: Path,
    chunk_filename_base: str,
    overwrite: bool,
    annotated_audios: List[AnnotatedAudio],
    conversation_id: str,
    label: str,
) -> List[AnnotatedAudio]:
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

        # Determine chunk end time
        # If start + chunk duration is longer than monologue
        # Chunk needs to be cut at monologue end
        if monologue_end_time > chunk_end:
            chunk_end_millis = chunk_end * 1000
        else:
            chunk_end_millis = monologue_end_time * 1000

        # Only allow chunk through if duration is greater than
        # min chunk length
        duration = (chunk_end_millis - chunk_start_millis) / 1000
        if duration >= min_audio_chunk_duration:
            # Get chunk
            chunk = audio[chunk_start_millis:chunk_end_millis]

            # Determine save pattern
            chunk_save_path = audio_output_dir / (
                f"{chunk_filename_base}-chunk_{chunk_i}.wav"
            )
            if chunk_save_path.exists() and not overwrite:
                raise FileExistsError(chunk_save_path)

            # Save audio chunk
            chunk.export(chunk_save_path, format="wav")

            # Append new chunk to list
            annotated_audios.append(
                AnnotatedAudio(
                    conversation_id=conversation_id,
                    label=label,
                    audio=str(chunk_save_path),
                    duration=duration,
                )
            )

    return annotated_audios


def expand_gecko_annotations_to_dataset(
    annotations_and_audios: List[GeckoAnnotationAndAudio],
    audio_output_dir: Union[str, Path] = Path("chunked-audio/"),
    overwrite: bool = False,
    min_audio_chunk_duration: float = 0.5,
    max_audio_chunk_duration: float = 2.0,
) -> pd.DataFrame:
    """
    Expand a list of annotation and audio files into a full dataset to be used for
    training and testing a speaker classification model.

    Parameters
    ----------
    annotations_and_audios: List[GeckoAnnotationAndAudio]
        A list of annotation and their matching audio files to expand into a speaker,
        audio file path, start and end times.
    audio_output_dir: Union[str, Path]
        A directory path to store the chunked audio files in.
        Default: "chunked-audio" directory in the current working directory.
    overwrite: bool
        When writting out an audio chunk, should existing files be overwritten.
        Default: False (do not overwrite)
    min_audio_chunk_duration: float
        Length of the minimum audio duration to allow through after chunking.
        default: 0.5 seconds
    max_audio_chunk_duration: float
        Length of the maximum audio duration to split larger audio files into.
        Default: 2.0 seconds

    Returns
    -------
    dataset: pd.DataFrame
        The expanded dataset with columns: label, audio, duration

    Raises
    ------
    NotADirectoryError
        A file exists at the specified destination.
    FileExistsError
        A file exists at the target chunk audio location but overwrite is False.

    Notes
    -----
    Generated and attached conversation ids are pulled from the annotation file name.
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

        # Iter through each "monologue" section
        for i, monologue in enumerate(annotations["monologues"]):
            monologue_speaker = monologue["speaker"]["id"].lower()
            monologue_start_time = monologue["start"]
            monologue_end_time = monologue["end"]

            # Construct basic filename
            filename_base = (
                f"{aaa.audio_file.with_suffix('').name}"
                f"-monologue_{i}-{monologue_speaker}"
            )

            # Chunk audios for this monologue
            annotated_audios = _chunk_audio(
                monologue_start_time=monologue_start_time,
                monologue_end_time=monologue_end_time,
                max_audio_chunk_duration=max_audio_chunk_duration,
                min_audio_chunk_duration=min_audio_chunk_duration,
                audio=audio,
                audio_output_dir=audio_output_dir,
                chunk_filename_base=filename_base,
                overwrite=overwrite,
                annotated_audios=annotated_audios,
                conversation_id=aaa.annotation_file.with_suffix("").name,
                label=monologue_speaker,
            )

        log.info(f"Completed expansion for annotation file: {aaa.annotation_file}")

    # Merge all into a single dataframe
    return pd.DataFrame([aa.to_dict() for aa in annotated_audios])  # type: ignore


def expand_labeled_diarized_audio_dir_to_dataset(
    labeled_diarized_audio_dir: Union[Union[str, Path], List[Union[str, Path]]],
    audio_output_dir: Union[str, Path] = Path("chunked-audio/"),
    overwrite: bool = False,
    min_audio_chunk_duration: float = 0.5,
    max_audio_chunk_duration: float = 2.0,
) -> pd.DataFrame:
    """
    Expand the provided labeled diarized audio into a dataset ready for training.

    Parameters
    ----------
    labeled_diarized_audio_dir: Union[Union[str, Path], List[Union[str, Path]]]
        A path or list of paths to diarization results directories. These directories
        should no longer have the "SPEAKER_00", "SPEAKER_01", default labeling but
        expert annotated labels.
    audio_output_dir: Union[str, Path]
        A directory path to store the chunked audio files in.
        Default: "chunked-audio" directory in the current working directory.
    overwrite: bool
        When writting out an audio chunk, should existing files be overwritten.
        Default: False (do not overwrite)
    min_audio_chunk_duration: float
        Length of the minimum audio duration to allow through after chunking.
        default: 0.5 seconds
    max_audio_chunk_duration: float
        Length of the maximum audio duration to split larger audio files into.
        Default: 2.0 seconds

    Returns
    -------
    dataset: pd.DataFrame
        The expanded dataset with columns: label, audio, duration

    Raises
    ------
    NotADirectoryError
        A file exists at the specified destination.
    FileExistsError
        A file exists at the target chunk audio location but overwrite is False.

    See Also
    --------
    diarize_and_split_audio
        Function to diarize an audio file and split into annotation directories.

    Notes
    -----
    The provided labeled diarized audio directory(s) should have the
    following structure::

        {labeled_diarized_audio_dir}/
        ├── label
        │   ├── 1.wav
        │   └── 2.wav
        ├── second_label
        │   ├── 1.wav
        │   └── 2.wav

    Generated and attached conversation ids are pulled from the labeled diarized audio
    directory names.
    """
    # Ensure dataset dir is valid
    audio_output_dir = Path(audio_output_dir).resolve()
    if audio_output_dir.is_file():
        raise NotADirectoryError(audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize annotation dir to list
    if isinstance(labeled_diarized_audio_dir, Path):
        labeled_diarized_audio_dir = [labeled_diarized_audio_dir]

    # Iter over all dia dirs
    annotated_audios: List[AnnotatedAudio] = []
    for anno_dir in labeled_diarized_audio_dir:
        # Convert to path
        anno_dir = Path(anno_dir).resolve(strict=True)

        # Get the conversation id
        conversation_id = anno_dir.name

        # Iter over all labels
        for label_dir in anno_dir.iterdir():
            if label_dir.is_dir():
                for i, audio_file in enumerate(label_dir.glob("*.wav")):
                    audio = AudioSegment.from_file(audio_file)
                    monologue_speaker = label_dir.name

                    # Construct basic filename
                    filename_base = (
                        f"{audio_file.with_suffix('').name}"
                        f"-monologue_{i}-{monologue_speaker}"
                    )

                    # Chunk audios for this monologue
                    annotated_audios = _chunk_audio(
                        monologue_start_time=0,
                        monologue_end_time=audio.duration_seconds,
                        max_audio_chunk_duration=max_audio_chunk_duration,
                        min_audio_chunk_duration=min_audio_chunk_duration,
                        audio=audio,
                        audio_output_dir=audio_output_dir,
                        chunk_filename_base=filename_base,
                        overwrite=overwrite,
                        annotated_audios=annotated_audios,
                        conversation_id=conversation_id,
                        label=monologue_speaker.lower(),
                    )

                log.info(f"Completed expansion for diarized audio dir: {label_dir}")

    # Merge all into a single dataframe
    return pd.DataFrame([aa.to_dict() for aa in annotated_audios])  # type: ignore


###############################################################################
# Data expections and cleaning


def check_and_create_dataset(
    dataset: pd.DataFrame,
    equalize_data: bool = True,
    test_and_valid_size: float = 0.4,
) -> DatasetDict:
    # partition to equal amounts of data per label
    # partition to splits with different meetings
    if equalize_data:
        groups = dataset.groupby("label")
        dataset = pd.DataFrame(
            groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True))
        )

    # Holdout by conversation id
    labels = set(dataset.label.unique())
    conversation_ids = dataset.conversation_id.unique()

    # Reroll holdouts until all labels are present in each split
    all_labels_present = False
    iters = 0
    while not all_labels_present:
        # Get conversation ids split randomly
        train_ids, test_and_valid_ids = train_test_split(
            conversation_ids, test_size=test_and_valid_size
        )
        test_ids, valid_ids = train_test_split(test_and_valid_ids, test_size=0.5)

        # Check the produced subsets
        train_ds = dataset.loc[dataset.conversation_id.isin(train_ids)]
        test_ds = dataset.loc[dataset.conversation_id.isin(test_ids)]
        valid_ds = dataset.loc[dataset.conversation_id.isin(valid_ids)]
        print(train_ds.conversation_id.unique())
        print(train_ds.label.unique(), len(train_ds.label.unique()))
        print(test_ds.conversation_id.unique())
        print(test_ds.label.unique(), len(test_ds.label.unique()))
        print(valid_ds.conversation_id.unique())
        print(valid_ds.label.unique(), len(valid_ds.label.unique()))
        if (
            set(train_ds.label.unique()) == labels
            and set(test_ds.label.unique()) == labels
            and set(valid_ds.label.unique()) == labels
        ):
            all_labels_present = True

        iters += 1
        print(f"attempting train test split construction {iters} times.")
        if iters == 5:
            raise ValueError(
                "Could not construct dataset holdouts from conversation ids while "
                "stratifying by label."
            )
        print("-" * 80)

    return train_ds, test_ds, valid_ds