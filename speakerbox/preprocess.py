#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from pyannote.audio import Pipeline
from pydub import AudioSegment
from sklearn.model_selection import train_test_split

from .types import AnnotatedAudio, GeckoAnnotationAndAudio
from .utils import set_global_seed

###############################################################################

log = logging.getLogger(__name__)

###############################################################################
# Annotation


def diarize_and_split_audio(
    audio_file: Union[str, Path],
    storage_dir: Optional[Union[str, Path]] = None,
    min_audio_chunk_duration: float = 0.5,
    diarization_pipeline: Optional[Pipeline] = None,
    seed: Optional[int] = None,
    hf_token: Optional[str] = None,
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
    seed: Optional[int]
        Seed to pass to torch, numpy, and Python RNGs.
        Default: None (do not set a seed)
    hf_token: Optional[str]
        Huggingface user access token to download the diarization model.
        Can also be set with the HUGGINGFACE_TOKEN environment variable.
        https://hf.co/settings/tokens

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
    Prior to using this function you need to accept user conditions:
    https://hf.co/pyannote/speaker-diarization and https://hf.co/pyannote/segmentation

    The output directory structure of the produced chunks will follow the pattern::

        {storage_dir}/
        ├── SPEAKER_00
        │   ├── {start_time_millis}-{start_end_millis}.wav
        │   └── {start_time_millis}-{start_end_millis}.wav
        ├── SPEAKER_01
        │   ├── {start_time_millis}-{start_end_millis}.wav
        │   └── {start_time_millis}-{start_end_millis}.wav
    """
    if seed:
        set_global_seed(seed)

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
        storage_dir = Path(audio_file_name)

    # Make storage dir
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Get token
    if hf_token:
        use_token = hf_token
    else:
        if "HUGGINGFACE_TOKEN" in os.environ:
            use_token = os.environ["HUGGINGFACE_TOKEN"]
        else:
            raise ValueError(
                "Must provide Huggingface token to download diarization model. "
                "Tokens can be created at: https://hf.co/settings/tokens "
                "and can be provided either via `hf_token` parameter or "
                "the `HUGGINGFACE_TOKEN` environment variable."
            )

    # Load pipeline
    if diarization_pipeline is None:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=use_token,
        )

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
        The expanded dataset with columns: conversation_id, label, audio, duration

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

        log.debug(f"Completed expansion for annotation file: {aaa.annotation_file}")

    # Merge all into a single dataframe
    return pd.DataFrame([aa.to_dict() for aa in annotated_audios])  # type: ignore


def expand_labeled_diarized_audio_dir_to_dataset(
    labeled_diarized_audio_dir: Union[
        str,
        Path,
        List[str],
        List[Path],
        List[Union[str, Path]],
    ],
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
        The expanded dataset with columns: conversation_id, label, audio, duration

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

                log.debug(f"Completed expansion for diarized audio dir: {label_dir}")

    # Merge all into a single dataframe
    return pd.DataFrame([aa.to_dict() for aa in annotated_audios])  # type: ignore


###############################################################################
# Data expections and cleaning


def prepare_dataset(
    dataset: pd.DataFrame,
    test_and_valid_size: float = 0.4,
    equalize_data_within_splits: bool = False,
    n_iterations: int = 100,
    seed: Optional[int] = None,
) -> Tuple[DatasetDict, pd.DataFrame]:
    """
    Prepare a dataset for training a new speakerbox / audio-classification model.
    This function attempts to randomly create train, test, and validation splits
    from the provided dataframe that meet the following two conditions:

    1. There is data holdout by conversation_id. I.e. if the dataset contains data from
    nine unique conversation ids, the training, test, and validation sets should all
    have different conversation ids (train has 0, 1, 2, 3; test has 4, 5, 6; validation
    has 7, 8).

    2. There is data stratification by label. I.e. if the dataset contains nine unique
    labels, the training, test, and validation sets should each have all nine labels
    present (train, test, and validation all have labels 0-8).

    Parameters
    ----------
    dataset: pd.DataFrame
        An expanded dataset with columns: conversation_id, label, audio, duration
    test_and_valid_size: float
        How much of the dataset to use for the combined test and validation sets
        as a percent (i.e. 0.4 = 40% of the dataset).
        The test and validation sets will further split this in half (i.e. 0.4 = 40%
        which means 20% of the total data for testing and 20% of the total data for
        validation).
    equalize_data_within_splits: bool
        After finding valid train, test, and validation splits, should the data within
        each split be reduced to have an equal number of data for each label.
        Default: False (Do not equalize labels within splits)
    n_iterations: int
        The number of iterations to attempt to find viable train, test, and validation
        sets.
        Default: 100
    seed: Optional[int]
        Seed to pass to torch, numpy, and Python RNGs.
        Default: None (do not set a seed)

    Returns
    -------
    dataset: DatasetDict
        The prepared dataset split into train, test, and validation splits.
    value_counts: pd.DataFrame
        A value count table where each row is a different label and each column is the
        count of that label in the matching train, test, or validation set.

    See Also
    --------
    expand_labeled_diarized_audio_dir_to_dataset
        Function to move from a directory of diarized audio (or multiple) into a
        dataset to provide to this function.
    expand_gecko_annotations_to_dataset
        Function to move from a gecko annotation file (or multiple) into a dataset
        to provide to this function.

    Raises
    ------
    ValueError
        Could not find train, test, and validation sets that meet the holdout and
        stratification criteria after n iterations. Recommended to annotate more data.
    """
    # Handle random seed
    if seed:
        set_global_seed(seed)

    # Holdout by conversation id
    labels = set(dataset.label.unique())
    conversation_ids = dataset.conversation_id.unique()

    # Reroll holdouts until all labels are present in each split
    all_labels_present = False
    iters = 0
    log.info("Finding data splits for provided dataset")
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
        if (
            set(train_ds.label.unique()) == labels
            and set(test_ds.label.unique()) == labels
            and set(valid_ds.label.unique()) == labels
        ):
            all_labels_present = True

        iters += 1
        if iters == n_iterations:
            raise ValueError(
                f"Could not find train, test, and validation sets that meet the "
                f"holdout and stratification criteria after {n_iterations} "
                f"sampling iterations. Recommended to annotate more data."
            )
        log.debug(f"Attempted train test validation split construction {iters} times.")

    log.debug(f"train_ds conversation_ids: {train_ds.conversation_id.unique()}")
    log.debug(f"test_ds conversation_ids: {test_ds.conversation_id.unique()}")
    log.debug(f"valid_ds conversation_ids: {valid_ds.conversation_id.unique()}")

    # Drop extra columns
    train_ds = train_ds.drop(["conversation_id", "duration"], axis=1)
    test_ds = test_ds.drop(["conversation_id", "duration"], axis=1)
    valid_ds = valid_ds.drop(["conversation_id", "duration"], axis=1)

    # Handle equalization
    if equalize_data_within_splits:
        # Subsets should be balanced by each label
        # Group by label and get random sample with the min of all labels
        train_groups = train_ds.groupby("label")
        train_ds = pd.DataFrame(
            train_groups.apply(
                lambda x: x.sample(train_groups.size().min()).reset_index(drop=True)
            )
        )
        test_groups = test_ds.groupby("label")
        test_ds = pd.DataFrame(
            test_groups.apply(
                lambda x: x.sample(test_groups.size().min()).reset_index(drop=True)
            )
        )
        valid_groups = valid_ds.groupby("label")
        valid_ds = pd.DataFrame(
            valid_groups.apply(
                lambda x: x.sample(valid_groups.size().min()).reset_index(drop=True)
            )
        )

    # Construct summary table
    value_counts = pd.DataFrame(
        {
            "train_counts": train_ds.label.value_counts(),
            "test_counts": test_ds.label.value_counts(),
            "valid_counts": valid_ds.label.value_counts(),
        }
    )
    log.info(f"Constructed train, test, validation sets contain:\n{value_counts}")

    # Construct DatasetDict
    train_ds = Dataset.from_pandas(train_ds, preserve_index=False)
    train_ds = train_ds.class_encode_column("label")
    test_ds = Dataset.from_pandas(test_ds, preserve_index=False)
    test_ds = test_ds.class_encode_column("label")
    valid_ds = Dataset.from_pandas(valid_ds, preserve_index=False)
    valid_ds = valid_ds.class_encode_column("label")

    # Return both the dataset dict and the value counts
    return (
        DatasetDict(
            {
                "train": train_ds,
                "test": test_ds,
                "valid": valid_ds,
            }
        ),
        value_counts,
    )
