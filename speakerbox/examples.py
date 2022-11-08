#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

from .main import eval_model, train
from .preprocess import prepare_dataset
from .utils import set_global_seed

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

SAMPLE_SIZE_LUT = {
    "15-minutes": (15 * 60) // 2,
    "30-minutes": (30 * 60) // 2,
    "60-minutes": None,
}

###############################################################################


@dataclass
class ModelEvalScores(DataClassJsonMixin):
    accuracy: float
    precision: float
    recall: float
    duration: float


@dataclass
class IteratedModelEvalScores(DataClassJsonMixin):
    dataset_size: str
    equalized_data: bool
    mean_audio_per_person_train: float
    std_audio_per_person_train: float
    mean_audio_per_person_test: float
    std_audio_per_person_test: float
    mean_audio_per_person_valid: float
    std_audio_per_person_valid: float
    mean_accuracy: float
    std_accuracy: float
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_duration: float
    std_duration: float


###############################################################################


def train_and_eval_example_model(
    example_dataset_dir: Union[str, Path],
    dataset_size_str: Literal["15-minutes", "30-minutes", "60-minutes"],
    n_interations: int = 5,
    seed: int = 182318512,
    equalize_data_within_splits: bool = False,
) -> IteratedModelEvalScores:
    """
    Train and evaluate a model multiple times for one of the dataset sizes.

    This was used to investigate the diminishing return of adding more data
    to the model.

    Parameters
    ----------
    example_dataset_dir: Union[str, Path]
        Path to the downloaded and unzipped example dataset.
    dataset_size_str: Literal["15-minutes", "30-minutes", "60-minutes"]
        The dataset size to choose from. This will load (and potentially)
        subset the packaged data.
    n_iterations: int
        The number of train and evaluation iterations to try for this model
        before averaging them all.
        Default: 5
    seed: int
        A random seed to set global random state.
    equalize_data_within_splits: bool
        Should the data splits be equalized to the smallest number of examples
        for any speaker in that split.
        Default: False (allow different amounts of examples per label)

    Returns
    -------
    IteratedModelEvalScores
        The average accuracy, precision, recall, and duration over the
        training and evaluation iterations.
    """
    set_global_seed(seed)

    # Handle paths
    example_dataset_dir = Path(example_dataset_dir).expanduser().resolve()

    # Open example training data
    example_training_data = pd.read_parquet(example_dataset_dir / "dataset.parquet")

    # Update all audio paths to be fully resolved
    example_training_data["audio"] = example_training_data["audio"].apply(
        lambda audio_filename: str(example_dataset_dir / audio_filename)
    )

    # Get requested sample
    sample_size_requested = SAMPLE_SIZE_LUT[dataset_size_str]
    if sample_size_requested is None:
        selected_data = example_training_data
    else:
        selected_data = example_training_data.sample(n=sample_size_requested)

    # Prepare data for training
    dataset, counts = prepare_dataset(
        selected_data,
        equalize_data_within_splits=equalize_data_within_splits,
    )
    log.info(f"Selected sampled data:\n{counts}")

    # Store the durations of audio per-person
    audio_durations_per_person_train = []
    audio_durations_per_person_test = []
    audio_durations_per_person_valid = []
    for _, row in counts.iterrows():
        # Assume max chunk duration of 2.0 which is the default chunk duration
        audio_durations_per_person_train.append(row["train_counts"] * 2.0)
        audio_durations_per_person_test.append(row["test_counts"] * 2.0)
        audio_durations_per_person_valid.append(row["valid_counts"] * 2.0)

    # Train
    eval_results = []
    for i in tqdm(range(n_interations)):
        model_name = (
            f"trained-example"
            f"-{dataset_size_str}"
            f"-equalized-{equalize_data_within_splits}"
            f"-iter-{i}"
        ).lower()
        start_time = time.time()
        model = train(dataset, model_name)
        duration = time.time() - start_time
        acc, pre, rec, _ = eval_model(dataset["valid"], str(model))
        eval_results.append(
            ModelEvalScores(
                accuracy=acc,
                precision=pre,
                recall=rec,
                duration=duration,
            )
        )

    # Average all results
    return IteratedModelEvalScores(
        dataset_size=dataset_size_str,
        equalized_data=equalize_data_within_splits,
        mean_audio_per_person_train=np.mean(audio_durations_per_person_train),
        std_audio_per_person_train=np.std(audio_durations_per_person_train),
        mean_audio_per_person_test=np.mean(audio_durations_per_person_test),
        std_audio_per_person_test=np.std(audio_durations_per_person_test),
        mean_audio_per_person_valid=np.mean(audio_durations_per_person_valid),
        std_audio_per_person_valid=np.std(audio_durations_per_person_valid),
        mean_accuracy=np.mean([r.accuracy for r in eval_results]),
        std_accuracy=np.std([r.accuracy for r in eval_results]),
        mean_precision=np.mean([r.precision for r in eval_results]),
        std_precision=np.std([r.precision for r in eval_results]),
        mean_recall=np.mean([r.recall for r in eval_results]),
        std_recall=np.std([r.recall for r in eval_results]),
        mean_duration=np.mean([r.duration for r in eval_results]),
        std_duration=np.std([r.duration for r in eval_results]),
    )


def train_and_eval_all_example_models(
    example_dataset_dir: Union[str, Path],
    n_iterations: int = 5,
    seed: int = 182318512,
    equalize_data_within_splits: bool = False,
) -> pd.DataFrame:
    """
    Train and evaluate a model multiple times for each of the dataset sizes.

    This was used to investigate the diminishing return of adding more data
    to the model.

    Parameters
    ----------
    example_dataset_dir: Union[str, Path]
        Path to the downloaded and unzipped example dataset.
    n_iterations: int
        The number of train and evaluation iterations to try for this model
        before averaging them all.
        Default: 5
    seed: int
        A random seed to set global random state.
    equalize_data_within_splits: bool
        Should the data splits be equalized to the smallest number of examples
        for any speaker in that split.
        Default: False (allow different amounts of examples per label)

    Returns
    -------
    pd.DataFrame
        A DataFrame of results for all the models tested.

    See Also
    --------
    train_and_eval_example_model
        The function used to train and eval a single model dataset size.
    """
    results = []
    for dataset_size_str in SAMPLE_SIZE_LUT.keys():
        results.append(
            train_and_eval_example_model(
                example_dataset_dir=example_dataset_dir,
                dataset_size_str=dataset_size_str,  # type: ignore
                n_interations=n_iterations,
                seed=seed,
                equalize_data_within_splits=equalize_data_within_splits,
            ).to_dict()
        )

    # Merge to DataFrame, save, and store selected results
    results_df = pd.DataFrame(results)

    # Store to parquet
    results_df.to_parquet("example-models-grid-eval-scores.parquet")

    # Store to markdown (for paper)
    results_df_for_render = results_df.set_index("dataset_size")
    results_df_for_render = results_df_for_render.drop(
        columns=[
            "equalized_data",
            "mean_audio_per_person_train",
            "std_audio_per_person_train",
            "mean_audio_per_person_test",
            "std_audio_per_person_test",
            "mean_audio_per_person_valid",
            "std_audio_per_person_valid",
        ]
    )

    # Make render results rounded and formatted
    results_df_for_render["mean_accuracy"] = results_df_for_render.apply(
        lambda r: (f"{round(r.mean_accuracy, 3)} " f"± {round(r.std_accuracy, 3)}"),
        axis=1,
    )
    results_df_for_render["mean_precision"] = results_df_for_render.apply(
        lambda r: (f"{round(r.mean_precision, 3)} " f"± {round(r.std_precision, 3)}"),
        axis=1,
    )
    results_df_for_render["mean_recall"] = results_df_for_render.apply(
        lambda r: (f"{round(r.mean_recall, 3)} " f"± {round(r.std_recall, 3)}"),
        axis=1,
    )
    results_df_for_render[
        "mean_training_duration_seconds"
    ] = results_df_for_render.apply(
        lambda r: (f"{round(r.mean_duration)} " f"± {round(r.std_duration)}"),
        axis=1,
    )
    results_df_for_render = results_df_for_render.drop(
        columns=[
            "std_accuracy",
            "std_precision",
            "std_recall",
            "mean_duration",
            "std_duration",
        ],
    )

    try:
        with open("example-models-grid-eval-scores.md", "w") as open_f:
            results_df_for_render.to_markdown(open_f)
    except ImportError:
        log.error(
            "Skipped generating the markdown table of the example "
            "model results because `tabulate` is not installed."
        )

    return results_df
