#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from pytest import Config

from speakerbox import eval_model, preprocess, train
from speakerbox.utils import set_global_seed

###############################################################################


def test_train(data_dir: Path, pytestconfig: Config) -> None:
    """
    This is a smoke test. We do not check the validity of the produced model,
    we simply care that the functions run through without error.

    Run with CPU with pytest:
    pytest speakerbox/tests/test_train.py --cpu

    Or with tox:
    tox -- --cpu

    On CI this test will skip
    tox -- --ci
    """
    if pytestconfig.getoption("ci"):
        return

    # Set global seed for some level of reproducibility across tests
    set_global_seed(20220421)

    # Get diarized dirs
    diarized_audio = data_dir / "diarized"
    diarized_audio_dirs = [
        dia_convo_dir
        for dia_convo_dir in diarized_audio.iterdir()
        if dia_convo_dir.is_dir()
    ]

    # Expand diarizations to full dataset
    expanded_dataset = preprocess.expand_labeled_diarized_audio_dir_to_dataset(
        labeled_diarized_audio_dir=diarized_audio_dirs,
        audio_output_dir="test-outputs/training/chunked-audio/",
        overwrite=True,
    )

    # Prepare and check
    dataset_dict, _ = preprocess.prepare_dataset(
        expanded_dataset,
        equalize_data_within_splits=True,
    )

    # Store prepared dataset to disk
    dataset_dict.save_to_disk("test-outputs/training/prepared-dataset/")

    # Train
    train(
        dataset_dict,
        model_name="test-outputs/trained-speakerbox",
        use_cpu=pytestconfig.getoption("use_cpu"),
    )

    # Eval
    eval_model(
        dataset_dict["valid"],
        model_name="test-outputs/trained-speakerbox",
    )
