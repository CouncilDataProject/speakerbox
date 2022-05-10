#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from speakerbox import preprocess, train

###############################################################################


def test_train(data_dir: Path) -> None:
    """
    This is a smoke test. We do not check the validity of the produced model,
    we simply care that the functions run through without error.
    """
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
        equalize_data=False,
    )

    # Train
    train(
        dataset_dict,
        model_name="test-outputs/trained-speakerbox",
        seed=20220421,
    )
