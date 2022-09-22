#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import pandas as pd

from speakerbox import eval_model, preprocess, train
from speakerbox.datasets import seattle_2021_proto

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

# Pull / prep original Seattle data
seattle_2021_ds_items = seattle_2021_proto.pull_all_files(
    annotations_dir="training-data/seattle-2021-proto/annotations/",
    transcript_output_dir="training-data/seattle-2021-proto/unlabeled_transcripts/",
    audio_output_dir="training-data/seattle-2021-proto/audio/",
)
seattle_2021_ds = preprocess.expand_gecko_annotations_to_dataset(
    seattle_2021_ds_items,
    overwrite=True,
)

# Expand diarized data
diarized_ds = ds = preprocess.expand_labeled_diarized_audio_dir_to_dataset(
    [
        "training-data/diarized/01e7f8bb1c03/",
        "training-data/diarized/2cdf68ae3c2c/",
        "training-data/diarized/6d6702d7b820/",
        "training-data/diarized/9f55f22d8e61/",
        "training-data/diarized/9f581faa5ece/",
    ],
    overwrite=True,
)

# Combine into single
combined_ds = pd.concat([seattle_2021_ds, diarized_ds], ignore_index=True)

# Generate train test validate splits
dataset, _ = preprocess.prepare_dataset(
    combined_ds,
    equalize_data_within_splits=False,
)

# dataset.save_to_disk(SOME_PATH)

# Train a model
model_path = train(dataset)
eval_model(dataset["valid"])
