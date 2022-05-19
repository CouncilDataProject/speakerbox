#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import torch
from cdp_data import CDPInstances, datasets

from speakerbox.preprocess import diarize_and_split_audio

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

# Pull specific meetings
for start_date, end_date in [
    ("2021-05-24", "2021-05-25"),
    ("2021-06-07", "2021-06-08"),
    ("2021-09-20", "2021-09-21"),
    ("2021-06-28", "2021-06-29"),
    ("2021-07-12", "2021-07-13"),
]:
    datasets.get_session_dataset(
        CDPInstances.Seattle,
        start_datetime=start_date,
        end_datetime=end_date,
        store_audio=True,
    )


dataset_dir = Path(f"cdp-datasets/{CDPInstances.Seattle}")
for audio_file in dataset_dir.glob("event-*/session-*/audio.wav"):
    storage = audio_file.parent.parent.name
    if not (Path(storage).exists() or Path(f"ANNOTATED-{storage}").exists()):
        print("working on file:", audio_file)
        print("storing:", storage)
        torch.cuda.empty_cache()
        diarize_and_split_audio(audio_file, storage_dir=storage)
    else:
        print("skipping", storage)
