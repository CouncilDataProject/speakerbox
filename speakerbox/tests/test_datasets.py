#!/usr/bin/env python
# -*- coding: utf-8 -*-

from speakerbox import preprocess
from speakerbox.datasets import seattle_2021_proto

###############################################################################


def test_unpack_pull_and_expand_seattle() -> None:
    # Unpack / unzip the Seattle city council 2021 prototype dataset
    seattle_2021_ds_dir = seattle_2021_proto.unpack(
        dest="test-outputs/seattle-2021-proto/",
        clean=True,
    )

    # Pull matching audio files for each annotation file
    seattle_2021_ds_items = seattle_2021_proto.pull_all_files(
        annotations_dir=seattle_2021_ds_dir / "annotations",
        transcript_output_dir=seattle_2021_ds_dir / "transcripts",
        audio_output_dir=seattle_2021_ds_dir / "audios",
        overwrite=True,
    )

    # Expand from multiple matching large gecko annotation files and large audio files
    # into many small audio clips with speaker labels
    seattle_2021_ds = preprocess.expand_gecko_annotations_to_dataset(
        seattle_2021_ds_items,
        audio_output_dir="test-outputs/seattle-2021-proto/chunked-audio/",
        overwrite=True,
    )

    assert len(seattle_2021_ds) > 0
