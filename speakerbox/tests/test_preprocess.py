#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from speakerbox import preprocess

###############################################################################


@pytest.mark.parametrize(
    "audio_filename",
    [
        "short_0.wav",
        "short_1.wav",
        "short_2.wav",
    ],
)
def test_diarize_and_split_audio(data_dir: Path, audio_filename: str) -> None:
    """
    This is a smoke test. We do not check the validity of the produced clusters,
    we simply care that the function runs through without error.
    """
    # Use data dir with filename
    audio_path = data_dir / audio_filename

    # Diarize
    preprocess.diarize_and_split_audio(
        audio_file=audio_path,
        storage_dir=f"test-outputs/diarization/{audio_path.with_suffix('').name}",
    )
