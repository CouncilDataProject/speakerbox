#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import cdptools
import pandas as pd
from cdptools import CDPInstance, configs
from cdptools.audio_splitters.ffmpeg_audio_splitter import FFmpegAudioSplitter
from distributed import LocalCluster, worker_client
from prefect import Flow, task, unmapped
from prefect.engine.executors import DaskExecutor
from pydub import AudioSegment

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s"
)
log = logging.getLogger(__name__)

###############################################################################
# Tasks


@task
def _download_video(
    event_id: str,
    db: cdptools.databases.Database,
    corpus_dir: Path,
    overwrite: bool
) -> Path:
    # Retrieve full event details
    event_details = db.select_row_by_id("event", event_id)

    # Set storage target
    save_dir = corpus_dir / event_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Download video
    video_ext = event_details["video_uri"].split(".")[-1]
    video_save_path = save_dir / f"video.{video_ext}"
    if not video_save_path.exists() or overwrite:
        video_save_path = cdptools.file_stores.FileStore._external_resource_copy(
            event_details["video_uri"],
            video_save_path
        )

    return video_save_path


@task
def _download_transcript(
    event_id: str,
    db: cdptools.databases.Database,
    fs: cdptools.file_stores.FileStore,
    corpus_dir: Path,
    overwrite: bool
) -> Path:
    # Retrieve full event details
    event_details = db.select_row_by_id("event", event_id)

    # Generate key
    key = hashlib.sha256(event_details["video_uri"].encode("utf8")).hexdigest()

    # Set storage target
    save_dir = corpus_dir / event_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Download transcript
    transcript_save_path = save_dir / "transcript.json"
    if not transcript_save_path.exists() or overwrite:
        transcript_save_path = fs.download_file(
            f"{key}_ts_speaker_turns_transcript_0.json",
            transcript_save_path
        )

    return transcript_save_path


@task
def _split_audio_from_video(video_path: Path, overwrite: bool) -> Path:
    # Init splitter
    splitter = FFmpegAudioSplitter()

    # Split audio
    audio_save_path = video_path.parent / "audio.wav"
    if not audio_save_path.exists() or overwrite:
        audio_save_path = splitter.split(
            video_path,
            video_path.parent / "audio.wav"
        )

    return audio_save_path


@task
def _generate_initial_download_manifest(
    event_ids: List[Dict[str, Any]],
    video_paths: List[Path],
    audio_paths: List[Path],
    transcript_paths: List[Path],
    save_dir: Path,
) -> pd.DataFrame:
    # Generate rows
    rows = []
    for event_id, video_path, audio_path, transcript_path in zip(
        event_ids, video_paths, audio_paths, transcript_paths
    ):
        # Create row
        rows.append({
            "event_id": event_id,
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript_path": transcript_path
        })

    # Create and save manifest
    manifest = pd.DataFrame(rows)
    manifest_save_path = save_dir / "download_manifest.csv"
    manifest.to_csv(manifest_save_path, index=False)

    # Log save location
    log.info(f"Download manifest saved to: {manifest_save_path}")

    return rows


def _split_audio_portion(
    main_audio_path: Path,
    sentence_data: Dict[str, Any],
    save_path: Path,
    overwrite: bool
) -> Path:
    # Read main audio
    source_audio = AudioSegment.from_wav(main_audio_path)

    # Split the audio portion
    split = source_audio[
        # Convert seconds to ms
        sentence_data["start_time"] * 1000:sentence_data["end_time"] * 1000
    ]

    # Save the audio
    # Generate audio / sentence id
    audio_id = str(uuid4())
    audio_split_save_path = (save_path / f"{audio_id}.wav").resolve()
    if not audio_split_save_path.exists() or overwrite:
        split.export(audio_split_save_path, format="wav")

    return {
        "audio_id": audio_id,
        "audio_path": audio_split_save_path,
        "duration": sentence_data["end_time"] - sentence_data["start_time"],
        **sentence_data
    }


def _process_speaker_block(
    speaker_block: Dict[str, Any],
    event_id: str,
    main_audio_path: Path,
    save_path: Path,
    overwrite: bool
) -> Path:
    # Create the save dir if it doesn't exist
    save_path.mkdir(exist_ok=True)

    # Map splitting audio
    with worker_client() as client:
        futures = client.map(
            _split_audio_portion,
            [main_audio_path for i in range(len(speaker_block["data"]))],
            [sentence_data for sentence_data in speaker_block["data"]],
            [save_path for i in range(len(speaker_block["data"]))],
            [overwrite for i in range(len(speaker_block["data"]))]
        )

        # Block until all complete
        results = client.gather(futures)

    # Add speaker block to the results
    for result in results:
        result["event_id"] = event_id

    return pd.DataFrame(results)


@task
def _generate_splits(
    event_details: Dict[str, Any],
    overwrite: bool
) -> Path:
    # Read transcript file
    with open(event_details["transcript_path"], "r") as transcript_read:
        transcript = json.load(transcript_read)

    # Create splits save dir
    splits_dir = (
        event_details["audio_path"].parent / "splits"
    )
    splits_dir.mkdir(exist_ok=True)

    # Mapped process of every speaker block
    with worker_client() as client:
        futures = client.map(
            _process_speaker_block,
            transcript["data"],
            [event_details["event_id"] for i in range(len(transcript["data"]))],
            [event_details["audio_path"] for i in range(len(transcript["data"]))],
            [splits_dir for i in range(len(transcript["data"]))],
            [overwrite for i in range(len(transcript["data"]))]
        )

        # Block until all complete
        results = client.gather(futures)

    # Join results into single dataframe for the event
    event_manifest = pd.concat(results)

    return event_manifest


@task
def _generate_audio_manifest(
    manifests: List[pd.DataFrame],
    save_dir: Path,
) -> Path:
    # Write splits manifest
    manifest = pd.concat(manifests)
    manifest_save_path = save_dir / "audio_manifest.csv"
    manifest.to_csv(manifest_save_path, index=False)

    return manifest_save_path


###############################################################################
# Args

class Args(argparse.Namespace):

    def __init__(self):
        self.__parse()

    def __parse(self):
        # Setup parser
        p = argparse.ArgumentParser(
            prog="download_cdp_dataset",
            description="Download a dataset ready for training from a CDP instance."
        )

        # Arguments
        p.add_argument(
            "instance_name",
            help="Which CDP instance to retrieve the dataset from."
        )
        p.add_argument(
            "--save_dir",
            type=Path,
            default=Path("cdp_data"),
            help="Path to save the dataset to."
        )
        p.add_argument(
            "--overwrite",
            action="store_true",
            dest="overwrite",
            help="Overwrite existing files."
        )
        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help="Show traceback if the script were to fail."
        )

        # Parse
        p.parse_args(namespace=self)


###############################################################################
# Download and prepare CDP dataset

def download_cdp_dataset(args: Args):
    # Try running the download pipeline
    try:
        # Get instance config
        instance_config = getattr(configs, args.instance_name.upper())

        # Create connection to instance
        cdp_instance = CDPInstance(instance_config)

        # Get speaker annotated transcripts
        sats = cdp_instance.database.select_rows_as_list(
            "transcript",
            [("confidence", 0.97)]
        )

        # Spawn local dask cluster
        cluster = LocalCluster()

        # Log dashboard link
        log.info(f"Dashboard available at: {cluster.dashboard_link}")

        # Setup workflow
        with Flow("get_dataset") as flow:
            # Download videos
            video_paths = _download_video.map(
                [sat["event_id"] for sat in sats],
                unmapped(cdp_instance.database),
                unmapped(args.save_dir),
                unmapped(args.overwrite)
            )

            # Split audio from video
            audio_paths = _split_audio_from_video.map(
                video_paths,
                unmapped(args.overwrite)
            )

            # Download transcripts
            transcript_paths = _download_transcript.map(
                [sat["event_id"] for sat in sats],
                unmapped(cdp_instance.database),
                unmapped(cdp_instance.file_store),
                unmapped(args.save_dir),
                unmapped(args.overwrite)
            )

            # Create large audio manifest
            events = _generate_initial_download_manifest(
                [sat["event_id"] for sat in sats],
                video_paths,
                audio_paths,
                transcript_paths,
                args.save_dir
            )

            # Generate sentence splits
            manifests = _generate_splits.map(
                events,
                unmapped(args.overwrite)
            )

            # Generate autio manifest
            _generate_audio_manifest(
                manifests,
                unmapped(args.save_dir)
            )

        # Run the flow
        state = flow.run(executor=DaskExecutor(cluster.scheduler_address))

        # Log resulting manifest
        manifest_save_path = (
            state.result[flow.get_tasks(name="_generate_audio_manifest")[0]].result
        )
        log.info(f"Dataset manifest stored to: {manifest_save_path}")

    # Catch any exception
    except Exception as e:
        log.error("=============================================")
        if args.debug:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)

###############################################################################
# Runner


def main():
    args = Args()
    download_cdp_dataset(args)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
