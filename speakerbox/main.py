#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from transformers import pipeline

if TYPE_CHECKING:
    import datasets
    from datasets import Dataset, DatasetDict, arrow_dataset
    from pyannote.core.annotation import Annotation
    from pydub import AudioSegment
    from transformers import EvalPrediction, feature_extraction_utils
    from transformers.pipelines.base import Pipeline

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_BASE_MODEL = "superb/wav2vec2-base-superb-sid"

EVAL_RESULTS_TEMPLATE = """
## Results

* **Accuracy:** {accuracy}
* **Precision:** {precision}
* **Recall:** {recall}
* **Validation Loss:** {loss}

### Confusion
"""

DEFAULT_TRAINER_ARGUMENTS_ARGS = dict(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=40,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    gradient_checkpointing=True,
)


###############################################################################


def eval_model(
    validation_dataset: "Dataset",
    model_name: str = "trained-speakerbox",
) -> Tuple[float, float, float, float]:
    """
    Evaluate a trained model.

    This will store two files in the model directory, one for the accuracy, precision,
    and recall in a markdown file and the other is the generated top one confusion
    matrix as a PNG file.

    Parameters
    ----------
    validation_dataset: Dataset
        The dataset to validate the model against.
    model_name: str
        A name for the model. This will also create a directory with the same name
        to store the produced model in.
        Default: "trained-speakerbox"

    Returns
    -------
    accuracy: float
        The model accuracy as returned by sklearn.metrics.accuracy_score.
    precision: float
        The model (weighted) precision as returned by sklearn.metrics.precision_score.
    recall: float
        The model (weighted) recall as returned by sklearn.metrics.recall_score.
    loss: float
        The model log loss as returned by sklearn.metrics.log_loss.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        log_loss,
        precision_score,
        recall_score,
    )

    log.info("Setting up evaluation pipeline")
    classifier = pipeline(
        "audio-classification",
        model=model_name,
    )
    log.info("Running eval")

    def predict(example: "datasets.arrow_dataset.Example") -> Dict[str, Any]:
        pred = classifier(example["audio"], top_k=1000)
        pred_as_dict = {i["label"]: i["score"] for i in pred}
        top_pred = max(pred_as_dict, key=pred_as_dict.get)  # type: ignore
        return {
            "pred_label": top_pred,
            "true_label": classifier.model.config.id2label[example["label"]],
            "pred_scores": [i["score"] for i in pred],
        }

    validation_dataset = validation_dataset.map(predict)

    # Create confusion
    ConfusionMatrixDisplay.from_predictions(
        validation_dataset["true_label"],
        validation_dataset["pred_label"],
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(f"{model_name}/validation-confusion.png", bbox_inches="tight")

    # Compute metrics
    accuracy = accuracy_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
    )
    precision = precision_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
        average="weighted",
    )
    recall = recall_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
        average="weighted",
    )
    loss = log_loss(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_scores"],
    )

    # Store metrics
    with open(f"{model_name}/results.md", "w") as open_f:
        open_f.write(
            EVAL_RESULTS_TEMPLATE.format(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                loss=loss,
            )
        )

    return (accuracy, precision, recall, loss)


def train(
    dataset: "DatasetDict",
    model_name: str = "trained-speakerbox",
    model_base: str = DEFAULT_BASE_MODEL,
    max_duration: float = 2.0,
    seed: Optional[int] = None,
    use_cpu: bool = False,
    trainer_arguments_kws: Dict[str, Any] = DEFAULT_TRAINER_ARGUMENTS_ARGS,
) -> Path:
    """
    Train a speaker classification model.

    Parameters
    ----------
    dataset: DatasetDict
        The datasets to use for training, testing, and validation.
        Should only contain the columns/features: "label" and "audio".
        The values in the "audio" column should be paths to the audio files.
    model_name: str
        A name for the model. This will also create a directory with the same name
        to store the produced model in.
        Default: "trained-speakerbox"
    model_base: str
        The model base to use before fine tuning.
    max_duration: float
        The maximum duration to use for each audio clip.
        Any clips longer than this will be trimmed.
        Default: 2.0
    seed: Optional[int]
        Seed to pass to torch, numpy, and Python RNGs.
        Default: None (do not set a seed)
    use_cpu: bool
        Should the model be trained using CPU.
        This also sets `no_cuda=True` on TrainerArguments.
        Default: False (use GPU if available)
    trainer_arguments_kws: Dict[Any]
        Any additional keyword arguments to be passed to the HuggingFace
        TrainerArguments object.
        Default: DEFAULT_TRAINER_ARGUMENTS_ARGS

    Returns
    -------
    model_storage_path: Path
        The path to the directory where the model is stored.
    """
    import numpy as np
    import torch
    import transformers
    from datasets import Audio, load_metric
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForSequenceClassification,
    )

    from .utils import set_global_seed

    # Handle seed
    if seed:
        set_global_seed(seed)

    # Handle cpu
    if use_cpu:
        trainer_arguments_kws["no_cuda"] = True

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)

    # Convert dataset audios
    log.info("Casting all audio paths to torch Audio")
    dataset = dataset.cast_column("audio", Audio(feature_extractor.sampling_rate))

    # Construct label to id and vice-versa LUTs
    label2id, id2label = dict(), dict()
    for i, label in enumerate(dataset["train"].features["label"].names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Construct preprocessing function
    def preprocess(
        examples: "arrow_dataset.Batch",
    ) -> "feature_extraction_utils.BatchFeature":
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            do_normalize=True,
            truncation=True,
            padding=True,
        )
        return inputs

    # Encode the dataset
    log.info("Extracting features from audio")
    dataset = dataset.map(preprocess, batched=True)

    # Create AutoModel
    log.info("Setting up Trainer")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_base,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Create fine tuning Trainer
    args = TrainingArguments(
        model_name,
        **trainer_arguments_kws,
    )

    # Compute accuracy metrics
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred: "EvalPrediction") -> Optional[Dict]:
        # Eval pred comes with both the predictions and the attention mask
        # grab just the predictions
        predictions = np.argmax(eval_pred.predictions[0], axis=-1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # Trainer and train!
    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    torch.cuda.empty_cache()
    transformers.logging.set_verbosity_info()
    trainer.train()

    # Save model
    trainer.save_model()

    return Path(model_name).resolve()


def apply(
    audio: Union[str, Path, "AudioSegment"],
    model: Union[str, "Pipeline"],
    min_chunk_duration: float = 0.5,
    max_chunk_duration: float = 2.0,
    confidence_threshold: float = 0.9,
) -> "Annotation":
    import numpy as np
    from pyannote.core.annotation import Annotation
    from pyannote.core.segment import Segment
    from pyannote.core.utils.types import Label, TrackName
    from pydub import AudioSegment

    # Generate random uuid filename for storing temp audio chunks
    TMP_AUDIO_CHUNK_SAVE_PATH = Path(".tmp-audio-chunk-during-apply.wav")

    # Load audio
    if isinstance(audio, (str, Path)):
        track_name = str(audio)
        audio = AudioSegment.from_file(audio)
    else:
        track_name = "track_0"

    # Load model
    if isinstance(model, str):
        classifier = pipeline("audio-classification", model=model)
    else:
        classifier = model

    # Move audio window, apply, and append annotation record
    records: List[Tuple[Segment, TrackName, Label]] = []
    try:
        for chunk_start_seconds in np.arange(
            0, audio.duration_seconds, max_chunk_duration
        ):
            # Calculate chunk end
            chunk_end_seconds = chunk_start_seconds + max_chunk_duration
            if chunk_end_seconds > audio.duration_seconds:
                chunk_end_seconds = audio.duration_seconds

            # Check if duration is long enough
            duration = chunk_end_seconds - chunk_start_seconds
            if duration >= min_chunk_duration:
                # Convert seconds to millis
                chunk_start_millis = chunk_start_seconds * 1000
                chunk_end_millis = chunk_end_seconds * 1000

                # Select chunk
                chunk = audio[chunk_start_millis:chunk_end_millis]

                # Write chunk to temp
                chunk.export(TMP_AUDIO_CHUNK_SAVE_PATH, format="wav")

                # Predict, keep top 1 and store to records
                pred = classifier(str(TMP_AUDIO_CHUNK_SAVE_PATH), top_k=1)[0]
                if pred["score"] >= confidence_threshold:
                    records.append(
                        (
                            Segment(chunk_start_seconds, chunk_end_seconds),
                            track_name,
                            pred["label"],
                        )
                    )

        # TODO:
        # merge segments
        # moving window for better application

        return Annotation.from_records(records)

    finally:
        # Always clean up tmp file
        if TMP_AUDIO_CHUNK_SAVE_PATH.exists():
            TMP_AUDIO_CHUNK_SAVE_PATH.unlink()
