#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Audio, DatasetDict, arrow_dataset, load_metric
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    feature_extraction_utils,
    pipeline,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_BASE_MODEL = "superb/wav2vec2-base-superb-sid"

###############################################################################


def train(
    dataset: DatasetDict,
    model_name: str = "trained-speakerbox",
    model_base: str = DEFAULT_BASE_MODEL,
    max_duration: float = 2.0,
    batch_size: int = 4,
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
    batch_size: int
        The number of examples to use in a batch during training.
        Default: 4

    Returns
    -------
    model_storage_path: Path
        The path to the directory where the model is stored.
    """
    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)

    # Convert dataset audios
    log.debug("Casting all audio paths to torch Audio")
    dataset = dataset.cast_column("audio", Audio(feature_extractor.sampling_rate))

    # Construct label to id and vice-versa LUTs
    label2id, id2label = dict(), dict()
    for i, label in enumerate(dataset["train"].features["label"].names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Construct preprocessing function
    def preprocess(
        examples: arrow_dataset.Batch,
    ) -> feature_extraction_utils.BatchFeature:
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
    dataset = dataset.map(preprocess, batched=True)

    # Create AutoModel
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        gradient_checkpointing=True,
    )

    # Compute accuracy metrics
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred: EvalPrediction) -> Optional[Dict]:
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
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,  # num evals that acc worsens before exit
                early_stopping_threshold=0.02,  # acc must improve by this or exit
            )
        ],
    )
    torch.cuda.empty_cache()
    trainer.train()

    # Save model
    trainer.save_model()

    # Eval validation set
    classifier = pipeline(
        "audio-classification",
        model=model_name,
    )
    dataset["valid"] = dataset["valid"].map(
        lambda example: {
            "prediction": classifier(example["audio"]["path"], top_k=1)[0]["label"],
            "label_str": classifier.model.config.id2label[example["label"]],
        }
    )

    # Create confusion
    ConfusionMatrixDisplay.from_predictions(
        dataset["valid"]["label_str"],
        dataset["valid"]["prediction"],
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(f"{model_name}/validation-confusion.png")

    return Path(model_name).resolve()
