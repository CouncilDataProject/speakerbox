#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, load_metric
from transformers import (
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)

###############################################################################

DEFAULT_BASE_MODEL = "superb/wav2vec2-base-superb-sid"

###############################################################################


def train(
    dataset: Union[
        pd.DataFrame,
        Dataset,
        DatasetDict,
    ],
    model_base: str = DEFAULT_BASE_MODEL,
    max_duration: float = 5.0,
    batch_size: int = 32,
) -> None:
    """
    Train a speaker classification model.

    Parameters
    ----------
    dataset: Union[pd.DataFrame, Dataset, DatasetDict]
        The dataset to use for training.

        Should only contain the columns/features: "label" and "audio".

        If provided a pandas DataFrame, this will convert it into a Dataset object
        and cast the "label" column into ClassLabels and the "audio" column into Audio.
        If provided a Dataset object (or the converted pandas DataFrame),
        this will convert it into a DatasetDict with random train and test splits.
    model_base: str
        The model base to use before fine tuning.
        Default: "superb/wav2vec2-base-superb-sid"
    max_duration: float
        The maximum duration to use for each audio clip.
        Any clips longer than this will be trimmed.
        Default: 5.0
    batch_size: int
        The number of examples to use in a batch during training.
        Default: 32

    Returns
    -------
    model
    """
    # Convert provided dataset to HF Dataset
    if isinstance(dataset, pd.DataFrame):
        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.class_encode_column("label")
        dataset = dataset.cast_column("audio", Audio())

    # Convert into train test dict
    if isinstance(dataset, Dataset):
        dataset = dataset.train_test_split()

    # Construct label to id and vice-versa LUTs
    label2id, id2label = {}, {}
    for i, label in enumerate(dataset["train"].features["label"].names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)

    # Construct preprocessing function
    def preprocess(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
        )
        return inputs

    # Encode the dataset
    dataset = dataset.map(preprocess, remove_columns=["audio"], batched=True)

    # Create AutoModel
    model = AutoModelForAudioClassification.from_pretrained(
        model_base,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Create fine tuning Trainer
    args = TrainingArguments(
        "speakerbox-fine-tuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Compute accuracy metrics
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
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
    trainer.train()

    # Eval
    trainer.evaluate()

    # Save
    trainer.save_model()
