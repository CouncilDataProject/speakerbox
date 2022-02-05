#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict, load_metric
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    pipeline,
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
    model_name: str = "trained-speakerbox",
    model_base: str = DEFAULT_BASE_MODEL,
    max_duration: float = 2.0,
    batch_size: int = 2,
) -> Path:
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
        this will convert it into a DatasetDict with
        random train, test, and validation splits.
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
        Default: 2

    Returns
    -------
    model_storage_path: Path
        The path to the directory where the model is stored.
    """
    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)

    # Convert provided dataset to HF Dataset
    if isinstance(dataset, pd.DataFrame):
        dataset = Dataset.from_pandas(dataset, preserve_index=False)
        dataset = dataset.class_encode_column("label")
        dataset = dataset.cast_column("audio", Audio(feature_extractor.sampling_rate))

    # Convert into train, test, and validate dict
    if isinstance(dataset, Dataset):
        train_and_test = dataset.train_test_split(test_size=0.4)
        test_and_valid = train_and_test["test"].train_test_split(test_size=0.5)
        ds_dict = DatasetDict(
            {
                "train": train_and_test["train"],
                "test": test_and_valid["train"],
                "valid": test_and_valid["test"],
            }
        )

    # Show dataset summary stats
    for subset in ["train", "test", "valid"]:
        pd_subset = ds_dict[subset].to_pandas()
        print(f"Summary stats for '{subset}' dataset")
        print(f"n-labels: {pd_subset.label.nunique()}")
        print(f"Avg duration: {pd_subset.duration.mean()}")
        print(f"Min duration: {pd_subset.duration.min()}")
        print(f"Max duration: {pd_subset.duration.max()}")
        print(f"StD duration: {pd_subset.duration.std()}")
        print("-" * 80)

    # Construct label to id and vice-versa LUTs
    label2id, id2label = dict(), dict()
    for i, label in enumerate(ds_dict["train"].features["label"].names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Construct preprocessing function
    def preprocess(examples):
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
    ds_dict = ds_dict.map(preprocess, batched=True)

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
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        gradient_checkpointing=True,
    )

    # Compute accuracy metrics
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        # Eval pred comes with both the predictions and the attention mask
        # grab just the predictions
        predictions = np.argmax(eval_pred.predictions[0], axis=-1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # Trainer and train!
    trainer = Trainer(
        model,
        args,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
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
    ds_dict["valid"] = ds_dict["valid"].map(
        lambda example: {
            "prediction": classifier(example["audio"]["path"], top_k=1)[0]["label"],
            "label_str": classifier.model.config.id2label[example["label"]],
        }
    )

    # Create confusion
    ConfusionMatrixDisplay.from_predictions(
        ds_dict["valid"]["label_str"],
        ds_dict["valid"]["prediction"],
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(f"{model_name}/validation-confusion.png")

    return Path(model_name).resolve()
