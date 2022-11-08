# speakerbox

[![Build Status](https://github.com/CouncilDataProject/speakerbox/workflows/CI/badge.svg)](https://github.com/CouncilDataProject/speakerbox/actions)
[![Documentation](https://github.com/CouncilDataProject/speakerbox/workflows/Documentation/badge.svg)](https://CouncilDataProject.github.io/speakerbox)

Few-Shot Multi-Recording Speaker Identification Transformer Fine-Tuning and Application

---

## Installation

**Stable Release:** `pip install speakerbox`<br>
**Development Head:** `pip install git+https://github.com/CouncilDataProject/speakerbox.git`

## Documentation

For full package documentation please visit [councildataproject.github.io/speakerbox](https://councildataproject.github.io/speakerbox).

## Problem

Given a set of recordings of multi-speaker recordings:

```
example/
├── 0.wav
├── 1.wav
├── 2.wav
├── 3.wav
├── 4.wav
└── 5.wav
```

Where each recording has some or all of a set of speakers, for example:

-   0.wav -- contains speakers: A, B, C, D, E
-   1.wav -- contains speakers: B, D, E
-   2.wav -- contains speakers: A, B, C
-   3.wav -- contains speakers: A, B, C, D, E
-   4.wav -- contains speakers: A, C, D
-   5.wav -- contains speakers: A, B, C, D, E

You want to train a model to classify portions of audio as one of the N known speakers
in future recordings not included in your original training set.

`f(audio) -> [(start_time, end_time, speaker), (start_time, end_time, speaker), ...]`

i.e. `f(audio) -> [(2.4, 10.5, "A"), (10.8, 14.1, "D"), (14.8, 22.7, "B"), ...]`

The `speakerbox` library contains methods for both generating datasets for annotation
and for utilizing multiple audio annotation schemes to train such a model.

![Speakerbox example workflow](https://raw.githubusercontent.com/CouncilDataProject/speakerbox/main/docs/_static/images/workflow.png)

The following table shows model performance results as the dataset size increases:

| dataset_size   | mean_accuracy   | mean_precision   | mean_recall   | mean_training_duration_seconds   |
|:---------------|----------------:|-----------------:|--------------:|---------------------------------:|
| 15-minutes     | 0.874 ± 0.029   | 0.881 ± 0.037    | 0.874 ± 0.029 | 101 ± 1                          |
| 30-minutes     | 0.929 ± 0.006   | 0.94 ± 0.007     | 0.929 ± 0.006 | 186 ± 3                          |
| 60-minutes     | 0.937 ± 0.02    | 0.94 ± 0.017     | 0.937 ± 0.02  | 453 ± 7                          |

All results reported are the average of five model training and evaluation trials for each
of the different dataset sizes. All models were fine-tuned using an NVIDIA GTX 1070 TI.

**Note:** this table can be reproduced in ~1 hour using an NVIDIA GTX 1070 TI by
[downloading the example dataset](https://drive.google.com/file/d/1snDuv45cYCYxCae19Dz4tsQauLrA425w/view?usp=sharing)
unzipping it's contents and then running:

```python
from speakerbox.examples import train_and_eval_all_example_models

results = train_and_eval_all_example_models(
    # be sure to provide the correct path to the unzipped directory
    "/home/eva/Downloads/example-speakerbox-dataset/",
)
```

## Workflow

### Diarization

We quickly generate an annotated dataset by first diarizing (or clustering based
on the features of speaker audio) portions of larger audio files and splitting each the
of the clusters into their own directories that you can then manually clean up
(by removing incorrectly clustered audio segments).

#### Notes

-   It is recommended to have each larger audio file named with a unique id that
    can be used to act as a "recording id".
-   Diarization time depends on machine resources and make take a long time -- one
    potential recommendation is to run a diarization script overnight and clean up the
    produced annotations the following day.
-   During this process audio will be duplicated in the form of smaller audio clips --
    ensure you have enough space on your machine to complete this process before
    you begin.
-   Clustering accuracy depends on how many speakers there are, how distinct their
    voices are, and how much speech is talking over one-another.
-   If possible, try to find recordings where speakers have a roughly uniform distribution
    of speaking durations.

⚠️ To use the diarization portions of `speakerbox` you need to complete the
following steps: ⚠️

1. Visit [hf.co/pyannote/speaker-diarization](hf.co/pyannote/speaker-diarization)
   and accept user conditions.
2. Visit [hf.co/pyannote/segmentation](hf.co/pyannote/segmentation)
   and accept user conditions.
3. Visit [hf.co/settings/tokens](hf.co/settings/tokens) to create an access token
   (only if you had to complete 1.)

```python
from speakerbox import preprocess

# The token can also be provided via the 'HUGGINGFACE_TOKEN` environment variable.
diarized_and_split_audio_dir = preprocess.diarize_and_split_audio(
    "0.wav",
    hf_token="token-from-hugging-face",
)
```

### Cleaning

Diarization will produce a directory structure organized by unlabeled speakers with
the audio clips that were clustered together.

For example, if `"0.wav"` had three speakers, the produced directory structure may look
like the following tree:

```
0/
├── SPEAKER_00
│   ├── 567-12928.wav
│   ├── ...
│   └── 76192-82901.wav
├── SPEAKER_01
│   ├── 34123-38918.wav
│   ├── ...
│   └── 88212-89111.wav
└── SPEAKER_02
    ├── ...
    └── 53998-62821.wav
```

We leave it to you as a user to then go through these directories and remove any audio
clips that were incorrectly clustered together as well as renaming the sub-directories
to their correct speaker labels. For example, labelled sub-directories may look like
the following tree:

```
0/
├── A
│   ├── 567-12928.wav
│   ├── ...
│   └── 76192-82901.wav
├── B
│   ├── 34123-38918.wav
│   ├── ...
│   └── 88212-89111.wav
└── D
    ├── ...
    └── 53998-62821.wav
```

#### Notes

-   Most operating systems have an audio playback application to queue an entire directory
    of audio files as a playlist for playback. This makes it easy to listen to a whole
    unlabeled sub-directory (i.e. "SPEAKER_00") at a time and pause playback and remove
    files from the directory which were incorrectly clustered.
-   If any clips have overlapping speakers, it is up to you as a user if you want to
    remove those clips or keep them and properly label them with the speaker you wish to
    associate them with.

### Training Preparation

Once you have annotated what you think is enough recordings, you can try preparing
a dataset for training.

The following functions will prepare the audio for training by:

1. Finding all labeled audio clips in the provided directories
2. Chunk all found audio clips into smaller duration clips _(parametrizable)_
3. Check that the provided annotated dataset meets the following conditions:
    1. There is enough data such that the training, test, and validation subsets all
       contain different recording ids.
    2. There is enough data such that the training, test, and validation subsets each
       contain all labels present in the whole dataset.

#### Notes

-   During this process audio will be duplicated in the form of smaller audio clips --
    ensure you have enough space on your machine to complete this process before
    you begin.
-   Directory names are used as recording ids during dataset construction.

```python
from speakerbox import preprocess

dataset = preprocess.expand_labeled_diarized_audio_dir_to_dataset(
    labeled_diarized_audio_dir=[
        "0/",  # The cleaned and checked audio clips for recording id 0
        "1/",  # ... recording id 1
        "2/",  # ... recording id 2
        "3/",  # ... recording id 3
        "4/",  # ... recording id 4
        "5/",  # ... recording id 5
    ]
)

dataset_dict, value_counts = preprocess.prepare_dataset(dataset)

# You can print the value_counts dataframe to see how many audio clips of each label
# (speaker) are present in each data subset.
print(value_counts)
```

### Model Training and Evaluation

Once you have your dataset prepared and available, you can provide it directly to the
training function to begin training a new model.

The `eval_model` function will store a filed called `results.md` with the accuracy,
precision, and recall of the model and additionally store a file called
`validation-confusion.png` which is a
[confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

#### Notes

-   The model (and evaluation metrics) will be stored in a new directory called
    `trained-speakerbox` _(parametrizable)_.
-   Training time depends on how much data you have annotated and provided.
-   It is recommended to train with an NVidia GPU with CUDA available to speed up
    the training process.
-   Speakerbox has only been tested on English-language audio and the base model for
    fine-tuning was trained on English-language audio. We provide no guarantees as to
    it's effectiveness on non-English-language audio. If you try Speakerbox on with
    non-English-language audio, please let us know!

```python
from speakerbox import train, eval_model

# dataset_dict comes from previous preparation step
train(dataset_dict)

eval_model(dataset_dict["valid"])
```

## Model Inference

Once you have a trained model, you can use it against a new audio file:

```python
from speakerbox import apply

annotation = apply(
    "new-audio.wav",
    "path-to-my-model-directory/",
)
```

The apply function returns a
[pyannote.core.Annotation](http://pyannote.github.io/pyannote-core/structure.html#annotation).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
