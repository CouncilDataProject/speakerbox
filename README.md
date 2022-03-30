# speakerbox

[![Build Status](https://github.com/CouncilDataProject/speakerbox/workflows/Build/badge.svg)](https://github.com/CouncilDataProject/speakerbox/actions)
[![Documentation](https://github.com/CouncilDataProject/speakerbox/workflows/Documentation/badge.svg)](https://CouncilDataProject.github.io/speakerbox)
[![Code Coverage](https://codecov.io/gh/CouncilDataProject/speakerbox/branch/main/graph/badge.svg)](https://codecov.io/gh/CouncilDataProject/speakerbox)

Speaker Annotation for Transcripts using Audio Classification

---

## Installation

**Stable Release:** `pip install speakerbox`<br>
**Development Head:** `pip install git+https://github.com/CouncilDataProject/speakerbox.git`

## Documentation

For full package documentation please visit [councildataproject.github.io/speakerbox](https://councildataproject.github.io/speakerbox).

## Quickstart

```python
from speakerbox import expand_annotations_to_dataset, train
from speakerbox.ds import seattle_2021_proto

# Unpack / unzip the Seattle city council 2021 prototype dataset
seattle_2021_ds_dir = seattle_2021_proto.unpack(clean=True)

# Pull matching audio files for each annotation file
seattle_2021_ds_items = seattle_2021_proto.pull_all_files()

# Expand from multiple matching large gecko annotation files and large audio files
# into many small audio clips with speaker labels
seattle_2021_ds = expand_annotations_to_dataset(seattle_2021_ds_items, overwrite=True)

# Train a new model
model_dir = train(seattle_2021_ds)

# See the created trained-speakerbox directory for a confusion matrix image
# generated from a holdout validation set.
```

Once training is complete, a model will be available for use with:

```python
from transformers import pipeline

# Assume you have some two second audio file stored
speaker_audio_clip_path = "..."

# Init the classifier
classifier = pipeline(
    "audio-classification",
    model=model_dir,  # output from `train` from above
)

# Apply and see results
print(classifier(speaker_audio_clip_path, top_k=5))

# For most use cases you really just care about `top_k=1`
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**
