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

Load the 2021 Seattle Prototype Dataset, get summary statistics
about speaker time, finally pull the matching audio file for each annotation file
and store annotation file matched to audio as a `pandas.DataFrame`.

```python
from speakerbox import datasets

seattle_2021_ds_dir = datasets.unpack_seattle_2021_proto(clean=True)
seattle_2021_ds_summary_stats = datasets.summarize_annotation_statistics(
    seattle_2021_ds_dir / "annotations"
)
seattle_2021_ds = datasets.pull_seattle_2021_proto_audio()
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**
