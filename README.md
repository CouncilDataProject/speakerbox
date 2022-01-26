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

Load the 2021 Seattle Prototype Dataset and get summary statistics for speaker time.

```python
from speakerbox.datasets import (
    unpack_seattle_2021_proto,
    summarize_annotation_statistics,
)

ds_dir = unpack_seattle_2021_proto(clean=True)
summary_stats = summarize_annotation_statistics(ds_dir / "annotations")
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**
