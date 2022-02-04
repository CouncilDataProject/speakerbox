from speakerbox.datasets.seattle_2021_proto import pull_audio
from speakerbox.datasets.utils import expand_annotations_to_dataset
from speakerbox.main import train

ds = pull_audio()
ds = expand_annotations_to_dataset(ds, overwrite=True)
ds = ds.sample(frac=0.05)
train(ds)
