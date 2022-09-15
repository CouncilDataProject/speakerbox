---
title: "Speakerbox: A Workflow for Annotation and Training for Multi-Speaker Classification Models"
tags:
    - Python
    - speech recognition
    - machine learning
authors:
    - name: Eva Maxfield Brown
      orcid: 0000-0003-2564-0373
      affiliation: 1
    - name: To Huynh
      orcid: 0000-0002-9664-3662
      affiliation: 2
    - name: Isaac Na
      orcid: 0000-0002-0182-1615
      affiliation: 3
    - name: Nicholas Weber
      orcid: 0000-0002-6008-3763
      affiliation: 1

affiliations:
    - name: University of Washington Information School, University of Washington, Seattle
      index: 1
    - name: University of Washington, Seattle
      index: 2
    - name: Washington University, St. Louis
      index: 3

date: 7 September 2022
bibliography: paper.bib
---

# Summary



# Statement of Need

Speaker annotated audio and transcript data has previously been used to create comprehensive analyses of conversation dynamics[@jacobi_justice_2017,@morris_reexamining_2001,@osborn_speaking_2010,@miller_effect_2022,@maltzman_politics_1996,@slapin_sound_2020]. However, multi-speaker audio classification models for the purpose of speaker classification can be cumbersome and expensive to train and unweldy to apply. Speaker diarization, "the unsupervised identification of each speaker within an audio stream and the intervals during which each speaker is active" [@SpeakerDia2012], is a useful method in certain applications and while there are free and open tools for application speaker diarization algorithms [@Bredin2020,Bredin2021], in many cases, unsupervised methods are not enough. Openly available and easy-to-use tools for supervised methods of speaker identification are lacking however and Speakerbox fills this gap by providing a workflow and method to annotate audio data, train and evaluate a speaker identification model, and apply a trained model in just a few hours.

## Related Work

pyannote-audio

transformers ?

auto-ml?

explosion ai ?

## Workflow

Speakerbox provides two methods to prepare a speaker identification training dataset: diarization for audio and import from existing annotation tools.

#### Diarization

While diarization cannot be used as a single solution for multi-speaker multi-recording speaker identification, it can be used for quickly generating large amounts of training examples which can be validated and labeled for use in a later speaker identification training set.

We make use of diarization as one method for preparing a speaker identification training dataset by using a diarization model provided by `pyannote.audio` to diarize an audio file and place the unlabeled chunks of audio into directories on the users file system. A user can then listen to a few or all of the samples of audio in each directory, remove any samples that were mis-classified, and finally rename each of the directories with a speaker identifier (i.e. a name, database id, etc.).

[TODO: insert fig?]

In certain cases, this diarization workflow may be enough to train an accurate speaker identification model.

#### Using Gecko Annotations

To improve model accuracy and improve coverage of edge cases, users may find it useful to use Gecko: a free web application for manual segmentation of an audio file by speaker as well as annotation of the linguistic content of a conversation [@Gecko2019]. Speakerbox makes use of Gecko annotations as a method for training dataset creation by providing functions to split and prepare audio files using a filepath to a Gecko output JSON file as a parameter.

### Data Preparation

Speakerbox is built with the goal of making multi-recording, multi-speaker classification model training as simple as possible while still attempting to help train an accurate, useful model for applciation. To ensure that the model is learning the features of each speaker's voice and not the features of the microphone or the specific words and phrases of the recording, we create dataset training, test, and evaluation splits based off of a recording holdout and speaker stratification pattern. Each train, test, or evaluation subset must contain unique recording IDs to reduce the chance of learning the features of specific microphones or recording contexts. We then check that our speaker stratifaction requirement is met by checking that each of the produced subsets contains recordings of every speaker available from the whole dataset. For example, if there are nine unique speakers in the dataset as a whole, then each train, test, and evaluation subset is required to have all nine speakers as well.

If either of these conditions are not met, Speakerbox retries this random sampling process. If we cannot find a valid recoding holdout and speaker stratification configuration given the sampling iterations, we inform the user of this failure and ask them to add more examples to the dataset.

TODO -- INSERT FIG OF THIS?

TODO -- allow param of "no holdout" and "no stratify" to ignore this process.

### Model Training

Once the data is prepared into valid training, evaluation, and testing subsets, we provide a method for training a new model.

This training process isn't a full model training but rather a fine-tuning of an existing multi-speaker classification transformer trained on the VoxCeleb dataset (TODO -- CITE TRANSFORMER FINE TUNED).

## Model Application

TODO -- WRITE APPLICATION FUNCTION / MOVE IT OVER TO THIS REPO FROM CDP-BACKEND

ALSO SPLIT DEPS TO MAKE TRAINING AND APPLICATION DEPS DIFFERENT.

## Usage in Existing Research

We are actively utilizing speakerbox trained models to turn unlabeled transcripts provided by the Council Data Project (TODO - CITE) into labelled ones which we can then conduct analysis on for understanding common speaker patterns within city councils across the United States.

Our two end goals of this work is to:

* produce a dataset of Council Data Project transcripts labelled using Speakerbox models (a model trained per municipal council)
* use such a dataset to conduct analysis on speaker patterns and behaviors in municipal council meetings

## Future Work

There are a few additions that would greatly extend the functionality of speakerbox that can be done in future work. First is the creation of a GUI for the application and workflow process. GUIs developed from pure Python have become much more accessible in recent years and a GUI would likely help non-computational scientists a great deal. We hope to additionally use Speakerbox produced models in a more "production" sense by directly integrating their use into the Council Data Project processing workflow so that transcripts are labelled after they are produced. In doing so, we hope to create a template GitHub repository that can be created with Cookiecutter (TODO -- CITE) to generate a repository for storing annotation files and audio files for which to use for automatically training a model using Continuous Integration (CI) systems.

We have previously used such methods for the training of speakerbox models for our work (TODO -- LINK TO PHD INFRASTRUCTURES) but we hope to package them into a more complete template for others to use.

# Acknowledgements

We wish to thank the University of Washington Information School for support.

# References
