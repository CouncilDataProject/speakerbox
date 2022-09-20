---
title: "Speakerbox: Supervised Speaker Identification Model Training and Application"
tags:
    - Python
    - speech identification
    - audio classification
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

Speaker annotated transcripts from audio recordings is an increasingly important applied research problem in natural language processing. For example, speaker annotated audio and transcript data has previously been used to create comprehensive analyses of conversation dynamics [@jacobi_justice_2017;@morris_reexamining_2001;@osborn_speaking_2010;@miller_effect_2022;@maltzman_politics_1996;@slapin_sound_2020]. However, multi-speaker audio classification models for the purpose of speaker identification can be cumbersome and expensive to train and unweldy to apply. Speaker diarization, "the unsupervised identification of each speaker within an audio stream and the intervals during which each speaker is active" [@SpeakerDia2012], is a useful method in certain applications and while there are free and open tools available for speaker diarization [@Bredin2020;@Bredin2021], in many cases, unsupervised methods do not adequately meet the needs of researchers.

Speakerbox is built with the goal of making multi-recording, multi-speaker identification model training as simple as possible while still attempting to help train an accurate, useful model for application. To this end Speakerbox provides:

1. Functions to create annotation sets and functions to import annotations.
2. A function to prepare an audio dataset into train, test, and validation subsets.
3. A function to train and evaluate a transformer model.
4. A function to apply a trained model to a new audio file.

## Related Work

pyannote-audio

transformers ?

auto-ml?

explosion ai ?

## Workflow

Speakerbox provides two methods to prepare a speaker identification training dataset: speaker diarization and labeling, and, importing annotations from other annotation tools.

#### Diarization

Diarization is the unsupervised process of splitting audio into segments grouped by speaker identity. The output of a diarization model is usually a random ID (i.e. "speaker_0", "speaker_1", etc.) where there is no guarantee that "speaker_0" from a first audio file, is the same "speaker_0" from a second audio file. Because of this unsupervised nature, diarization cannot be used as a single solution for multi-speaker multi-recording speaker identification. It can however be used for quickly generating large amounts of training examples which can be validated and labeled for use in a later speaker identification training set.

We make use of diarization as one method for preparing a speaker identification training dataset by using a diarization model provided by `pyannote.audio` to diarize an audio file and place the unlabeled chunks of audio into directories on the user's file system. A user can then listen to a few or all of the samples of audio in each directory, remove any samples that were mis-classified, and finally rename each of the directories with a true and consistent speaker identifier (i.e. a name, database ID, etc.).

TODO WRITING: insert fig?

#### Using Gecko Annotations

To improve model accuracy and improve coverage of edge cases, users may find it useful to use [Gecko](https://github.com/gong-io/gecko): a free web application for manual segmentation of an audio files by speaker as well as annotation of the linguistic content of a conversation [@Gecko2019]. Speakerbox makes use of Gecko annotations as a method for training dataset creation by providing functions to split and prepare audio files using the annotations stored in a Gecko created JSON file.

### Data Preparation

To ensure that the model is learning the features of each speaker's voice and not the features of the microphone or the specific words and phrases of the recording, we create dataset training, test, and evaluation splits based off of a recording holdout and speaker stratification pattern. Each train, test, or evaluation subset must contain unique recording IDs to reduce the chance of learning the features of specific microphones or recording contexts. We then check that our speaker stratifaction requirement is met by checking that each of the produced subsets contains recordings of every speaker available from the whole dataset. For example, if there are nine unique speakers in the dataset as a whole, then each train, test, and evaluation subset is required to have all nine speakers as well.

If either of these conditions are not met, Speakerbox retries this random sampling process. If we cannot find a valid recording holdout and speaker stratification configuration given the sampling iterations, we inform the user of this failure and ask them to add more examples to the dataset.

TODO WRITING -- INSERT FIG OF THIS?

TODO CODING -- allow param of "no holdout" and "no stratify" to ignore this process.

### Model Training

The Speakerbox training process consists of fine-tuning a pre-trained Wav2Vec2 speaker identification model [@yang2021superb] provided by Huggingface's Transformers library [@wolf-etal-2020-transformers]. The default model for fine-tuning ([superb/wav2vec2-base-superb-sid](https://huggingface.co/superb/wav2vec2-base-superb-sid)) was pre-trained on the VoxCeleb1 dataset [@Nagrani17].

## Model Application

TODO CODING -- WRITE APPLICATION FUNCTION / MOVE IT OVER TO THIS REPO FROM CDP-BACKEND AND ALSO SPLIT DEPS TO MAKE TRAINING AND APPLICATION DEPS DIFFERENT.

## Usage in Existing Research

We are utilizing Speakerbox trained models to annotate municipal council meeting transcripts provided by the Council Data Project [@Brown2021]. In our initial work thusfar, we first annotated ~10 hours of audio using the Gecko platform in ~12 hours of time, we then used our diarization and labeling method to annotate an additional ~21 hours of audio in ~6 hours of time. In total, the dataset was annotated and compiled in less than ~18 hours and contained ~31 hours of audio from meetings of the Seattle City Council. The model trained from the annotated dataset with the best precision and recall achieved 0.977 and 0.976 respectively. We additionally have used this model to annotate ~200 audio-aligned transcripts of Seattle City Council meetings and are now conducting our analysis of speaker behaviors and group dynamics.

## Future Work

There are a few additions that would greatly extend the functionality of Speakerbox that can be done in future work. First is the creation of a GUI for the application and workflow process. GUIs developed from pure Python have become much more accessible in recent years and a GUI would likely help non-computational scientists a great deal. We hope to additionally use Speakerbox produced models in a more "production" sense by directly integrating their use into the Council Data Project processing workflow so that transcripts are labelled immediately after they are produced. In doing so, we hope to create a template GitHub repository to generate a repository for storing annotation and audio files for which to use for automatically training a model using Continuous Integration systems.

# Acknowledgements

We wish to thank the University of Washington Information School for support. We wish to thank all the past and present contributors of the Council Data Project.

# References
