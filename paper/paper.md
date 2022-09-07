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

ML generated transcripts are usually produced without speaker annotations, at best they are diarized and show "Speaker I"

Additionally, there is no consistenty in labels across meetings. Using diarization, in two different meetings, "Speaker 0" may be two entirely different people.

This hinders both the end user review of these transcripts in an archival setting and the downstream computational analysis of these transcripts.

There exist methods for fine tuning multi-speaker classification models and for diarization but no tools are available packaged together to make the process simpler.

## Related Work

pyannote-audio

transformers ?

auto-ml?

## Workflow

### Bootstraping with Diarization

Diarization is an unsupervised process of clustering speaker audio chunks together based off of the features of those chunks. -- TODO BETTER EXPLANATION AND CITATION

Diarization as a method for bootstrapping multi-speaker classification models is quite common (TODO -- CITATIONS) and while there are many resources about how diarization bootstrapping can be done, there hasn't been work in combining tools together to make a cohesive, easy-to-use system for diarization and model generation.

Diarization however is known to mis-classify or miss entirely certain edge cases where two or more speakers sound incredibly alike resulting in such speakers being labeled together (TODO -- CITE).

We make use of diarization as a method for bootstrapping a multi-speaker classification dataset for model training and provide methods to diarize audio directly, as well as guidance on how to use the diarized audio for later training.

(TODO -- MAYBE CODE SNIPPET)

### Using Gecko Annotations

Gecko (TODO -- CITE) is a free web application for annotating audio and video clips (TODO -- ANNOTATING WITH WHAT?). 

Using Gecko, users can easily annotate speakers and the portions in which they speak. All of these annotations can then be exported to a JSON file for downstream processing.

We make use of Gecko annotations as a method for dataset creation by providing functions to split and prepare audio files using the annotations from a Gecko JSON file.

### Combining Diarized Audio Datasets and Gecko Annotation Datasets

As diarization can sometimes perform poorly in edgecases where multiple speakers sound similar, it may be useful to some users to combine a bootstrapped diarized audio dataset with a gecko annotation dataset, specifically using the Gecko annotation dataset as method for ensuring that speakers are labelled correctly and more fully.

We make combining these two datasets simple due to the fact that the outputs from both the Gecko dataset preparation and the diarized bootstrapped dataset preparation are the same format such that the output DataFrames can simply be concattenated together to form a larger, more complete dataset.

### Data Preparation

Speakerbox is built with the goal of making multi-meeting, multi-speaker classification model training as simple and as safe as possible. To accomplish this, we have baked in some minimum standards that a dataset must meet to be allowed for training and these standards are checked during data preparation time.

To ensure that the model is learning the features of the speaker's voice and not the features of, perhaps, the microphone, or the context of the meeting, we check and create dataset splits based off of a holdout and stratify pattern.

We first select three random subsets of meetings for training, evaluation, and testing. Each subset must holdout by meeting id and have an entirely different set of meetings, this is to reduce the chance of learning the features of specific microphones or meeting contexts.

We then check that speaker stratifaction is met by checking that each of the produced subsets contains every speaker available from the whole dataset. For example, if there are nine unique speakers in the dataset as a whole, then each meeting subset should have all nine speakers as well regardless of how they are spread across the subset's meetings.

If either of these conditions aren't met we retry this random meeting subset sampling multiple times and if we cannot find a valid meeting subset configuration given the sampling iterations, we raise a warning to the user informing them of such and that they should try adding more data to the dataset.

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
