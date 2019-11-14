# TensorFlow MNIST Project

## Overview
TensorFlow neural network implementation and training for classifying MNIST hand written images.

## Requirements
These are package versions the code was developed and tested on, it might work with earlier versions.
```python
h5py >= 2.9.0
keras >= 2.3.1
mlxtend >= 0.17.0
tensorflow >= 2.0.0
keras >= 2.3.1
numpy >= 1.17.0
```

## Project description
This post will be covering the two graphs that were set up in TensorFlow to process MNIST digit data, how training was conducted and finally how the results were converted into a tangible model to be leveraged down stream. This post is part of the [TensorFlow + Docker MNIST Classifier](http://theappliedarchitect.com/tensorflow-docker-mnist-classifier-project/) series.

## Data
Data was directly loaded from the official [mnist source](http://yann.lecun.com/exdb/mnist/)

## Objective
The code in this repo is used to create and train 2 seperate TensorFlow models and export the result in a format that can be served using a TensorFlow [docker container](https://www.tensorflow.org/tfx/serving/docker).

## Folder strcture
```python    
    .
    ├── data                        # Raw data folder, MNIST data should be extracted here
    ├── models                      # Trained model folder
    │   ├── classifier              # Classifier models
    │   |   ├── checkpoint          # Checkpoint model storage folder
    │   |   └── production          # Finalized model storage folder
    │   ├── autoencoder             # Autoencoder models
    |   |   ├── checkpoint          # Checkpoint model storage folder
    │   |   └── production          # Finalized model storage folder
    │   └── src                     # Source code
    |   |   ├── autoencoder         # Autoencoder graph code
    |   |   |   ├── config.py       # Config file with params and path info
    |   |   |   └── graph.py        # Graph strcture and training code
    │   |   └── classifier          # Classifier graph code
    |   |   |   ├── config.py       # Config file with params and path info
    |   |   |   └── graph.py        # Graph strcture and training code
    │   |   └── libs                # Custom helper objects
    |   |   |   ├── data.py         # MNIST data loading 
    |   |   |   └── visualizer.py   # Graphing functions
```

## Usage instructions
-- pull MNIST data
-- unzip MNIST data
-- train auto encoder
-- train classifier
-- grab models

