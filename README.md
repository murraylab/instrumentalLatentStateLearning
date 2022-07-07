# MultiLatentStateInstrumentalLearning
Copyright 2021, Warren Woodrich Pettine

This is a preliminary code base associated with:

Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D. “Human latent-state generalization through prototype 
learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code base is under active development, and is being made available for reviewers. 

For examples of running the code base, see the examples.py file in the main directory.
# Installation

All code was written using python version 3.7. Please see the requirements file for required packages. To install the
requirements, just run. 

```bash
pip install -r requirements.txt
```

# Organization
The code is organized in the following packages.

- simulations
- subjects
- utils
- analysis
- plots

See package modules for details on each function. 

In addition, the main directory contains the module 'examples.py' that runs the algorithmic model on relevant experiments.  

## simulations
The simulations package contains all code necessary to simulate the various models. It is organized in the following
modules:

- algorithmicmodels
- neuralnetwork
- parameters
- experiments

### algorithmicmodels.py
This module contains the main code for the prototype and exemplar algorithmic models. 

### neuralnetwork.py
This module contains the code to run the neural network simulations. This module is unfinished. 

### parameters.py
Module for creating parameters dictionaries to create agents, networks and the worlds with with they interact. 

### experiments.py
Module that creates specific experiments used in the paper. 

## utils
This package contains the module utils.py, which organizes Various useful functions, such as those for unit testing. 

## analysis
This package contains Code for higher-level analyses of behavior and model fun

- general
- plotfunctions

### general.py
Functions used for general analysis

### plotfunctions.py
Functions used to plot the data
