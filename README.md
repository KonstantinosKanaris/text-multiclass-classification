# Multi-class Image Classification with PyTorch

## Table of Contents

* [Overview](#Overview)
* [Key Features](#Key--Features)
* [Development](#Development)
* [Data Preparation](#Data--Preparation)
* [Experiment Configuration](#Experiment--Configuration)
* [Training](#Training)
* [Experiment Tracking](#Experiment--Tracking)
* [TODOs](#TODOs)

## Overview 🔍

A simple framework for text multi-class classification projects using PyTorch.

### Project Structure 🌲
```
text-multiclass-classification
├── .gitignore
├── .pre-commit-config.yaml           | Pre-commit hooks
├── LICENSE
├── Makefile                          | Development commands for code formatting, linting, etc
├── Pipfile                           | Project's dependencies and their versions using the `pipenv` format
├── Pipfile.lock                      | Auto-generated file that locks the dependencies to specific versions for reproducibility
├── README.md
├── colours.mk                        | A Makefile fragment containing color codes for terminal output styling
├── configs
│   └── experiments.yaml        | Experiment configuration file. Define data paths, hyperparameters for each experiemnt
├── mypy.ini
└── text_multiclass_classification    | The main Python package containing the project's source code
    ├── __about__.py                  | Metadata about the project, i.e., version number, author information, etc
    ├── __init__.py
    ├── __main__.py                   | The entry point for running the package as a script
    ├── datasets
    │   ├── __init__.py
    │   ├── ag_news.py
    │   └── utils.py
    ├── engine
    │   ├── __init__.py
    │   └── trainer.py          | Contains the training process
    ├── factories
    │   ├── __init__.py
    │   ├── client.py          | Interacts with the factories to return different instances of models, optimizers, and transforms
    │   └── factories.py       | Contains factory classes for creating different types of models, optimizers, and transforms
    ├── logger
    │   └── logging.ini         | Configuration file for Python's logging module
    ├── models                        | A sub-package containing definitions for different model architectures
    │   ├── __init__.py
    │   └── news_classifier.py
    └── utils
        ├── __init__.py
        ├── aux.py                   | Auxilliary functions/classes such as EarlyStopping
        ├── constants.py             | Defines constants
        ├── custom_exceptions.py     | Implements custom exceptions
        ├── embeddings.py            | Wrapper for pre-trained embeddigns and their use
        ├── error_messages.py        | Contains custom error messages
        ├── vectorizers.py           | Vectorizes a text sequence to numbers
        └── vocabulary.py
```


## Key Features 🔑

* **Customizable Experiments**: Define multiple experiments easily by configuring model
architecture, optimizer, learning rate, batch size, and other hyperparameters in a YAML
configuration file
* **Customizable Models**: Easily integrate custom PyTorch models into your classification
tasks, allowing for seamless experimentation with novel architectures and configurations
* **Experiment Tracking**: Utilize TensorBoard for real-time visualization of training
metrics and performance evaluation, enabling easy monitoring of experiment progress
* **Checkpointing**: Ensure training progress is saved with checkpointing functionality,
allowing for easy resumption of training from the last saved state
* **EarlyStopping**: Automatically stop training when the model's performance stops
improving on a validation set

## Development 🐍
Clone the repository:
  ```bash
  $ git clone https://github.com/KonstantinosKanaris/text-multiclass-classification.git
  ```

### Set up the environment

#### Create environment
Python 3.10 is required.

- Create the environment and install the dependencies:
    ```bash
    $ pipenv --python 3.10
    $ pipenv install --dev
    ```
- Enable the newly-created virtual environment, with:
    ```bash
    $ pipenv shell
    ```

## Experiment Configuration 🧪
To conduct training experiments, define the configuration parameters for each experiment, including data paths and
hyperparameters, in the configuration (YAML) file. Here's how you can define experiments:

```yaml
experiments:
  -
    name: experiment_1
    data:
        train_csv: ./data/train.csv
    hyperparameters:
      general:
        num_epochs: 10
        batch_size: 32
      optimizer:
        optimizer_name: adam
        learning_rate: 0.001
        weight_decay: 0.3
      early_stopping:
        patience: 3
        delta: 0.008
      model:
        model_name: news_classifier_cnn
        embeddings:
  -
    name: experiment_2
    data:
      train_csv: ./data/train.csv
    hyperparameters:
      general:
        num_epochs: 2
        batch_size: 32
      optimizer:
        optimizer_name: adam
        learning_rate: 0.002
        weight_decay: 0
      early_stopping:
        patience: 5
        delta: 0
      model:
        model_name: news_classifier_rnn
        embeddings: ./embeddings/glove.6B.100d.txt
  ...
```

## Training 🚀
*Command Line*
>
>From the root directory of the project execute:
>```bash
>$ python -m text_multiclass_classification train --config ./configs/experiments.yaml
>```
>To resume training from a saved checkpoint execute:
>```bash
>$ python -m text_multiclass_classification train --config ./configs/experiments.yaml --resume_from_checkpoint yes
>```
>where the checkpoint directory path is defined in the configuration file.

## Experiment Tracking 📉
>Track your experiments with MLFlow by executing:
>```bash
>$ mlflow ui
>```
>
