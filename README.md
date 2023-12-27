# LSTMIX: Multivariate multistep prediction LSTM framework for multidimensional mixing performance metrics.

## Overview

This Python repository provides a comprehensive framework for preprocessing multivariate and multidimensional mixing performance metrics and statistics, as well as constructing a multistep LSTM RNN using PyTorch. The framework includes scripts for data generation, model training, hyperparameter tuning, and evaluation of model performance.

## Repository Structure

The repository is organized into several key components:

### 1. `datagen.py`

This script is responsible for generating and preprocessing datasets for training, validation, and testing. It utilizes two additional scripts:

- `input.py`: Handles the import, organization, scaling, and smoothing of raw data from DNS simulations stored in CSV files.
- Windowing class from `modeltrain_LSTM.py`: Used for augmenting data by windowing and packaging it into corresponding pickle (pkl) files with labels and relevant data for later use.

### 2. `modeltrain_LSTM.py`

This script contains essential classes and functionalities for model training. It includes:

- Windowing class: Implements data windowing for augmentation.
- LSTM classes: Provides implementations for both fully connected and encoder-decoder architectures.
- Model training logic: Conducts the main training process, saving model states, trained datasets, and the hyperparameters used during training.

### 3. `hyperparam_tuning.py`

This script leverages Ray Tune to perform hyperparameter tuning on the LSTM architectures defined in `modeltrain_LSTM.py`. It aims to optimize the model's performance by systematically exploring hyperparameter combinations.

### 4. `rollout_prediction.py`

This script facilitates the evaluation of the trained model. It includes functionalities to:

- Plot trained and validated datasets.
- Execute a rollout operation to predict values from the test set and compare them against the ground truth.
- Plot various metrics such as a y=x plot, Wasserstein and K-L divergence plots, and more.

## Directory Structure

```plaintext
LSTMIX/
│
├── datagen.py
├── hyperparam_tuning.py
├── modeltrain_LSTM.py
├── rollout_prediction.py
├── README.md
├── requirements.txt
├── figs/
│   ├── input_data/
│   ├── rollouts/
│   ├── windowed/
│   ├── split_data_temporal_EMD/
│   └── temporal_dist/
│
├── input_data/
│
├── trained_models/
│   ├── data_sets_DMS/
│   └── data_sets_S2S/
│
├── tuning/
│   ├── best_models/
│   ├── DMS/
│   └── S2S/
│
└── RawData/  # Not part of the repository, user data

## Getting Started

To use this framework, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Execute the data generation script: `python datagen.py`
4. Train the LSTM model: `python modeltrain_LSTM.py`
5. Tune hyperparameters (optional): `python hyperparam_tuning.py`
6. Evaluate model performance: `python rollout_prediction.py`

## Dependencies

- PyTorch
- Ray Tune
- NumPy
- Matplotlib
- Other dependencies specified in `requirements.txt`

## Acknowledgments

- The developers and contributors of PyTorch, Ray Tune, and other open-source libraries used in this framework.
