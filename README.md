# TFG
AI-based Intrusion Detection Systems for IoT Networks

[![Status](https://img.shields.io/badge/status-WIP-orange)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()


## Table of Contents

...

## Overview - Motivation

Credits:
    Sklearn
    [TabNet](https://github.com/dreamquark-ai/tabnet) 
    [TabPFN](https://github.com/phillipburgess/tabpfn)


## Project Structure

```plaintext
TFG/
├── src/
│   ├── data/
│   │   ├── ...
│   │   ├── cic_ids2017.py
│   │   ├── config.py
│   │   ├── utils.py
│   ├── models/
│   │   ├── API/
│   │   │   ├── ...
│   │   │   ├── llm.py
│   │   ├── trainable/
│   │   │   ├── ...
│   │   │   ├── ml.py
│   │   │   ├── tabnet.py
│   │   ├── ...
│   │   ├── config.py
│   ├── pipelines/
│   │   ├── ...
│   │   ├── pipelines.py
│   │   ├── utils.py
├── .gitgnore
├── README.md
├── develop_guide.md
└── .*.ipynb (example notebooks)
```

## How To Use

Based on modules ...


## Modules - Specs

The code is divided into three main categories: ``data``, ``models`` and ``pipelines``.

Each contains a series of modules that can be used and easily changed for research. For this purpose, ``data`` and ``models`` contain its own configuration file that allows the user to change the behaviour of its modules. The pipelines can be changed using the arguments of each method.

Also, this structure allows for great expandability, making very convenient the extension of functionalities.

### DATA

This category wraps all the modules related to the datasets. Each dataset counts with its own module, that allows for easy download, preprocessing and adaptation.

#### Modules hierarchy
```
BaseDataset
├── TabularDataset
│   └── CIC-IDS2017
└── (other possible types)
```

#### CIC-IDS2017

...

### MODELS

The models are divided into LLMs (API based models) and trainable models, like ML or DL models.

For further explanation the behaviour or construction of this models, look into the notebook examples or the specific docstrings.

#### Modules hierarchy
```
BaseModel
├── TrainableModel
│   ├── SklearnTrainableModel
│   │   ├── MLClassifier
│   │   │   ├── PreConfigured_DecisionTree
│   │   │   ├── PreConfigured_KNeighbors
│   │   │   ├── PreConfigured_LinearSVC
│   │   │   ├── PreConfigured_LogisticRegression
│   │   │   └── PreConfigured_RandomForest
│   │   ├── TabNetModel
│   │   └── TabPFNModel
│   └── (other kinds of models, like torch-based DL models)
└── LLModel
    ├── Mistral
    └── Gemini
```

#### Configurations

Some models can have their configuration or their training modified via arguments. However, all of them can be modified in full detail using the configuration file.

Below one can find the configuration parameters for every model explained.

##### TabNet Configuration Parameters


- ``TABNET_PRETRAINER_CONFIG``
    - **n_d**: Dimension of the prediction layer
    - **n_a**: Dimension of the attention layer
    - **n_steps**: Number of successive steps or layers in the network
    - **gamma**: Scaling factor for attention updates (usually between 1.0 and 2.0)
    - **n_independent**: Number of independent GLU layer in each GLU block
    - **n_shared**: Number of independent GLU layer in each GLU block
    - **lambda_sparse**: Sparsity loss parameter
    - **optimizer_fn**: Optimizer
    - **optimizer_params**: Parameters for the optimizer

- ``TABNET_PRETRAINING_PARAMS``
    - max_epochs: Maximum pretraining epochs
    - batch_size: Batch size. 
        - ⚠️ Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.
    - patience: Number of epochs with no improvement after which training will be stopped

- ``TABNET_CONFIG``
    - **n_d**: Dimension of the prediction layer
    - **n_a**: Dimension of the attention layer
    - **n_steps**: 5,
    - **gamma**: Scaling factor for attention updates (usually between 1.0 and 2.0)
    - **n_independent**: Number of independent GLU layer in each GLU block
    - **n_shared**: Number of independent GLU layer in each GLU block
    - **lambda_sparse**: Sparsity loss parameter
    - **optimizer_fn**: Optimizer
    - **optimizer_params**: Parameters for the optimizer
    - **scheduler_fn**: Scheduler
    - **scheduler_params**: Scheduler parameters
    - **mask_type**: masking function

- ``TABNET_TRAINING_PARAMS``
    - **eval_metric**: Evaluation metric
    - **max_epochs**: Maximum number of training epochs
    - **batch_size**: Batch size.
        - ⚠️ Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.
    - **patience**: Number of epochs with no improvement after which training will be stopped

##### ML Models Configuration Parameters

- ``LOGISTIC_REGRESSION_CONFIG``
    - **multi_class**: {'auto', 'ovr', 'multinomial'}
    - **max_iter**: Maximum number of iterations taken for the solvers to converge
    - **solver**: Algorithm to use in the optimization problem
    - **C**: Inverse of regularization strength; must be a positive float
    - **random_state**: Used when solver` == 'sag', 'saga' or 'liblinear' to shuffle the data

- ``LINEAR_SVC_CONFIG``
    - **C**: Inverse of regularization strength; must be a positive float
    - **random_state**: Used when solver` == 'sag', 'saga' or 'liblinear' to shuffle the data
    - **tol**: Tolerance for stopping criteria
    - **max_iter**: Maximum number of iterations taken for the solvers to converge

- ``RANDOM_FOREST_CONFIG``
    - **n_estimators**: The number of trees in the forest.
    - **max_depth**: The maximum depth of the tree
    - **max_features**: The number of features to consider when looking for the best split
    - **random_state**: Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node

- ``KNEIGHBORS_CONFIG``
    - **n_neighbors**: Number of neighbors to use

- ``DECISION_TREE_CONFIG``
    - **max_depth**: The maximum depth of the tree

##### TabPFN Configuration Parameters

- ``TABPFN_CONFIG``
    - **n_estimators**: The number of estimators in the TabPFN ensemble
    - **balance_probabilities**: Whether to balance the probabilities based on the class distribution in the training data
    - **average_before_softmax**:  Whether to average the predictions of the estimators before applying the softmax function
    - **device**: The device to use for inference with TabPFN
    - **ignore_pretraining_limits**: Whether to ignore the pre-training limits of the model
    - **inference_precision**: The precision to use for inference
    - **fit_mode**: Determines how the data is preprocessed and cached for inference
    - **memory_saving_mode**: Enable GPU/CPU memory saving mode
    - **random_state**: Controls the randomness of the model
    - **n_jobs**: The number of workers for tasks that can be parallelized across CPU cores

- ``TABPFN_EXPERT_CONFIG``
    - **CLASS_SHIFT_METHOD**: The method used to shift classes during preprocessing for ensembling to emulate the effect of invariance to class order
    - **FEATURE_SHIFT_METHOD**: The method used to shift features during preprocessing for ensembling to emulate the effect of invariance to feature position
    - **FINGERPRINT_FEATURE**: Whether to add a fingerprint feature to the data
    - **MAX_NUMBER_OF_CLASSES**: The number of classes seen during pretraining for classification / Size of the alphabet for the codebook
    - **MAX_NUMBER_OF_FEATURES**: The number of features that the pretraining was intended for
    - **MAX_NUMBER_OF_SAMPLES**: The number of samples that the pretraining was intended for
    - **MAX_UNIQUE_FOR_CATEGORICAL_FEATURES**: The maximum number of unique values for a feature to be considered categorical
    - **MIN_UNIQUE_FOR_NUMERICAL_FEATURES**: The minimum number of unique values for a feature to be considered numerical
    - **OUTLIER_REMOVAL_STD**: The number of standard deviations from the mean to consider a sample an outlier
    - **POLYNOMIAL_FEATURES**: The number of 2 factor polynomial features to generate and add to the original data before passing the data to TabPFN
    - **SUBSAMPLE_SAMPLES**: Subsample the input data sample/row-wise before performing any preprocessing and the TabPFN forward pass
    - **USE_SKLEARN_16_DECIMAL_PRECISION**: Whether to round the probabilities to float 16 to match the precision of scikit-learn

- ``TABPFN_PARAMS``
    - **predicting_batch_size**: Batch size for predicting using batches. Use -1 for single batch.

##### LLMs Configuration Parameters
- ``MISTRAL_API_KEY``: Mistral API key

- ``GEMINI_API_KEY``: Gemini API key

- ``BACKOFF_MAX_TRIES``: The maximum number of attempts to make before giving up

- ``BACKOFF_FACTOR``: The amount by which the backoff time increases on each successive backoff.

- ``GEMINI_GENERATION_CONFIG``
    - **top_p**: Tokens are selected from the most to least probable until the sum of their probabilities equals this value.
    - **temperature**: Value that controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more creative results.
    - **candidate_count**: Number of response variations to return.

- ``MISTRAL_GENERATION_CONFIG``
    - **top_p**: Nucleus sampling, where the model considers the results of the tokens with `top_p` probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.
    - **temperature**: What sampling temperature to use, we recommend between 0.0 and 0.7. Higher values like 0.7 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    - **n**: Number of completions to return for each request, input tokens are only billed once.

### PIPELINES

#### Modules hierarchy
```
BasePipeline
├── TTPipeline
└── TAPipeline
```