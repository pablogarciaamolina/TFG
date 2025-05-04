# TFG
AI-based Intrusion Detection Systems for IoT Networks


## Table of Contents

...

## Overview - Motivation

...


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
├── .*.ipynb (example notebooks)
```


## Modules - Use and Specs

The code is divided into three main categories: ``data``, ``models`` and ``pipelines``.

Each contain the series of modules that can be used and easily changed for research. For this purpuse, each module counts with its own configuration file that allows the user to change the behaviour of the modules.

Also, this structure allows for great expandability, making very convenient the extension of functionalities.

### DATA

This category wraps all the modules related to the datasets. Each dataset counts with its own module, that allows for easy download, preprocessing and adaptaion.

Modules hierarchy:
```
- BaseDataset
    - TabularDataset
        - CIC-IDS2017
    - (other possible types)
```

#### CIC-IDS2017

...

### MODELS

The models are divided into LLMs (API based models) and trainable models, like ML or DL models.

Modules hierarchy:
```
- BaseModel
    - TrainableModel
        - SklearnTrainableModel
            - MLClassifier
                - PreConfigured_DecisionTree
                - PreConfigured_KNeighbors
                - PreConfigured_LinearSVC
                - PreConfigured_LogisticRegression
                - PreConfigured_RandomForest
            - TabNetModel
            - TabPFNModel
        - (other kinds of models, like torch-based DL models)
    - LLModel
        - Mistral
        - Gemini
```

For further explanation the behaviour or construction of this models, look into the notebook examples or the specific docstrings.

#### Configurations

Some models can have their configuration or their training modified via arguments. However, all of them can be modified in full detail using the configuration file.

Below one can find the configuration parameters for every model explained.

##### TabNet


``TABNET_PRETRAINER_CONFIG``
- n_d: Dimension of the prediction layer
- n_a: Dimension of the attention layer
- n_steps: Number of succesive steps or layers in the network
- gamma: Scaling factor for attention updates (ussually between 1.0 and 2.0)
- n_independent: Number of independent GLU layer in each GLU block
- n_shared: Number of independent GLU layer in each GLU block
- lambda_sparse: Sparsity loss parameter
- optimizer_fn: Optimizer
- optimizer_params: Parameters for the optimizer

``TABNET_PRETRAINING_PARAMS``
- max_epochs: Maximum pretraining epochs
- batch_size: Batch size. Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.
- patience: Number of epochs with no improvement after which training will be stopped

``TABNET_CONFIG``
- n_d: Dimension of the prediction layer
- n_a: Dimension of the attention layer
- n_steps: 5,
- gamma: Scaling factor for attention updates (ussually between 1.0 and 2.0)
- n_independent: Number of independent GLU layer in each GLU block
- n_shared: Number of independent GLU layer in each GLU block
- lambda_sparse: Sparsity loss parameter
- optimizer_fn: Optimizer
- optimizer_params: Parameters for the optimizer
- scheduler_fn: Scheduler
- scheduler_params: Scheduler parameters
- mask_type: masking function

``TABNET_TRAINING_PARAMS``
- eval_metric: Evaluation metric
- max_epochs: Maximum number of training epochs
- batch_size: Batch size. Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.
- patience: Number of epochs with no improvement after which training will be stopped

##### ML Models

...

##### LLMs


...

### PIPELINES

...