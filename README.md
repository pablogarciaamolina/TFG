# TFG
Intrusion Detection Systems for IoT Networks


## Table of Contents

...

## Overview

...

## Project Structure

```plaintext
TFG/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cic_ids2017.py
│   │   ├── config.py
│   │   ├── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ml.py
│   │   ├── tabnet.py
│   │   ├── lmm_api.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── utils.py
├── notebooks/
│   ├── ...
├── .gitgnore
├── README.md
├── develop_guide.md
```


## CODE - Use and Specs

### DATA

...

### MODELS

...

#### Configurations

The models include a predifinable configuration, modifiable via the configuration file.

##### TabNet

``TABNET_PRETRAINING_PARAMS``
- max_epochs: Maximum pretraining epochs
- batch_size: Batch size. Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.

``TABNET_TRAINING_PARAMS``
- eval_name: Evaluation name
- eval_metric: Evaluation metric
- max_epochs: Maximum number of training epochs
- batch_size: Batch size. Note that if the batch size is higher than the amount of data samples provided (both training or validation) an error will be raised.

##### ML Models

...

##### LLMs


...

### PIPELINES

...