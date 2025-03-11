# TFG
Intrusion Detection Systems for IoT Networks


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
│   │   ├── __init__.py
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
│   │   ├── utils.py
├── .gitgnore
├── README.md
├── develop_guide.md
├── .*.ipynb (example notebooks)
```




## Modules - Use and Specs

The code is divided into three main categories: ``data``, ``models`` and ``pipelines``. Each contain the series of modules that can be used and easily changed for research. Also, this structure allows for great expandability, making very convenient the extension of functionalities.

### DATA

This category wraps all the modules related to the datasets. Each dataset counts with its own module, that allows for easy download, preprocessing and adaptaion.

Modules hierarchy:
```
...
```

#### CIC-IDS2017

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