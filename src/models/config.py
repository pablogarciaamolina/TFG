import os

SAVING_PATH = os.path.join("models", "tabnet")

TABNET_PRETRAINING_PARAMS = {
    "max_epochs": 1,
    "batch_size": 1024
}

TABNET_TRAINING_PARAMS = {
    "eval_name": ['val'],
    "eval_metric": ["accuracy"],
    "max_epochs": 1,
    "batch_size": 1024,
}