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

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

LOGISTIC_REGRESSION_CONFIG = {
    "multi_class": "multinomial"
}

LINEAR_SVC_CONFIG = {
    ...
}

RANDOM_FOREST_CONFIG = {
    ...
}

KNEIGHBORS_CONFIG = {
    ...
}

DECISION_TREE_CONFIG = {
    ...
}
