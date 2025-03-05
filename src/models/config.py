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
    "multi_class": "multinomial",
    "max_iter": 15000, "solver": 'sag',
    "C": 100,
    "random_state": 0
}

LINEAR_SVC_CONFIG = {
    "C": 1,
    "random_state": 0,
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 15, 
    "max_depth": 8, 
    "max_features": 20, 
    "random_state": 0
}

KNEIGHBORS_CONFIG = {
    "n_neighbors": 8
}

DECISION_TREE_CONFIG = {
    "max_depth": 8
}
