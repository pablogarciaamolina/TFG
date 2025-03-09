import os
import torch

ML_SAVING_PATH = os.path.join("models", "ml")
TABNET_SAVING_PATH = os.path.join("models", "tabnet")

TABNET_PRETRAINER_CONFIG = {
    "n_d": 10,
    "n_a": 10,
    "n_steps": 5,
    "gamma": 1.3,
    "n_independent": 4,
    "n_shared": 4,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.AdamW,
    "optimizer_params": {"lr": 2e-2},
    "mask_type": "entmax",
}
TABNET_PRETRAINING_PARAMS = {
    "max_epochs": 100,
    "batch_size": 1024,
    "patience": 10,  # Early stopping patience
}

TABNET_CONFIG = {
    "n_d": 10,
    "n_a": 10,
    "n_steps": 5,
    "gamma": 1.3,
    "n_independent": 4,
    "n_shared": 4,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.AdamW,
    "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 50, "gamma": 0.9},
    "mask_type": "entmax",
}
TABNET_TRAINING_PARAMS = {
    "eval_metric": ["accuracy"],
    "max_epochs": 200,
    "batch_size": 1024,
    "patience": 20,  # Higher patience for training
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
