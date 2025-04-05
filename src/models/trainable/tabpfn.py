import os
import numpy as np
import logging
import joblib

from tabpfn import TabPFNClassifier
from tabpfn.config import ModelInterfaceConfig

from ._base import SklearnTrainableModel
from src.models.config import TABPFN_CONFIG, TABPFN_SAVING_PATH, TABPFN_EXPERT_CONFIG


class TabPFNModel(SklearnTrainableModel):

    def __init__(self, name: str = "tabpnf"):

        expert_config = ModelInterfaceConfig(
            **TABPFN_EXPERT_CONFIG
        )
        model = TabPFNClassifier(**TABPFN_CONFIG, inference_config=expert_config)
        super().__init__(name, model)


    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Performs .fit method

        Args:
            x_train: Training data features
            y_train: Training datat labels
        """

        logging.info("Fitting TabPNF model...")
        self.model.fit(
            x_train,
            y_train
        )


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the given inputs

        Args:
            x: Iputs for which to predict the labels
        """

        return self.model.predict(x)
    

    def save(self):
        
        logging.info(f'Saving model...')

        filepath = os.path.join(TABPFN_SAVING_PATH, self.name + ".zip")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)        
        joblib.dump(self.model, filepath)
        logging.info(f'Model saved to {filepath}')
    

    def load(self):
        
        logging.info(f'Loading model...')

        filepath = os.path.join(TABPFN_SAVING_PATH, self.name + ".zip")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found.")
        
        self.model = joblib.load(filepath)
        self.model.device = "auto"
        logging.info(f'Model loaded from {filepath}')
    