import os
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.metrics import accuracy_score, roc_auc_score

from .config import TABNET_PRETRAINING_PARAMS, TABNET_TRAINING_PARAMS, TABNET_SAVING_PATH



class TabNetModel:

    def __init__(self, pretrain: bool = True) -> None:
        """
        Constructor for the class

        Args:
            pretrain: Whether to perform, pretraining or not
            name: Name for the model
        """
        
        self.pretrainer = TabNetPretrainer() if pretrain else None
        self.model = TabNetClassifier()


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
        """
        Trianing pipeline for the model. Performs pretreining (if specified) and then trains the model based on the configuration paramenters

        Args:
            X_train: Training data features
            y_train: Trianing datat labels
            X_val: Validation data features
            y_val: Validation data labels

        Return:
            The training and validation accuracy final metrics for the model
        """

        # Pretraining
        logging.info("Pretraining for TabNet...")
        self.pretrainer.fit(
            X_train,
            eval_set=[X_val],
            **TABNET_PRETRAINING_PARAMS
        )

        # Training
        logging.info("Training TabNet model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            **TABNET_TRAINING_PARAMS,
            from_unsupervised=self.pretrainer,
        )

        # Evaluate training
        y_train_pred = self.predict(X_train)
        y_val_pred = self.predict(X_val)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        logging.info(f"Train Accuracy: {train_accuracy:.4f}")
        logging.info(f"Validation Accuracy: {val_accuracy:.4f}")

        return train_accuracy, val_accuracy
    
    def predict(self, X) -> np.ndarray:
        """
        Method for predicting samples with the trained model
    
        Args:
            X: The data matrix for which we want to get the predictions.

        Returns:
            ndarray of shape (n_samples,). Vector containing the class labels for each sample.
        """

        return self.model.predict(X)

    def plot_metrics(self) -> None:
        """
        Plots the metrics for the last training done to the model.
        NOTE: Not having trained the model will raise an error. Loading the model from memory does not include its past training history.
        """
        
        plt.plot(self.model.history['loss'])
        # plt.plot(self.model.history['val'])
        plt.plot(self.model.history['lr'])

    def save(self, name: str = None) -> None:
        """
        Saves the model on memory following the saving path and the name for the model
        """

        name = name if name else f"tabnet_{time.time()}"
        path = os.path.join(TABNET_SAVING_PATH, name)

        self.model.save_model(path)

    def load(self, name: str) -> None:
        """
        Loads the model from memory using its name
        """

        path = os.path.join(TABNET_SAVING_PATH, name + ".zip")
        self.model.load_model(path)

    def evaluate(self, X_test, y_test) -> tuple[float, float]:
        """
        Evaluates performance of the model over a test set

        Args:
            X_test: Test data features
            y_test: Test data labels

        Returns:
            A tuple containing the accuracy and roc auc score
        """

        logging.info("Evaluating TabNet model...")
        
        y_pred = self.model.predict(X_test) 
        y_proba = self.model.predict_proba(X_test)
        
        all_classes = np.arange(y_proba.shape[1])
        y_test_one_hot = label_binarize(y_test, classes=all_classes)

        logging.info(f"Unique values in y_test: {np.unique(y_test)}")
        logging.info(f"First 5 rows of y_proba:\n {y_proba[:5]}")
        logging.info(f"Sum of probabilities per sample: {np.sum(y_proba, axis=1)[:5]}")

        test_accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        test_auc = roc_auc_score(y_test_one_hot, y_proba, multi_class="ovr")
        logging.info(f"Test auc score: {test_auc}") # NOTE: If not all classes are present then the ROC AUC score will be 'nan'

        return test_accuracy, test_auc

    def explain(self, X) -> None:
        """
        Local explanation method for visualizing generated masks over a features set

        Args:
            X: Data features to use for explanation
        """

        explanations, masks = self.model.explain(X)

        fig, axs = plt.subplots(1, 3, figsize=(20,20))
        for i in range(3):
            axs[i].imshow(masks[i][:50])
            axs[i].set_title(f"mask {i}")

