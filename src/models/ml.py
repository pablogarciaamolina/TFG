import os
import time
import numpy as np
import logging
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from .config import ML_SAVING_PATH, LOGISTIC_REGRESSION_CONFIG, LINEAR_SVC_CONFIG, RANDOM_FOREST_CONFIG, KNEIGHBORS_CONFIG, DECISION_TREE_CONFIG

class MLClassifier:
    """
    A general machine learning classifier wrapper supporting multiple scikit-learn models.
    """

    def __init__(self, model_class, **kwargs):
        """
        Constructor for the class

        Args:
            model_class: The classifier class from scikit-learn.
            **kwargs: Additional parameters to pass to the model.
        """

        if not issubclass(model_class, BaseEstimator):
            raise ValueError("Provided model_class must be a subclass of BaseEstimator")
        
        self.model = model_class(**kwargs)
        self.cross_val_scores = None
        self.trained = False

    def fit(self, X, y, cv: int = 10, scoring: str = 'accuracy', verbose: int = 0):
        """
        Fits the model using the given dataset and performs cross-validation.

        Args:
            X: Feature matrix.
            y: Target vector.
            cv: Number of cross-validation folds. Defaults to 10.
            scoring: Scoring metric for evaluation. Defaults to 'accuracy'.
            verbose: Verbosity level for logging. Defaults to 0.
        """

        self.model.fit(X, y)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        self.cross_val_scores = cross_val_score(self.model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        self.trained = True

        if verbose > 0:
            logging.info(f'Cross-validation scores: {self.cross_val_scores}')
            logging.info(f'Mean cross-validation score: {self.cross_val_scores.mean():.4f}')

    def predict(self, X) -> np.ndarray:
        """
        Method for predicting samples with the trained model
    
        Args:
            X: The data matrix for which we want to get the predictions.

        Returns:
            ndarray of shape (n_samples,). Vector containing the class labels for each sample.
        """

        assert self.trained, "Model not trained yet"

        return self.model.predict(X)
    
    def save_model(self, name: str = None):
        """
        Saves the trained model to the specified file.

        Args:
            name: Name under which the model should be saved.
        """

        if not self.trained:
            raise RuntimeError("Model must be trained before saving.")
        
        name = name if name else self.__class__.__name__ + f"{time.time()}"
        filepath = os.path.join(ML_SAVING_PATH, name + ".zip")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)        
        joblib.dump(self.model, filepath)
        logging.info(f'Model saved to {filepath}')

    @staticmethod
    def load_model(name: str):
        """
        Loads a saved model from the specified file.

        Args:
            name: Name of the saved model file.

        Returns:
            Loaded model instance.
        """
        
        filepath = os.path.join(ML_SAVING_PATH, name + ".zip")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found.")
        
        model = joblib.load(filepath)
        logging.info(f'Model loaded from {filepath}')
        return model
    
class PreConfigured_LogisticRegression(MLClassifier):

    def __init__(self):
        super().__init__(LogisticRegression, **LOGISTIC_REGRESSION_CONFIG)

class PreConfigured_LinearSVC(MLClassifier):

    def __init__(self):
        super().__init__(LinearSVC, **LINEAR_SVC_CONFIG)

class PreConfigured_RandomForest(MLClassifier):

    def __init__(self):
        super().__init__(RandomForestClassifier, **RANDOM_FOREST_CONFIG)

class PreConfigured_KNeighbors(MLClassifier):

    def __init__(self):
        super().__init__(KNeighborsClassifier, **KNEIGHBORS_CONFIG)

class PreConfigured_DecisionTree(MLClassifier):

    def __init__(self):
        super().__init__(DecisionTreeClassifier, **DECISION_TREE_CONFIG)