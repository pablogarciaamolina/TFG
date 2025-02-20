import logging
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
    
    def save_model(self, filepath):
        """
        Saves the trained model to the specified file.

        Args:
            filepath: Path where the model should be saved.
        """

        if not self.trained:
            raise RuntimeError("Model must be trained before saving.")
        joblib.dump(self.model, filepath)
        logging.info(f'Model saved to {filepath}')

    @staticmethod
    def load_model(filepath):
        """
        Loads a saved model from the specified file.

        Args:
            filepath: Path to the saved model file.

        Returns:
            Loaded model instance.
        """
        
        model = joblib.load(filepath)
        logging.info(f'Model loaded from {filepath}')
        return model
