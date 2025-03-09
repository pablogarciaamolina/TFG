from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from src.models.base import BaseModel
from src.models.trainable.base import SklearnTrainableModel
from src.models.API.llm import LLModel
from src.models.trainable.tabnet import TabNetModel
from src.models.trainable.ml import MLClassifier

class BasePipeline(ABC):

    def __init__(self, model: BaseModel):
        
        self.model = model

    @abstractmethod
    def evaluate(self, x_test, y_test) -> dict:
        """
        Method for evaluating a classifier

        Args:
            x_test: Testing features
            y_test: Testing labels

        Returns:
            A dictionary containing the metrics and a figure with the report.
        """

        pass

    
    def evaluate_given_predictions(self,
        predictions,
        true_labels,
    ) -> dict:
        """
        Evaluating with predictions already obtained

        Args:
            predictions: Predictions
            true_labels: True labels
            classes: All possible labels for the data

        Returns:
            A dictionary containing the metrics and a figure with the report.
        
        """
        
        # Extract class names and metrics
        classes = np.sort(np.unique(true_labels))
        metrics = classification_report(y_true=true_labels, y_pred=predictions, target_names=classes, output_dict=True)
        precision = [metrics[target_name]['precision'] for target_name in classes]
        recall = [metrics[target_name]['recall'] for target_name in classes]
        f1_score = [metrics[target_name]['f1-score'] for target_name in classes]
        data = np.array([precision, recall, f1_score])

        # Generate heatmap report
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(data, cmap='Pastel1', annot=True, fmt='.2f', xticklabels=classes, yticklabels=['Precision', 'Recall', 'F1-score'], ax=ax)
        ax.set_title(f'Metrics Report ({self.model.name})')
        fig.tight_layout()

        accuracy = accuracy_score(true_labels, predictions)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "metrics_report": fig
        }


class TTPipeline(BasePipeline):
    """
    Tabular-Trainable Pipeline
    """

    model: SklearnTrainableModel

    def __init__(self, model: SklearnTrainableModel):
        super().__init__(model)

    def train(self,
        x_train,
        y_train,
        cv: int = 10,
        x_val = None,
        y_val = None
    ) -> None:
        
        if isinstance(self.model, TabNetModel):
            self.model.fit(x_train, y_train, x_val, y_val)
        elif isinstance(self.model, MLClassifier):
            self.model.fit(x_train, y_train, cv=cv, verbose=1)

    def evaluate(self, x_test, y_test) -> dict:
        
        pred = self.model.predict(x_test)

        results = self.evaluate_given_predictions(
            pred,
            y_test,
        )

        return results


class TAPipeline(BasePipeline):
    """
    Tabular-API Pipeline 
    """

    model: LLModel

    def __init__(self, model: LLModel):

        super().__init__(model)

    def evaluate(self,
        icl_data: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        task: str = "Behaving like a Intrusion Detection System, classifing the data based on the PC features in the possible classes of attacks or not attack",
        class_column: str = "Label",
    ) -> dict:
        """
        ...
        
        Args:
            icl_data: A pandas Dataframe containing the ICL examples
            test_data: A pandas Dataframe containing the test data to be clasified
            task: A sentence that further explains the classification task
            class_column: The name of the column containing the labels in the ``icl_data``

        Returns:
            A dictionary containing the metrics and a figure with the report.
        """

        preds = self.model.predict(
            icl_data,
            x_test,
            task,
            class_column
        )

        results = self.evaluate_given_predictions(
            preds,
            y_test,
        )

        return results