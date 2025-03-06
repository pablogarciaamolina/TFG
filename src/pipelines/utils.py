import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from src.data.utils import jsonize_rows

def clasiffy_with_llm(
        llm,
        icl_data: pd.DataFrame,
        test_data: pd.DataFrame,
        class_column: str = "Label"
) -> list[list[str]]:

    test_X = test_data.drop(columns=[class_column])
    test_y = test_data[class_column]


    context = "You are a Intrusion Detection Classifier. Given this examples: \n"
    for json in jsonize_rows(icl_data):
        context += json + "\n"
    context += f"Where {class_column} is the classification of the example, classify the data that comes to you."

    text_list = jsonize_rows(test_X)
    instruction = f"Give me the {class_column} for each of the next data:\n"
    for t in text_list:
        instruction += t + "\n"


    response = llm.ask(instruction=instruction, context=context)

    ...


def train_and_evaluate_ml_model(
    model,
    model_name: str,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 10
) -> dict:
    """
    Pipeline for training and evaluating a ML model

    Args:
        model: ML Model
        model_name: Name of the model
        x_train: Training features
        y_train: Training labels
        x_test: Testing features
        y_test: Testing labels
        cv: Number of cross-validation folds. Defaults to 10.

    Returns:
        A dictionary containing the metrics and a figure with the report
    """
    
    model.fit(x_train, y_train, cv=cv, verbose=1)
    pred = model.predict(x_test)

    target_names = model.model.classes_
    metrics = classification_report(y_true=y_test, y_pred=pred, target_names=target_names, output_dict=True)
    precision = [metrics[target_name]['precision'] for target_name in target_names]
    recall = [metrics[target_name]['recall'] for target_name in target_names]
    f1_score = [metrics[target_name]['f1-score'] for target_name in target_names]
    data = np.array([precision, recall, f1_score])

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(data, cmap='Pastel1', annot=True, fmt='.2f', xticklabels=target_names, yticklabels=['Precision', 'Recall', 'F1-score'], ax=ax)
    ax.set_title(f'Metrics Report ({model_name})')
    fig.tight_layout()

    accuracy = accuracy_score(y_test, pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "metrics_report": fig
    }

def train_and_evaluate_tabnet_model(
    model,
    model_name: str,
    x_train,
    y_train,
    x_test,
    y_test,
) -> dict:
    """
    Pipeline for training and evaluating a ML model

    Args:
        model: Model
        model_name: Name of the model
        x_train: Training features
        y_train: Training labels
        x_test: Testing features
        y_test: Testing labels

    Returns:
        A dictionary containing the metrics and a figure with the report
    """
    
    train_acc, val_acc = model.fit(x_train, y_train, x_test, y_test)

    ...
    


def plot_accuracies(
    scores: list[float],
    names: list[str]
) -> plt.Figure:
    """
    Function for plotting a report of accuracies

    Args:
        scores: List containing the scores of the models
        names: Names of the models from which the accuracies where obtained

    Returns:
        A figure containing the report
    """
    
    assert len(scores) == len(names), f"{len(scores)} scores and {len(names)} provided"
    
    palette = sns.color_palette('Blues', n_colors = 2)
    fig, ax = plt.subplots(figsize = (9, 3))
    ax.barh(names, scores, color = palette)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Accuracy Score')
    ax.set_title('Models Comparison')

    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, str(round(v, 3)), ha = 'left', va = 'center')

    return fig
