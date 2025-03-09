import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from src.data.utils import jsonize_rows
from src.models.tabnet import TabNetModel
from src.models.ml import MLClassifier

def classify_with_llm(
    llm,
    icl_data: pd.DataFrame,
    test_data: pd.DataFrame,
    task: str = "Classify data",
    class_column: str = "Label",
) -> list[str]:
    """
    Pipeline for classifying test input data using an LLM

    Args:
        llm: The API for LLM model
        icl_data: A pandas Dataframe containing the ICL examples
        test_data: A pandas Dataframe containing the test data to be clasified
        task: A sentence that further explains the classification task
        class_column: The name of the column containing the labels in the ``icl_data``

    Return:
        A list containing the labels for each of the test inputs in order.
    """
    
    icl_inputs = jsonize_rows(icl_data.drop(columns=["Label"]))
    context = (
        f"You are performing the following classification task: {task}\n\n"
        "Here are some examples:\n"
    )
    for i, o in zip(icl_inputs, icl_data[class_column]):
        context += str(
            {
                "Input": i,
                "Output": o
            }
        ) + "\n"
    context += f"Where Output is the classification label for each data entry."
    
    # INSTRUCTIONS
    test_inputs = jsonize_rows(test_data)
    pre_instruction = f"Classify the following input and provide the corresponding Output:\n"
    last_instruction = f"Your answer must only be the Output for the Input. Make sure you provide the raw Output, and nothing else."
    
    results = []
    for i in test_inputs:
        instructions = pre_instruction + i + "\n" + last_instruction
        response = llm.ask(instruction=instructions, context=context)
        results.append(response["answer"])
    
    return results


def train_and_evaluate(
    model,
    model_name: str,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 10
) -> dict:
    """
    Pipeline for training and evaluating a model, supporting both MLClassifier and TabNetModel.

    Args:
        model: The model instance (MLClassifier or TabNetModel)
        model_name: Name of the model
        x_train: Training features
        y_train: Training labels
        x_test: Testing features
        y_test: Testing labels
        cv: Number of cross-validation folds (only applies to MLClassifier). Defaults to 10.

    Returns:
        A dictionary containing the metrics and a figure with the report.
    """

    # Determine whether to pass validation data based on model type
    if isinstance(model, TabNetModel):
        _, _ = model.fit(x_train, y_train, x_test, y_test)
    elif isinstance(model, MLClassifier):
        model.fit(x_train, y_train, cv=cv, verbose=1)

    results_dict = evaluate(
        model,
        model_name,
        x_test,
        y_test
    )

    return results_dict

def evaluate(
    model,
    model_name: str,
    x_test,
    y_test
) -> dict:
    """
    Method for evaluating a model

    Args:
        model: The model instance (MLClassifier or TabNetModel)
        model_name: Name of the model
        x_test: Testing features
        y_test: Testing labels

    Returns:
        A dictionary containing the metrics and a figure with the report.
    """

    # Predictions
    pred = model.predict(x_test)

    # Extract class names and metrics
    target_names = model.model.classes_
    metrics = classification_report(y_true=y_test, y_pred=pred, target_names=target_names, output_dict=True)
    precision = [metrics[target_name]['precision'] for target_name in target_names]
    recall = [metrics[target_name]['recall'] for target_name in target_names]
    f1_score = [metrics[target_name]['f1-score'] for target_name in target_names]
    data = np.array([precision, recall, f1_score])

    # Generate heatmap report
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
