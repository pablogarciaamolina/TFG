def accuracy(y_true, y_pred) -> float:
    """
    Generic function for calculating accuracy given two sets of objects

    Args:
        y_pred: Predicted labels
        y_true: True labels
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    return correct / total if total > 0 else 0
