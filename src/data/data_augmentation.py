import logging
import numpy as np
from pprint import pformat
from collections import Counter
from typing import Optional

import torch

from src.data.config import SMOTE_CONFIG


def smote(x: np.ndarray, y: np.ndarray, threshold: Optional[float] = None, n: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Specifically for DL model, initiall conceived for TabNET. Performs data augmentation for clasification using a modified version of SMOTE.

    Given the data divided into features and labels, the data augmentation will be performed to those classes that have a certain percentange of samples 
    less than the class with the higher number of samples. The augmentation will be performed `augmentations` times, increasing the amount of data for the class each time.

    Args:
        x: The features of the data.
        y: The labels of the data associate dto the features.
        threshold: The percentage that indicates whether to augment a class samples or not depending of the class with the most number of samples. (0, 1].
        n: The amount of times to augment the data.

    Returns:
        The tuple feeatures-labels of the original data plus the augmented samples.
    """

    logging.info("Performing SMOTE for data augmentation...")

    if not threshold:
        threshold = SMOTE_CONFIG["class_samples_threshold"]
    
    assert isinstance(threshold, float)
    assert 0 < threshold <= 1

    if not n:
        n = SMOTE_CONFIG["n"]

    assert isinstance(n, int)

    # Obtain labels and their counts as well as the max amount of samples in a class
    unique_labels, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    max_n_samples = max(counts)

    # Get the threshold value classes for augmentation
    threshold_value = int(max_n_samples * threshold)
    augmentation_classes = [label for label, count in label_counts.items() if count < threshold_value]

    # Perform augmentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmented_x = torch.from_numpy(x.copy()).to(device)
    augmented_y = y.copy()

    logging.info("Starting Data Augmentation...")
    for _ in range(n):

        new_x, new_y = _classification_SMOTE_step(
            torch.from_numpy(x.copy()).to(device),
            y.copy(),
            SMOTE_CONFIG["alpha"],
            SMOTE_CONFIG["beta"],
            augmentation_classes,
            seed=np.random.randint(1, 200)
        )

        augmented_x = torch.cat((augmented_x, new_x), dim=0)
        augmented_y = np.concatenate((augmented_y, new_y), axis=0)

        # Update counts and classes being augmented
        new_unique_labels, new_counts = np.unique(y, return_counts=True)
        new_label_counts = dict(zip(new_unique_labels, new_counts))
        label_counts = dict(Counter(label_counts) + Counter(new_label_counts))
        augmentation_classes = [label for label, count in label_counts.items() if count < threshold_value]

    unique_classes, counts = np.unique(augmented_y, return_counts=True)
    class_counts = {cls: int(count) for cls, count in zip(unique_classes, counts)}
    logging.info("Final counts after Data Augmentation:\n%s", pformat(class_counts))

    return augmented_x.cpu().numpy(), augmented_y

def _classification_SMOTE_step(
    x: torch.torch.Tensor,
    y: np.ndarray,
    alpha: float,
    beta: float,
    augmentation_classes: list[str],
    seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """
    Personalized version of SMOTE based on the pytorch-tabnet library implementation ClassificationSMOTE.

    Specifically designed to augment all the samples based on specific classes and not a percentage of the data, at the same time that the augmented samples can utilize any of the samples 
    from classes that are not being augmented.

    Args:
        x: Features samples
        y: Labels associated to the features samples
        alpha: alpha for the beta distribution
        beta: beta for the beta distribution
        augmentation_classes: List of classes being augmented
        seed: Seed for random numbers

    Returns:
        The tuples features-labels of augmented data (not including the original samples).
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size = x.shape[0]
    device = x.device
    mask_to_change = np.isin(y, augmentation_classes)
    mask_to_change_t = torch.tensor(mask_to_change, device=device)

    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas = torch.from_numpy(np_betas).to(device).float()
    index_permute = torch.randperm(batch_size, device=device)

    x[mask_to_change_t] = random_betas[mask_to_change_t, None] * x[mask_to_change_t]
    x[mask_to_change_t] += (1 - random_betas[mask_to_change_t, None]) * x[index_permute][mask_to_change_t].view(x[mask_to_change_t].size())  # noqa

    return x[mask_to_change_t], y[mask_to_change]


