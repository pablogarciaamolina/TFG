import os
import glob
import numpy as np
import logging
import re
from pprint import pformat
from collections import Counter
from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from src.data.config import SMOTE_CONFIG


def features_correction(data: pd.DataFrame) -> None:
    """
    Inplace function for correcting DataFrame features names.

    Args:
        data: DataFrame
    """

    new_col_names = {col: col.strip().replace(" ", "_") for col in data.columns}
    data.rename(columns=new_col_names, inplace=True)

def category_column_ascii_correction(data: pd.DataFrame, col_name: str) -> None:
    """
    Inplace function for correcting the categories in a column of a DataFrame, encoding them to ascii.

    Args:
        data: DataFrame
    """

    data[col_name] = data[col_name].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', str(x)))


def concat_and_save_csv(path: str, name: str, encoding: str = "utf-8", save_in_path: bool = False, sep: str = ",", return_df: bool = False) -> pd.DataFrame | None:
    """
    Reads all the CSV files in `path` and saves their concatenation as a CSV file named `name`

    Args:
        path: Path for the directory in wich the CSV files are stored
        name: Name under which to save the CSV file resulted of the concatenation of the other files

        encoding: Encoding for reading the CSV files. Also used for saving. Defaults to `utf-8`
        save_in_path: Wheteher to save the file in the same directory as the rest of CSVs, else it will be saved in the root directory under `name`. Defaults to False
        sep: Separator used for reading and saving.
        return_df: Whether to return the merged dataframe. Defaults to False

    Return:
        Either a DataFrame of the data or None
    """

    files = glob.glob(os.path.join(path, "*.csv"), recursive=False)
    df = pd.concat((pd.read_csv(file, encoding=encoding, sep=sep) for file in files))
    df.to_csv(os.path.join(path, name) if save_in_path else name, sep=sep, index=False)

    if return_df:
        return df
    else:
        del df

def sample(data: pd.DataFrame, p: float = 0.2) -> pd.DataFrame:
    """
    Samples (without replacement) a percentage of the data in a pandas DataFrame

    Args:
        data: DataFrame to sample
        p: Percentage to sample. Must be between 0% and 100%
    """

    assert (0 <= p) and (p <= 1)

    sample_size = int(p * len(data))
    sampled_data = data.sample(n = sample_size, replace = False, random_state = 0).reset_index(drop=True)

    return sampled_data

def balanced_sample(df: pd.DataFrame, category_col: str, n_per_class: int) -> pd.DataFrame:
    """
    Draws a balanced sample from a DataFrame based on a categorical variable.
    
    Args:
        df: DataFrame
        category_col: Column name of the categorical variable
        n_per_class: Number of samples per category
    Returns:
        A balanced DataFrame sample
    """
    return df.groupby(category_col).apply(lambda x: x.sample(n=min(len(x), n_per_class))).reset_index(drop=True)

def jsonize_rows(data: pd.DataFrame) -> list[str]:
    """
    Transforms input datatframe into json format row by row

    Args:
        data: Dataframe containing the data
    """

    jsonized_rows = [row.to_json() for _, row in data.iterrows()]

    return jsonized_rows

def encode_labels(data: pd.DataFrame, label_col: str) -> tuple[np.ndarray]:
    """
    Encodes categorical labels with a label encoder

    Args:
        data: Pandas Dataframe containing the data
        label_col: Name of the column containing the categorical labels

    Returns:
        Data separated into features and encoded labels
    """

    x = data.drop(columns=[label_col]).values
    y = data[label_col].values
    y = LabelEncoder().fit_transform(y)

    return x, y

def smote(x: np.ndarray, y: np.ndarray, threshold: Optional[float], n: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
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


