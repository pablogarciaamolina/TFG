import os
from typing import Optional
from collections import Counter
import logging
from pprint import pformat
import numpy as np
import matplotlib.pyplot as plt
import torch

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

from ._base import SklearnTrainableModel
from src.models.config import TABNET_PRETRAINER_CONFIG, TABNET_CONFIG, TABNET_PRETRAINING_PARAMS, TABNET_TRAINING_PARAMS, TABNET_SAVING_PATH, TABNET_DATA_AUGMENTATION_CONFIG

class TabNetModel(SklearnTrainableModel):

    model: TabNetClassifier

    def __init__(self, name: str = "tabnet", pretrain: bool = True) -> None:
        """
        Constructor for the class

        Args:
            pretrain: Whether to perform, pretraining or not
            name: Name for the model
        """

        model = TabNetClassifier(**TABNET_CONFIG)
        super().__init__(name, model)
        self.pretrainer = TabNetPretrainer(**TABNET_PRETRAINER_CONFIG) if pretrain else None

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: Optional[np.ndarray], y_val: Optional[np.ndarray], augmentation: bool = True) -> None:
        """
        Performs pretreining (if specified) and then trains the model based on the configuration paramenters

        Args:
            x_train: Training data features
            y_train: Training datat labels
            x_val: Optional validation data features
            y_val: Optional validation data labels
            augmentation: Whether to perform data augmentation or not
        """

        # Data Augmentation
        if augmentation:
            logging.info("Augmenting data for TabNet...")
            x, y = data_augmentation(
                x,
                y,
                TABNET_DATA_AUGMENTATION_CONFIG["class_samples_threshold"],
                TABNET_DATA_AUGMENTATION_CONFIG["n"]
            )

        # Pretraining
        if self.pretrainer:
            logging.info("Pretraining for TabNet...")
            eval_set = [x_val] if x_val is not None else None
            self.pretrainer.fit(
                x,
                eval_set=eval_set,
                **TABNET_PRETRAINING_PARAMS
            )

        # Training
        logging.info("Training TabNet model...")
        if (x_val is not None) and (y_val is not None):
            eval_set = [(x_val, y_val),]
            eval_name = ['val']
        else:
            eval_set = None
            eval_name = None
        self.model.fit(
            x, y,
            eval_set=eval_set,
            eval_name=eval_name,
            **TABNET_TRAINING_PARAMS,
            from_unsupervised=self.pretrainer,
        )
    
    def predict(self, x) -> np.ndarray:

        return self.model.predict(x)

    def plot_metrics(self) -> None:
        """
        Plots the metrics for the last training done to the model.
        NOTE: Not having trained the model will raise an error. Loading the model from memory does not include its past training history.
        """
        
        plt.plot(self.model.history['lr'])

        clf = self.model
        train_loss = clf.history["loss"]
        if "val_accuracy" in clf.history.history.keys():
            val_metric = clf.history["val_accuracy"]  # Replace with "val_accuracy" if using accuracy
        else:
            val_metric = None

        # Plot Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Tabnet Training Loss Over Epochs")
        plt.legend()

        # Plot Evaluation Metric
        plt.subplot(1, 2, 2)
        if val_metric is not None:
            plt.plot(val_metric, label="Validation Accuracy", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Tabnet Validation Acuracy Over Epochs")
        plt.legend()

        plt.show()

    def save(self) -> None:

        path = os.path.join(TABNET_SAVING_PATH, self.name)
        self.model.save_model(path)

    def load(self) -> None:

        path = os.path.join(TABNET_SAVING_PATH, self.name + ".zip")
        self.model.load_model(path)

    def explain(self, x) -> None:
        """
        Local explanation method for visualizing generated masks over a features set

        Args:
            x: Data features to use for explanation
        """

        explanations, masks = self.model.explain(x)

        fig, axs = plt.subplots(1, 3, figsize=(20,20))
        for i in range(3):
            axs[i].imshow(masks[i][:50])
            axs[i].set_title(f"mask {i}")


def data_augmentation(x: np.ndarray, y: np.ndarray, threshold: Optional[float], n: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Specifically for TabNet. Performs data augmentation for clasification using a modified version of SMOTE.

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

    if not threshold:
        threshold = TABNET_DATA_AUGMENTATION_CONFIG["class_samples_threshold"]
    
    assert isinstance(threshold, float)
    assert 0 < threshold <= 1

    if not n:
        n = TABNET_DATA_AUGMENTATION_CONFIG["n"]

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
            TABNET_DATA_AUGMENTATION_CONFIG["alpha"],
            TABNET_DATA_AUGMENTATION_CONFIG["beta"],
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

