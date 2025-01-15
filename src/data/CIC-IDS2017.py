import os
import numpy as np
import pandas as pd

from utils import concat_and_save_csv, features_correction

DATA_PATH = r"data\CIC-IDS2017"
DATASET_FOLDER = r"MachineLearningCVE"
EXTENDED_DATASET_FOLDER = r"TrafficLabelling_"
DATASET_SAVE_NAME = "dataset.csv"

def preprocess_dataframe(data: pd.DataFrame) -> None:
    """
    Inplace preprocessing for dataset's DataFrame. Corrects features names, manages duplicates and deals with missing and infinite values.
    """

    # Correct feature names
    features_correction(data)

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    # Infinite values
    ## Replacing with NaN is a good answer ?? (Shaould check the feature, infinity might have a meaning)
    # maybe dropping might be better ??
    data.replace([np.inf, -np.inf], np.nan, inplace = True)

    # Missing values
    ## Dropping might be better ?? the rows with infinites have them always both in Flow Bytes and Fow Packages
    ## also, only around 6% missing values (including infinites)
    ## Filling with median might affect performance or learning ??
    med_flow_bytes = data['Flow_Bytes/s'].median()
    med_flow_packets = data['Flow_Packets/s'].median()
    data['Flow_Bytes/s'].fillna(med_flow_bytes, inplace = True)
    data['Flow_Packets/s'].fillna(med_flow_packets, inplace = True)


def get_dataframe(extended: bool = False) -> pd.DataFrame:
    """
    Obtains or extracts the DataFrame of the dataset CIC-IDS2017 (either extended or normal version) from local save.

    Args:
        extended: Whether to load the extended (True) or normal (False) version of the dataset. Defaults to False (normal version loading)

    Return:
        A DataFrame of the data
    """

    dataset_folder = os.path.join(DATA_PATH, EXTENDED_DATASET_FOLDER if extended else DATASET_FOLDER)
    if not os.path.isfile(os.path.join(dataset_folder, DATASET_SAVE_NAME)):
        df = concat_and_save_csv(
            dataset_folder,
            DATASET_SAVE_NAME,
            encoding="latin-1",
            save_in_path=True,
            sep=",",
            return_df=True
        )
        preprocess_dataframe(df)
        df.to_csv(os.path.join(dataset_folder, DATASET_SAVE_NAME), sep=",", index=False)
    else:
        df = pd.read_csv(os.path.join(dataset_folder, DATASET_SAVE_NAME), encoding="latin-1", sep=",")

    return df


if __name__ == "__main__":

    df = get_dataframe(extended=True)
    df.info()