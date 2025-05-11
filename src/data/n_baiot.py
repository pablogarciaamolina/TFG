import os
import logging
import zipfile
import glob
import re

import pandas as pd

from .config import DATA_DIR, KAGGLE_USERNAME, KAGGLE_KEY, N_BAIOT_URL, N_BAIOT_KAGGLE_ENDPOINT, N_BAIOT_DEFAULT_CONFIG
from ._base import TabularDataset

class Kaggle_Dataset(TabularDataset):

    def __init__(self, data_dir, file_name, url = None):
        
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.kaggle_api = KaggleApi

        super().__init__(data_dir, file_name, url)

class N_BaIoT(Kaggle_Dataset):

    dataset_folder = "N-BaIoT"

    def __init__(self, **kwargs) -> None:

        # Config
        self.config = N_BAIOT_DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        self.raw_data_dir = os.path.abspath(os.path.join(DATA_DIR, self.dataset_folder, "raw"))
        self.processed_data_dir = os.path.abspath(os.path.join(DATA_DIR, self.dataset_folder, "processed"))
        url = N_BAIOT_URL
        
        super().__init__(self.processed_data_dir, self._get_id() + ".csv", url)

    def _get_id(self) -> str:
        
        return "n_baiot"
    
    def _download(self) -> None:
        """
        Dowloads and extracts data files
        """

        if not KAGGLE_USERNAME or not KAGGLE_KEY:
            raise Exception("KAGGLE_USERNAME or KAGGLE_KEY environment variables not set")
        
        if not self.url:
            logging.error("No URL provided for dataset download.")
            return
        
        os.makedirs(self.raw_data_dir, exist_ok=True)

        api = self.kaggle_api()
        api.authenticate()

        logging.info(f"Downloading dataset from {self.url}")
        try:
            api.dataset_download_files(N_BAIOT_KAGGLE_ENDPOINT, path=self.raw_data_dir, unzip=True)
            logging.info("Dowload successful")
        except Exception as e:
            logging.info(f"Failed to download dataset, error: {e}")

        os.makedirs(self.processed_data_dir, exist_ok=True)

    def _collect(self):
        
        if not os.path.exists(self.processed_data_dir):
            self._download()
    
        if not os.path.isfile(self.file_path):
            logging.info(f"{self.file_path} not found. Collecting and cleaning raw data in {self.raw_data_dir}...")
            self.data = self._clean(target_column_name=self.config["target_column_name"])
            os.makedirs(self.file_path, exist_ok=True)
            self.data.to_csv(self.file_path, sep=",", index=False)
        else:
            self.data = pd.read_csv(self.file_path)

        logging.info("Done collecting data")

    def _clean(self, target_column_name: str = "Label") -> pd.DataFrame:
        
        raw_files = [f for f in glob.glob(os.path.join(self.raw_data_dir, "*.csv"), recursive=False) if re.match(r"[1-9]*\.", os.path.basename(f))]
        
        dfs = []
        for path in raw_files:
            target_class = os.path.basename(path)[2:-4].replace(".", "-")
            df = pd.read_csv(path)
            df[target_column_name] = target_class
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        return combined_df
        

    
    def _preprocess(self):
        return super()._preprocess()

        
