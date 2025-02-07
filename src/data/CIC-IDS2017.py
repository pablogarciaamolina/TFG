import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


from utils import concat_and_save_csv, features_correction

class CICIDS2017:

    DATA_PATH = r"data\CIC-IDS2017"
    DATASET_FOLDER = r"MachineLearningCVE"
    EXTENDED_DATASET_FOLDER = r"TrafficLabelling_"
    DATASET_SAVE_NAME = "dataset.csv"
    ANALYSIS_PATH = r"results\data_analysis\CIC-IDS2017"
    
    def __init__(self, extended: bool = False):
        """
        Constructor for CIC-IDS2017 class

        Args:
            extended: Whether to load the extended (True) or normal (False) version of the dataset. Defaults to False (normal version loading)
        """
        
        self.df = self.load_dataframe(extended=extended)
        self.extended = extended

    def sample(self, p: float = 0.2) -> pd.DataFrame:
        """
        Samples (without replacement) a percentage of the data

        Args:
            p: Percentage to sample. Must be between 0% and 100%
        """

        assert (0 <= p) and (p <= 1)

        sample_size = int(p * len(self.df))
        sampled_data = self.df.sample(n = sample_size, replace = False, random_state = 0)
        
        return sampled_data

    def load_dataframe(self, extended: bool = False) -> pd.DataFrame:
        """
        Obtains or extracts the DataFrame of the dataset CIC-IDS2017 (either extended or normal version) from local save.

        Args:
            extended: Whether to load the extended (True) or normal (False) version of the dataset. Defaults to False (normal version loading)

        Return:
            A DataFrame of the data
        """

        dataset_folder = os.path.join(self.DATA_PATH, self.EXTENDED_DATASET_FOLDER if extended else self.DATASET_FOLDER)
        if not os.path.isfile(os.path.join(dataset_folder, self.DATASET_SAVE_NAME)):
            df = concat_and_save_csv(
                dataset_folder,
                self.DATASET_SAVE_NAME,
                encoding="latin-1",
                save_in_path=True,
                sep=",",
                return_df=True
            )
            self.cleaning(df)
            df.to_csv(os.path.join(dataset_folder, self.DATASET_SAVE_NAME), sep=",", index=False)
        else:
            df = pd.read_csv(os.path.join(dataset_folder, self.DATASET_SAVE_NAME), encoding="latin-1", sep=",")

        return df
    
    def cleaning(self, df: pd.DataFrame) -> None:
        """
        Cleaning for dataset's DataFrame. Corrects features names, manages duplicates and deals with missing and infinite values.
        """

        # Correct feature names
        features_correction(df)

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Infinite values
        ## Replacing with NaN is a good answer ?? (Shaould check the feature, infinity might have a meaning)
        # maybe dropping might be better ??
        df.replace([np.inf, -np.inf], np.nan, inplace = True)

        # Missing values
        ## Dropping might be better ?? the rows with infinites have them always both in Flow Bytes and Fow Packages
        ## also, only around 6% missing values (including infinites)
        ## Filling with median might affect performance or learning ??
        med_flow_bytes = df['Flow_Bytes/s'].median()
        med_flow_packets = df['Flow_Packets/s'].median()
        df['Flow_Bytes/s'].fillna(med_flow_bytes, inplace = True)
        df['Flow_Packets/s'].fillna(med_flow_packets, inplace = True)

    def analysis(self, verbose: int = 1) -> None:
        """
        Method for executing an analysis over the CIC-IDS20217 dataset and saving results on memory

        Args:
            verbose: An indicator of how much information to display while executing
        """

        save_directory = os.path.join(self.ANALYSIS_PATH, self.EXTENDED_DATASET_FOLDER if self.extended else self.DATASET_FOLDER)
        os.makedirs(save_directory, exist_ok=True)

        if verbose > 0: print(f"\n---Starting CIC-IDS2017 {'(extended)' if self.extended else ''} analysis---")

        # CORRELATION MATRIX
        if verbose > 1: print("Correlation matrix...", end="")

        corr = self.df.corr(numeric_only = True).round(2)
        corr.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)
        fig, ax = plt.subplots(figsize = (24, 24))
        sns.heatmap(corr, cmap = 'coolwarm', annot = False, linewidth = 0.5)
        plt.title('Correlation Matrix', fontsize = 18)

        if verbose > 2: plt.show()

        plt.savefig(os.path.join(save_directory, "correlation_matrix.png"))

        if verbose > 1: print("DONE")

        # DISTRIBUTION OF ATTACKS
        if verbose > 1: print("Distribution of attacks...", end="")

        plt.figure(figsize = (14, 8))
        ax = sns.countplot(x='Label', hue='Label', data=self.df, palette='pastel', order=self.df['Label'].value_counts().index, legend=False)
        plt.title('Types of attacks')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation = 90)
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height() + 1000), ha = 'center')

        if verbose > 2: plt.show()

        plt.savefig(os.path.join(save_directory, "attacks_distribution.png"))

        if verbose > 1: print("DONE")


        # BOXPLOT OF FEATURES FOR EACH ATTACK / FIRMS FOR EACH ATTACK
        if verbose > 1: print("Distribution of attacks...", end="")

        os.makedirs(os.path.join(save_directory, "attacks_boxplots"), exist_ok=True)
        for attack_type in self.df['Label'].unique():

            attack_data = self.df[self.df['Label'] == attack_type]
            plt.figure(figsize=(20, 20))
            sns.boxplot(data = attack_data.drop(columns = ['Label']), orient = 'h')
            plt.title(f'Boxplot of Features for Attack Type: {attack_type}')
            plt.xlabel('Feature Value')

            if verbose > 2:
                plt.show()
            
            plt.savefig(os.path.join(save_directory, "attacks_boxplots", f"{str(attack_type).strip().replace(" ", "_")}.png"))
        
        if verbose > 1: print("DONE")

        if verbose > 0: print(f"---DONE---")

    def preprocessing(self, verbose: int = 1) -> None:
        """
        Performs data preprocessing for the dataset
        
        Args:
            verbose: An indicator of how much information to display while executing
        """

        if verbose > 0: print(f"\n---Starting CIC-IDS2017 {'(extended)' if self.extended else ''} preprocessing---")

        # QUANTIZATION
        if verbose > 1: print("Quantization...", end="")

        for col in self.df.columns:
            col_type = self.df[col].dtype
            if col_type != object:
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                # Downcasting float64 to float32
                if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = self.df[col].astype(np.float32)

                # Downcasting int64 to int32
                elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = self.df[col].astype(np.int32)

        if verbose > 1: print("DONE")

        # DROPPING COLUMNS WITH ONLY ONE UNIQUE VALUE
        if verbose > 1: print("Droping columns with one value...", end="")

        num_unique = self.df.nunique()
        one_variable = num_unique[num_unique == 1]
        not_one_variable = num_unique[num_unique > 1].index

        dropped_cols = one_variable.index
        self.df = self.df[not_one_variable]

        if verbose > 1: print("DONE")

        # PCA
        if verbose > 1: print("PCA...", end="")

        features = self.df.drop('Label', axis = 1)
        attacks = self.df['Label']

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        size = len(features.columns) // 2
        ipca = IncrementalPCA(n_components = size, batch_size = 500)
        for batch in np.array_split(scaled_features, len(features) // 500):
            ipca.partial_fit(batch)

        # print(f'information retained: {sum(ipca.explained_variance_ratio_):.2%}')
        transformed_features = ipca.transform(scaled_features)
        self.df = pd.DataFrame(transformed_features, columns = [f'PC{i+1}' for i in range(size)])
        self.df['Label'] = attacks.values

        if verbose > 1: print("DONE")


        if verbose > 0: print(f"---DONE---")



if __name__ == "__main__":

    data = CICIDS2017(extended=True)

    # General information
    data.df.info()
    data.df.describe().transpose()

    # Analysis
    data.analysis(verbose=2)