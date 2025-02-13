import os
import glob
import pandas as pd

def features_correction(data: pd.DataFrame) -> None:
    """
    Inplace function for correcting DataFrame features names.

    Args:
        data: DataFrame
    """

    new_col_names = {col: col.strip().replace(" ", "_") for col in data.columns}
    data.rename(columns=new_col_names, inplace=True)


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
    sampled_data = data.sample(n = sample_size, replace = False, random_state = 0)

    return sampled_data