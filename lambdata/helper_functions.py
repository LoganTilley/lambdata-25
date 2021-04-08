import pandas as pd
import numpy as np
import random as ran


class DataWrangling:
    """
    Provides useful methods for cleaning and wrangling data. Inludes common
    chain commands encapsulated into a single 'convenience' function, as well as
    more intricate functions.
    """

    @staticmethod
    def null_count(df: pd.DataFrame):
        """Returns total null count of a dataframe."""
        return df.isnull().sum().sum()

class Manipulation:
    """
    Provides methods for manipulating dataframes post-wrangling.
    Instantiating this class allows one to set the default seed for randomizing
    methods. The default seed is 42.
    """

    seed = 42

    def __init__(self, seed: int=42):
        self.seed = seed
    
    @staticmethod
    def train_test_split(df: pd.DataFrame, train_size: float=0.8):
        """
        Returns a tuple of train and test dataframes. Currently this function
        does not account for target variables.
        """
        cutoff = len(df) * train_size # Index at train_size
        train = df[df.index < cutoff]
        test = df[df.index > cutoff]
        return (train, test)
    
    @classmethod
    def randomize(cls, df: pd.DataFrame, seed: int=seed):
        """
        Shuffles ALL VALUES within a dataframe across ALL axes, inplace.
        """
        # Is there a better approach to this? Need to research.

        r = ran.Random()
        r.seed(seed)
        values = df.values.flatten().tolist() # Necessary for r.shuffle
        r.shuffle(values)

        # Need to access the assignable values, then de-flatten the new values
        # array.
        df.values.real = np.asarray(values).reshape(df.values.real.shape)
        return df