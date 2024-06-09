"""
Synthetic Data Set Generator for DS3

This script generates and saves synthetic data sets (DS3) into CSV files. The generated data sets
contain samples with specified parameters such as the number of samples, number of features, and noise level.
The generated CSV files are saved in the same directory as this script. These data sets are publicly available
at https://zenodo.org/record/8180739.

Usage:
    Run this script to generate synthetic data sets (DS3) with varying random seeds and save them as CSV files.

Requirements:
    - Datasets module: Provides functions to create synthetic data sets.
    - pandas library: Used to handle and save data in CSV format.
    - Utils module: Offers utility functions to manage file paths and directory creation.
    - Constants module: Contains predefined constant values.

Generated Data Set Details:
    - Number of samples: 10000
    - Number of features: 200
    - Noise level: 25
    - Random seeds: SEEDS from the Constants module

Data Set Saving:
    - The generated data sets are saved as CSV files in the directory specified by path_to_save_data_set.
    - Each CSV file is named in the format "003_DS3_{seed}.csv", where {seed} corresponds to the random seed.

Note:
    The generated data sets are synthetic and can be used for various testing and analysis purposes.

"""

from Datasets import SyntheticDS
import pandas as pd
from Utils import Util, Constants

path_to_save_data_set = Util.get_path_to_save_generated_dataset_file('03_DS3')
Util.create_directory(path_to_save_data_set)

n_samples = 10000
n_features = 200
noise = 25

SEEDS = Constants.SEEDS
for seed in SEEDS:
    X, y = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=True,
                                      random_state=seed)

    # Create a DataFrame using X and y
    df = pd.DataFrame(data=X, columns=[f"feature_{i}" for i in range(n_features)])
    df['target'] = y

    # Save DataFrame to a CSV file
    df.to_csv(path_to_save_data_set + '\\003_DS3_' + str(seed) + '.csv', index=False)




