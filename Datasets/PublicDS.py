"""
PublicDS: CSV Dataset Reader and Feature Engineering

This Python script reads CSV dataset files and performs basic feature engineering operations, such as normalization
and one-hot encoding. The resulting dataset is returned as a feature matrix (X) in matrix format and a target vector (y)
as an array.

Usage:
    Call the respective functions to load and preprocess specific CSV datasets. The functions return feature matrix (X)
    and target vector (y) ready for further analysis and modeling.

Requirements:
    - pandas library: Used to read and manipulate CSV data.
    - sklearn.preprocessing.MinMaxScaler: Used for Min-Max scaling.
    - sklearn.preprocessing.StandardScaler: Used for standard scaling.

Supported Datasets and Their Purposes:
    1. Kind County House Sales Dataset: Predicts the sales price of houses in King County, Seattle.
    2. Medical Cost Personal Datasets: Predicts individual medical costs billed by health insurance.
    3. 1000 Companies Profit: Predicts the profit of companies based on their operating expenses and other factors.
    4. Combined Cycle Powerplant: Predicts the net hourly electrical energy output of a power plant.

Functions:
    - get_king_county_house_sales_data(path): Loads and preprocesses the King County House Sales dataset.
    - get_medical_cost_personal_dataset(path): Loads and preprocesses the Medical Cost Personal dataset.
    - get_profit_estimation_for_companies_dataset(path): Loads and preprocesses the 1000 Companies Profit dataset.
    - get_CCPP(path): Loads and preprocesses the Combined Cycle Powerplant dataset.

Data Preprocessing Steps:
    - Load the CSV dataset using pandas.
    - Drop unnecessary columns and handle missing values.
    - Perform one-hot encoding on categorical variables where applicable.
    - Normalize numerical features using Min-Max scaling or standard scaling.

Note:
    The purpose of this script is to facilitate the preprocessing of specific CSV datasets for further analysis and
    modeling.
    The specific data preprocessing steps vary based on the dataset characteristics.

"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def get_king_county_house_sales_data(path):
    """
    Dataset: Kind County House Sales Dataset
    source: https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa
    Purpose: Predicts the sales price of houses in King County, Seattle.

    Load and preprocess the Kind County House Sales Dataset.

    This function reads the CSV dataset file, performs necessary preprocessing steps, and returns the feature matrix (X)
    and target vector (y) for predicting the sales price of houses in King County, Seattle.

    Parameters:
        path (str): Path to the CSV dataset file.

    Returns:
        X_scaled (numpy.ndarray): Scaled feature matrix (X) after preprocessing.
        y_scaled (numpy.ndarray): Scaled target vector (y) after preprocessing.
    """

    # Load the dataset
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis=1)

    # Just keep those features
    columns_to_keep = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15',
                       'bedrooms', 'bathrooms', 'view', 'price']
    df = df[columns_to_keep]
    # Handling missing values (if any)
    df = df.dropna()
    # Create feature matrix X
    X = df.drop('price', axis=1)
    # Create target vector y
    y = df['price']

    # Scaling and normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled


def get_medical_cost_personal_dataset(path):
    """
        Dataset: Medical Cost Personal Datasets
        source: https://www.kaggle.com/datasets/mirichoi0218/insurance
        Purpose: Predicts charges "Individual medical costs billed by health insurance"

        Load and preprocess the Medical Cost Personal Datasets.

        This function reads the CSV dataset file, performs one-hot encoding on categorical variables, and returns
        the feature matrix (X) and target vector (y) for predicting individual medical costs billed by health insurance.

        Parameters:
            path (str): Path to the CSV dataset file.

        Returns:
            X_scaled (numpy.ndarray): Scaled feature matrix (X) after preprocessing.
            y_scaled (numpy.ndarray): Scaled target vector (y) after preprocessing.
        """

    data = pd.read_csv(path)
    data = data.dropna()

    # Perform one-hot encoding using pd.get_dummies()
    data_encoded = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    X = data_encoded.drop('charges', axis=1)
    y = data_encoded['charges'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled.flatten()


def get_profit_estimation_for_companies_dataset(path):
    """
        Dataset: 1000 Companies Profit
        source: https://www.kaggle.com/datasets/rupakroy/1000-companies-profit
        Purpose: Predicts the profit of these companies based on their operating expenses
        and other factors.

        Load and preprocess the 1000 Companies Profit dataset.

        This function reads the CSV dataset file, performs one-hot encoding on the categorical variable 'State',
        scales numerical features using Min-Max scaling, and returns the feature matrix (X) and target vector (y)
        for predicting the profit of companies based on their operating expenses and other factors.

        Parameters:
            path (str): Path to the CSV dataset file.

        Returns:
            X (numpy.ndarray): Feature matrix (X) after preprocessing.
            y (numpy.ndarray): Target vector (y) after preprocessing.
        """

    # Load the dataset from the given path
    df = pd.read_csv(path)
    # Encode categorical variable 'State' using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=["State"])
    # Scale numerical features using Min-Max scaling to normalize the data
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    # Separate features (X) and target variable (y)
    X = df_normalized.drop(["Profit"], axis=1).values
    y = df_normalized["Profit"].values

    return X, y


def get_CCPP(path):
    """
        Dataset: combined cycle powerplant
        source: https://www.kaggle.com/datasets/gova26/airpressure
        Purpose: Predicts the net hourly electrical energy output (EP) of the plant.

        Load and preprocess the Combined Cycle Power Plant dataset.

        This function reads the CSV dataset file, normalizes the features using Min-Max scaling,
        and returns the feature matrix (X) and target vector (y) for predicting the net hourly
        electrical energy output (EP) of the power plant.

        Parameters:
            path (str): Path to the CSV dataset file.

        Returns:
            X (numpy.ndarray): Feature matrix (X) after preprocessing.
            y (numpy.ndarray): Target vector (y) after preprocessing.
        """
    # Load the dataset from the given path
    df = pd.read_csv(path)
    # Normalize the features using Min-Max scaling
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    # Separate features (X) and target variable (y)
    X = df_normalized.drop(["PE"], axis=1).values
    y = df_normalized["PE"].values

    return X, y




