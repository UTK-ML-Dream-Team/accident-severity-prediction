#!/usr/bin/env python
# coding: utf-8
from typing import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Custom Libs
from project_libs import ColorizedLogger
from sklearn.impute import KNNImputer

logger = ColorizedLogger('Preprocessing', 'green')


def one_hot_encode(data):
    y = data.copy().T.astype(int)
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


def one_hot_unencode(onehot_data):
    return np.argmax(onehot_data, axis=1)[:, np.newaxis]


def isolate_city_state(data, cities, states):
    """ This ensures that each city is selected with it's respective state
    which is why I didn't simply run a merge statement. """

    for ind, x in enumerate(zip(cities, states)):
        tmp = data.loc[(data['City'] == x[0]) & (data['State'] == x[1])].copy()
        if ind == 0:
            new_data = tmp
        else:
            new_data = new_data.append(tmp)
    return new_data


def subset_df(df, keep_list):
    mask = df.columns.isin(keep_list)
    selectedCols = df.columns[mask]
    return df[selectedCols]


def OLS(x, y):
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()
    logger.info(results.summary())


def basic_impute(input_data):
    data = input_data.copy()
    df_num = data.select_dtypes(include=np.number)

    for i in data:
        if i in df_num:
            data.loc[data.loc[:, i].isnull(), i] = df_num.loc[:, i].median()
        else:
            data.loc[data.loc[:, i].isnull(), i] = data.loc[:, i].mode()

    return data


def knn_imputer(df, k):
    temp_data = df.select_dtypes(include=np.number)
    imputer = KNNImputer(n_neighbors=k)
    vals_arr = imputer.fit_transform(temp_data)
    cols = df.select_dtypes(include=np.number).columns
    df[cols] = vals_arr

    return df


def encode_std_extract_split(X):
    # Dropping the following features
    # 1. Civil_Twilight        - Use Sunrise_Sunset instead
    # 2. Nautical_Twilight     - Use Sunrise_Sunset instead
    # 3. Astronomical_Twilight - Use Sunrise_Sunset instead
    # 4. Roundabout            - Carries only one category
    # 5. ID                    - Unique for each row
    # 6. Start_Time            - Extracted month, week, and hour
    # 7. End_Time              - Extracted duration using start time and endtime
    # 8. Start_Lat             - Convert to X, Y, Z coordinate
    # 9. Start_Lng             - Convert to X, Y, Z coordinate
    # 10. End_Lat              - We have distance and starting lat long
    # 11. End_Lng              - We have distance and starting lat long
    # 12. Description          - We can defer doing the text analysis at this moment
    # 13. Number               - Number is missing or 75% of the records
    # 14. Street               - Dropping Street at the moment
    # 15. State                - State is unique for each city
    # 16. Country              - Doing the analysis for US
    # 17. Zipcode              - Already have other location information
    # 18. Airport_Code         - Carry weather station code
    # 19. Weather_Timestamp    - Not relevant for analysis

    # Target variable
    X.loc[(X['Severity'] == 1) | (X['Severity'] == 2), 'Severity'] = 0
    X.loc[(X['Severity'] == 3) | (X['Severity'] == 4), 'Severity'] = 1
    y = X['Severity']

    # Label Encoding
    binary_features = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
        'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Side', 'Turning_Loop',
        'Sunrise_Sunset'
    ]

    for features in binary_features:
        le = LabelEncoder()
        X[features] = le.fit_transform(X[features])

    # Create duration feature
    X['Start_Time'] = pd.to_datetime(X['Start_Time'])
    X['End_Time'] = pd.to_datetime(X['End_Time'])
    X['Duration'] = (X['End_Time'] - X['Start_Time']).dt.total_seconds() / 60.0  # Minutes

    # Create month, weekday, and hour feature
    X["Month"] = X["Start_Time"].dt.month
    X["DayOfWeek"] = X["Start_Time"].dt.dayofweek
    X["Hour"] = X["Start_Time"].dt.hour

    # Create Cartesian Coordinates
    X["Coord_X"] = np.cos(X["Start_Lat"] * np.cos(X["Start_Lng"]))
    X["Coord_Y"] = np.cos(X["Start_Lat"] * np.sin(X["Start_Lng"]))
    X["Coord_Z"] = np.sin(X["Start_Lat"])

    # One hot encoding
    # one_hot_features remaining = ['Wind_Direction', 'Weather_Condition' ]
    # We need to get cleaner categories for the above two features

    city_df = pd.get_dummies(X['City'], prefix='City')
    county_df = pd.get_dummies(X['County'], prefix='County')
    timezone_df = pd.get_dummies(X['Timezone'], prefix='Timezone')
    month_df = pd.get_dummies(X['Month'], prefix='Month')
    weekday_df = pd.get_dummies(X['DayOfWeek'], prefix='DayOfWeek')
    timezone_df = pd.get_dummies(X['Hour'], prefix='Hour')
    wind_direction_df = pd.get_dummies(X['Wind_Direction'], prefix='Wind_Direction')
    weather_condition_df = pd.get_dummies(X['Weather_Condition'], prefix='Weather_Condition')
    X = pd.concat([X, city_df, county_df, timezone_df, wind_direction_df, weather_condition_df],
                  axis=1)

    # Drop non essential features
    non_essential_features = [
        'ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat',
        'End_Lng', 'Description', 'Number', 'Street', 'State', 'Zipcode', 'Country',
        'Airport_Code', 'Weather_Timestamp', 'Roundabout', 'Civil_Twilight', 'Nautical_Twilight',
        'Astronomical_Twilight'
    ]
    X.drop(non_essential_features, axis=1, inplace=True)

    # Drop one hot encoded features
    one_hot = ['City', 'County', 'Timezone', 'Month', 'DayOfWeek', 'Hour', 'Wind_Direction',
               'Weather_Condition']
    X.drop(one_hot, axis=1, inplace=True)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.10, shuffle=True, stratify=y,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.10 / 0.90, shuffle=True,
                                                      stratify=y_train,
                                                      random_state=42)

    # Standardize Numerical Variables
    numerical_features = [
        'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)',
        'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
        'Precipitation(in)', 'Duration', 'Coord_X', 'Coord_Y', 'Coord_Z'
    ]

    for feature in numerical_features:
        mu = X_train[feature].mean()
        sigma = X_train[feature].std()
        X_train[feature] = X_train[feature].apply(lambda x: (x - mu) / sigma)
        X_val[feature] = X_val[feature].apply(lambda x: (x - mu) / sigma)
        X_test[feature] = X_test[feature].apply(lambda x: (x - mu) / sigma)

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def filter_loc_basic_var(accident_df):
    """
    Function to preprocess the location and basic variables. Note that in our dataset
    we didn't find missing values for most of the columns except for Number which was
    missing for almost 70% of the rows.
    """

    basic_location = [
        'ID', 'Severity', 'Start_Time', 'End_Time', 'Distance(mi)', 'Description',
        'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Number', 'Street', 'Side',
        'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone', 'Airport_Code'
    ]
    return accident_df[basic_location]


class PCA:
    """ Principal Component Analysis. """
    means: np.ndarray
    basis_vector: np.ndarray

    def __init__(self) -> None:
        pass

    def fit(self, data: np.ndarray, max_dims: int = None,
            max_error: float = None, split: bool = True):
        # Split features and class labels
        if split:
            data_x, data_y = self.x_y_split(dataset=data)
        else:
            data_x = data

        if max_dims:
            if max_dims > data_x.shape[1] - 1:
                raise Exception("Max dims should be no more than # of features -1!")
        elif not max_error:
            logger.warning("Neither of max_dims, max_error was given. Using max_dims=1")
            max_dims = 1

        # Calculate overall means
        self.means = np.expand_dims(np.mean(data_x, axis=0), axis=1).T
        # Calculate overall covariances
        covariances = np.cov(data_x.T)
        # Calculate lambdas and eigenvectors
        lambdas, eig_vectors = np.linalg.eigh(covariances)
        # Calculate the basis vector based on the largest lambdas
        lambdas_sorted_idx = np.argsort(lambdas)[::-1]
        lambdas_sorted = lambdas[lambdas_sorted_idx]
        # If max_error is set, derive max_dims based on that
        if max_error:
            lambdas_sum = np.sum(lambdas_sorted)
            for n_dropped_dims in range(0, data_x.shape[1]):
                n_first_lambdas = lambdas_sorted[:n_dropped_dims + 1]
                pca_error = 1 - (np.sum(n_first_lambdas) / lambdas_sum)
                if pca_error <= max_error:
                    max_dims = n_dropped_dims + 1
                    logger.info(f"For # dims={max_dims} error={pca_error} <= {max_error}")
                    break
        self.basis_vector = eig_vectors[:, lambdas_sorted_idx[:max_dims]]

    def transform(self, data: np.ndarray, split: bool = True) -> np.array:
        if split:
            data_x, data_y = self.x_y_split(dataset=data)
        else:
            data_x = data
        data_x_proj = data_x @ self.basis_vector
        if split:
            return np.append(data_x_proj, data_y[:, np.newaxis], axis=1)
        else:
            return data_x_proj

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)


import numpy as np


# Implement k-fold cross-validation for a 2-class dataset

def crossval_split(data, kfold):
    """
    # How to use the cross-validation function

    #ypredict = {}
    #for k in range(kfold):
    #    ypredict[k] = MODEL_FUNCTION_HERE(Xtrain[k], ytrain[k], Xvalid[k])

    #avg_overall, avg_class0, avg_class1 = acc_crossval(yvalid, ypredict, kfold=10)
    """
    ### --- CLASS 0 --- ###

    # Get samples for class 0
    data0 = data[data[:, -1] == 0]
    n0 = data0.shape[0]  # number of samples in class 0

    valid0 = {}  # Values for each of k validation chunks in class 0
    valid0_ind = {}  # Indices for each of k validation chunks in class 0

    train0 = {}
    train0_ind = {}

    # First of k chunks of data
    valid0_ind[0] = np.random.choice(n0, size=round(n0 / kfold), replace=False)
    valid0[0] = data0[valid0_ind[0], :]  # Randomly select (1/k)*n rows from data in class 0
    ind0 = np.arange(0, n0, 1)  # Indices for all rows in class 0 data

    train0_ind[0] = np.delete(ind0,
                              valid0_ind[0])  # Indices of n - (n/k) rows for training set for class 0
    train0[0] = data0[train0_ind[0],
                :]  # Training set is all samples not in validation set for class 0

    for k in range(1, kfold):

        # If class 0 samples cannot be evenly divided by k and the final partition has fewer than n0/k samples
        if len(train0_ind[k - 1]) < round(n0 / kfold):
            valid0_ind[k] = train0_ind[
                k - 1]  # Validation set is all remaining samples that haven't been selected

        else:  # Select indices randomly from remaining indices in training section
            valid0_ind[k] = np.random.choice(train0_ind[k - 1], size=round(n0 / kfold), replace=False)

        # Select rows with those indices from original dataset
        valid0[k] = data0[valid0_ind[k], :]
        # Delete those indices from remaining indices that have not been selected for validation set
        train0_ind[k] = np.delete(train0_ind[k - 1], np.argwhere(valid0_ind[k]))
        # Training data is all rows not selected for validation set
        train0[k] = data0[np.delete(ind0, valid0_ind[k]), :]

    ### --- CLASS 1 --- ###

    # Get samples for class 1
    data1 = data[data[:, -1] == 1]
    n1 = data1.shape[0]  # number of samples in class 1

    valid1 = {}  # Values for each of k validation chunks in class 1
    valid1_ind = {}  # Indices for each of k validation chunks in class 1

    train1 = {}
    train1_ind = {}

    # First of k chunks of data
    valid1_ind[0] = np.random.choice(n1, size=round(n1 / kfold), replace=False)
    valid1[0] = data1[valid1_ind[0], :]  # Randomly select (1/k)*n rows from data in class 1
    ind1 = np.arange(0, n1, 1)  # Indices for all rows in class 0 data

    train1_ind[0] = np.delete(ind1, valid1_ind[
        0])  # Indices of n - (n/k) rows for validation set for class 0
    train1[0] = data1[train1_ind[0],
                :]  # Validation set is all samples not in training set for class 0

    for k in range(1, kfold):

        # If class 1 samples cannot be evenly divided by k and the final partition has fewer than n0/k samples
        if len(train1_ind[k - 1]) < round(n1 / kfold):
            valid1_ind[k] = train1_ind[
                k - 1]  # Validation set is all remaining samples that haven't been selected

        else:  # Select indices randomly from remaining data
            valid1_ind[k] = np.random.choice(train1_ind[k - 1], size=round(n1 / kfold), replace=False)

        # Select rows with those indices from remaining data
        valid1[k] = data1[valid1_ind[k], :]
        # Delete those indices from remaining data
        train1_ind[k] = np.delete(train1_ind[k - 1], np.argwhere(valid1_ind[k]))
        # Training data is all rows not selected for validation set
        train1[k] = data1[np.delete(ind1, valid1_ind[k]), :]

    # Combine training & validation sets for classes 0 and 1

    training_data = {}
    validation_data = {}
    Xtrain = {}
    ytrain = {}
    Xvalid = {}
    yvalid = {}

    for k in range(kfold):
        training_data[k] = np.concatenate((train0[k], train1[k]), axis=0)
        Xtrain[k] = training_data[k][:, :-1]
        ytrain[k] = training_data[k][:, -1].astype(int)

        validation_data[k] = np.concatenate((valid0[k], valid1[k]), axis=0)
        Xvalid[k] = validation_data[k][:, :-1]
        yvalid[k] = validation_data[k][:, -1].astype(int)

    return Xtrain, ytrain, Xvalid, yvalid, training_data, validation_data


def acc_crossval(yvalid, ypredict, kfold):
    acc_overall = []
    acc_class0 = []
    acc_class1 = []

    for k in range(kfold):
        assert len(yvalid[k]) == len(ypredict[k])

        correct_all = yvalid[k] == ypredict[k]  # all correctly classified samples

        acc_overall.append(np.sum(correct_all) / len(yvalid[k]))

        acc_class0.append(np.sum(correct_all[yvalid[k] == 0]) / len(yvalid[k][yvalid[k] == 0]))
        acc_class1.append(np.sum(correct_all[yvalid[k] == 1]) / len(yvalid[k][yvalid[k] == 1]))

    # Average accuracies
    avg_overall = np.mean(acc_overall)
    avg_class0 = np.mean(acc_class0)
    avg_class1 = np.mean(acc_class1)

    return avg_overall, avg_class0, avg_class1
