#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import NaN


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
    print(results.summary())


def basic_impute(data):
    df_num = data.select_dtypes(include=np.number)

    for i in data:
        if i in df_num:
            data.loc[data.loc[:, i].isnull(), i] = df_num.loc[:, i].median()
        else:
            data.loc[data.loc[:, i].isnull(), i] = data.loc[:, i].mode()

    return data


def preprocess_loc_basic_var(accident_df):
    """
    Function to preprocess the location and basic variables. Note that in our dataset
    we didn't find missing values for most of the columns except for Number which was
    missing for almost 70% of the rows.
    """

    basic_location = ['ID', 'Severity', 'Start_Time', 'End_Time', 'Distance(mi)', 'Description',
                      'Start_Lat',
                      'Start_Lng', 'End_Lat', 'End_Lng', 'Number', 'Street', 'Side', 'City', 'County',
                      'State', 'Zipcode', 'Country', 'Timezone', 'Airport_Code']
    accident_bl_df = accident_df[basic_location]

    # Dropping the following columns:
    # 1. ID: It'll be unique for all the rows and hence not required.
    # 2. Country: Since we are doing the analysis for the US
    # 3. Dropping state as it's unique for each city that we have chosen
    # 4. Number: This column has too many missing values and hence dropping it
    # 5. Zipcode: We already have enough location information
    # 6. Airport_Code: This carries the code for the weather station, not really needed.

    accident_bl_df = accident_bl_df.drop(
        ['ID', 'Country', 'State', 'Number', 'Zipcode', 'Airport_Code', 'End_Lat', 'End_Lng'], axis=1)

    # Convert start and end time to datetime type. This would help in the feature extraction step
    accident_bl_df['Start_Time'] = pd.to_datetime(accident_bl_df['Start_Time'], errors='coerce')
    accident_bl_df['End_Time'] = pd.to_datetime(accident_bl_df['End_Time'], errors='coerce')

    return accident_bl_df
