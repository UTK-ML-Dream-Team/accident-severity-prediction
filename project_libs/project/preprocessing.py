#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import NaN


def isolate_city_state(data, cities, states):
    ''' This ensures that each city is selected with it's respective state
    which is why I didn't simply run a merge statement.'''

    for x in zip(cities, states):
        df_x = data.loc[(data['City'] == x[0]) & (data['State'] == x[1])]
        df_x.append(df_x)

    return df_x

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
