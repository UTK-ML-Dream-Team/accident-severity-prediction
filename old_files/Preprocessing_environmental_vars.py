#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import NaN

raw_data = pd.read_csv('C:\\...\\raw_data.csv')

print(raw_data.shape)

env_vars = ['Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
            'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition',
            'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']

def subset_df(df, keep_list):
    mask = df.columns.isin(keep_list)
    selectedCols = df.columns[mask]
    return df[selectedCols]

raw_data = subset_df(raw_data, env_vars)

print('Percent of missing rows by column', '\n\n', raw_data.isnull().sum()/len(raw_data))

def OLS(x, y):    
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()
    print(results.summary())

def basic_impute(data):
    
    df_num = data.select_dtypes(include=np.number)

    for i in data:
        if i in df_num:
            data.loc[data.loc[:,i].isnull(),i] = df_num.loc[:,i].median()
        else:
            data.loc[data.loc[:,i].isnull(),i] = data.loc[:,i].mode()
    
    return data

x = subset_df(raw_data, ['Temperature(F)', 'Wind_Speed(mph)'])
OLS(x, np.array(raw_data['Wind_Chill(F)']))

raw_data['Wind_Chill(F)'].fillna((raw_data['Temperature(F)']*1.0778 + raw_data['Wind_Speed(mph)']*-0.7083), inplace=True)

print('Percent of missing rows by column', '\n\n', raw_data.isnull().sum()/len(raw_data))

raw_data = basic_impute(raw_data)

print('Percent of missing rows by column', '\n\n', raw_data.isnull().sum()/len(raw_data))
