#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os


dirname = os.path.dirname(__file__)
relative_path = '../../data/raw/accident_data.csv'
datafile = os.path.join(dirname, relative_path)
data = pd.read_csv(datafile)
print(data.shape)


city_list = ['Phoenix', 'Los Angeles', 'New York', 'Philadelphia', 'Houston', 'Chicago']
state_list = ['AZ', 'CA', 'NY', 'PA', 'TX', 'IL']

def isolate_city_state(data, cities, states):
    ''' This ensures that each city is selected with it's respective state
    which is why I didn't simply run a merge statement.'''
    
    for x in zip(cities, states):
        df_x = data.loc[(data['City'] == x[0]) & (data['State'] == x[1])]
        df_x.append(df_x)
    
    return df_x


raw_data = isolate_city_state(data, city_list, state_list)
print(raw_data.shape, '\n\n', raw_data.head())


raw_data.to_csv('C:\\...\\raw_data.csv')

