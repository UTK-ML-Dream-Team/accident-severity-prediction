#!/usr/bin/env python
# coding: utf-8

# This notebook preprocess the infrastructure variables of the accident data.
# 
# The infrastructure variables include:
# 
# 'Traffic_Signal', 'Crossing', 'Station','Amenity', 'Bump', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout','Stop', 'Traffic_Calming', 'Turning_Loop'.

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


accident_df = pd.read_csv('../data/raw/accident_data.csv')
infra_vars = ['Traffic_Signal', 'Crossing', 'Station','Amenity', 'Bump', 'Give_Way', 
                  'Junction', 'No_Exit', 'Railway', 'Roundabout','Stop', 'Traffic_Calming', 'Turning_Loop']
infra_accident_df = accident_df[infra_vars]


# In[3]:


print('Number of missing rows by column', '\n', infra_accident_df.isnull().sum())


# There is no missing value and duplicated meaing variables, all of the infra structures variables would be used. 
