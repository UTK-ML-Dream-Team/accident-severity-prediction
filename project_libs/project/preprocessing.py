#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import NaN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
                        'Amenity','Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                        'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Side', 'Turning_Loop', 
                        'Sunrise_Sunset'
                    ]

    for features in binary_features:
        le = LabelEncoder()
        X[features] = le.fit_transform(X[features])


    # Create duration feature
    X['Start_Time'] = pd.to_datetime(X['Start_Time'])
    X['End_Time'] = pd.to_datetime(X['End_Time'])
    X['Duration'] = (X['End_Time'] - X['Start_Time']).dt.total_seconds() / 60.0 # Minutes

    # Create month, weekday, and hour feature
    X["Month"] = X["Start_Time"].dt.month
    X["DayOfWeek"]  = X["Start_Time"].dt.dayofweek
    X["Hour"]  = X["Start_Time"].dt.hour

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
    X = pd.concat([X, city_df, county_df, timezone_df, wind_direction_df, weather_condition_df], axis=1)

    # Drop non essential features
    non_essential_features = [
                                'ID', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng','End_Lat', 
                                'End_Lng', 'Description', 'Number', 'Street', 'State', 'Zipcode', 'Country', 
                                'Airport_Code', 'Weather_Timestamp', 'Roundabout','Civil_Twilight', 'Nautical_Twilight', 
                                'Astronomical_Twilight'
                            ]
    X.drop(non_essential_features, axis=1, inplace=True)
    
    # Drop one hot encoded features
    one_hot = ['City', 'County', 'Timezone', 'Month', 'DayOfWeek', 'Hour', 'Wind_Direction', 'Weather_Condition' ]
    X.drop(one_hot, axis=1, inplace=True)
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.10, shuffle=True, stratify=y, random_state=42)

    # Standardize Numerical Variables
    numerical_features = [
                          'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)',
                          'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 
                          'Precipitation(in)', 'Duration', 'Coord_X', 'Coord_Y', 'Coord_Z'
                        ]
    
    for feature in numerical_features:
        mu = X_train[feature].mean()
        sigma = X_train[feature].std()
        X_train[feature] = X_train[feature].apply(lambda x: (x-mu)/sigma)
        X_test[feature]  = X_test[feature].apply(lambda x: (x-mu)/sigma)
    
    return (X_train, X_test, y_train, y_test)

