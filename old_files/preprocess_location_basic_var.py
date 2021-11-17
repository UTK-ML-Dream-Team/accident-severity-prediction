import os
import numpy as np
import pandas as pd

dirname = os.path.dirname(__file__)
relative_path = '../../data/raw/accident_data.csv'
datafile = os.path.join(dirname, relative_path)

def preprocess_loc_basic_var():
    """
    Function to preprocess the location and basic variables. Note that in our dataset
    we didn't find missing values for most of the columns except for Number which was
    missing for almost 70% of the rows.
    """

    accident_df = pd.read_csv(datafile)
    basic_location = ['ID', 'Severity', 'Start_Time', 'End_Time', 'Distance(mi)', 'Description', 'Start_Lat', 
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

    accident_bl_df = accident_bl_df.drop(['ID', 'Country', 'State', 'Number', 'Zipcode', 'Airport_Code'], axis=1)

    # Convert start and end time to datetime type. This would help in the feature extraction step
    accident_bl_df['Start_Time'] = pd.to_datetime(accident_bl_df['Start_Time'], errors='coerce')
    accident_bl_df['End_Time'] = pd.to_datetime(accident_bl_df['End_Time'], errors='coerce')

    return accident_bl_df