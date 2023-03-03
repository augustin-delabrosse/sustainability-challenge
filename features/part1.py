import numpy as np
import pandas as pd
import geopandas as gpd
import math

def calculate_hydrogen_stations(df, daimler_perc, nikola_perc, daf_perc, year):
    """
    Calculate the number of hydrogen stations needed for a given scenario year based on the percentage of three brands of trucks.

    Args:
        df (pandas.DataFrame): The input dataframe with the traffic data for each region.
        daimler_perc (float): The percentage of Daimler trucks in the scenario (between 0 and 1).
        nikola_perc (float): The percentage of Nikola trucks in the scenario (between 0 and 1).
        daf_perc (float): The percentage of DAF trucks in the scenario (between 0 and 1).
        year (int): The year of the scenario.

    Returns:
        pandas.DataFrame: A dataframe with the region and the number of hydrogen stations needed.
    """ 
    # Assumptions and constraints
    avg_speed = 80  # km/h
    max_drive_time_daily = 9  # hours
    break_time = 0.33 # hours
    autonomy_daimler = 1000  # km
    autonomy_nikola = 400  # km
    autonomy_daf = 150  # km
    tank_size_daimler = 80
    tank_size_nikola = 32
    tank_size_daf = 30
    capacity_stations = 3*1000

    # Determine the total number of trucks based on the scenario year
    if year == 2030:
        total_trucks = 10000
    elif year == 2040:
        total_trucks = 60000
    else:
        raise ValueError("Year must be 2030 or 2040")
    
    # Calculate the number of trucks for each brand based on the percentages
    num_daimler_trucks = int(total_trucks * daimler_perc)
    num_nikola_trucks = int(total_trucks * nikola_perc)
    num_daf_trucks = int(total_trucks * daf_perc)

    # Initialize list to store results
    results = []

    # Loop through each row in the dataframe and perform necessary calculations
    for index, row in df.iterrows():
        #calculate max daily distance
        max_daily_distance = avg_speed * max_drive_time_daily

        # Calculate the number of stops per day for each truck
        daimler_num_stops_per_day = math.ceil(max_daily_distance / autonomy_daimler)
        nikola_num_stops_per_day = math.ceil(max_daily_distance / autonomy_nikola)
        daf_num_stops_per_day = math.ceil(max_daily_distance / autonomy_daf)

        # Calculate the total time for each truck to cover the region
        total_time = row['Routes_tot'] / avg_speed

        # Calculate the working time for each truck, including breaks and recharge times
        dailmer_working_time = total_time + (daimler_num_stops_per_day * break_time)
        nikola_working_time = total_time + (nikola_num_stops_per_day * break_time)
        daf_working_time = total_time + (daf_num_stops_per_day * break_time)
    
        # Calculate the number of days each truck needs to cover the region
        dailmer_day_region = math.ceil(dailmer_working_time / max_drive_time_daily)
        nikola_day_region = math.ceil(nikola_working_time / max_drive_time_daily)
        daf_day_region = math.ceil(daf_working_time / max_drive_time_daily)
        
        # Calculate the total hydrogen consumption for each truck in each region
        hydrogen_dailmer = daimler_num_stops_per_day * tank_size_daimler * (num_daimler_trucks * row['percentage_traffic'])
        hydrogen_nikola = nikola_num_stops_per_day * tank_size_nikola * (num_nikola_trucks * row['percentage_traffic'])
        hydrogen_daf = daf_num_stops_per_day * tank_size_daf * (num_daf_trucks * row['percentage_traffic'])

        #total hydrogen needed daily
        total_hydrogen = hydrogen_dailmer + hydrogen_dailmer + hydrogen_daf

        #stations needed
        stations = total_hydrogen / capacity_stations

        # Append region and number of hydrogen stations to results list
        results.append({'Region': row['Région'], 'Hydrogen Stations Needed': math.ceil(stations)})
        
    # Create dataframe from results list and return it
    results_df = pd.DataFrame(results)

    return results_df
    

