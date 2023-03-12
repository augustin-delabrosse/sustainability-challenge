import pyproj
import numpy as np
import pandas as pd
import math
import geopandas as gpd

def preprocess_data(df):
    """
    Preprocesses a given DataFrame.

    Args:
    - df (pandas DataFrame): input DataFrame to preprocess

    Returns:
    - df (pandas DataFrame): preprocessed DataFrame
    """
    df['longueur'] = df['longueur'].str.replace(',', '.')
    df['longueur'] = df['longueur'].astype(float)
    df['ratio_PL'] = df['ratio_PL'].str.replace(',', '.')
    df['ratio_PL'] = df['ratio_PL'].fillna(0.0)
    df['ratio_PL'] = df['ratio_PL'].astype(float)

    #fix ratio_PL issue
    condition = df['ratio_PL'] > 40
    df.loc[condition, 'ratio_PL'] /= 10

    #Add density
    df['TMJA_PL'] = round((df['TMJA']*(df['ratio_PL']/100)),2)

    # Calculate the sum of the Avg TMJA_PL column
    tmja_sum = df['TMJA_PL'].sum()

    # Calculate the percentage of traffic in each region
    df['percentage_traffic'] = round(df['TMJA_PL'] / tmja_sum, 2)

    return df

def grouped_region(df, shape_file):
    """
    Preprocesses a given DataFrame to group per region, sum the total distance of roads, and add the shape file informations.

    Args:
    - df (pandas DataFrame): input DataFrame to preprocess
    - shape_file: the path of the shape file

    Returns:
    - df (pandas DataFrame): new DataFrame
    """

    #load the shape file
    map_df = gpd.read_file(shape_file)
    map_df['nom'] = map_df['nom'].astype(str)

    #groupby region, sum the distance and average the traffic and surface
    df = df.groupby(['region']).agg({'longueur': 'sum', 'TMJA_PL': 'mean'}).reset_index()
    df['TMJA_PL'] = round(df['TMJA_PL'], 2)
    df['longueur'] = df['longueur'] / 1000 # convert to thousands of kilometers
    df = df.rename(columns={'longueur': 'longueur (K km)', 'TMJA_PL': 'Avg TMJA_PL'})

    #merges shape file
    merged = map_df.merge(df, left_on='nom', right_on='region')
    merged = merged[merged.columns[4:9]]
    merged['surf_km2'] = merged['surf_km2'] / 1000  # convert to K km²

    #add density column
    df_grouped_r = merged
    df_grouped_r['density_road (K km/km2)'] = round((df_grouped_r['longueur (K km)']*df_grouped_r['surf_km2']),2)

    return df_grouped_r

def distance_road_region(data_path):
    """
    Takes in a data path, load the DF and sums the total distance of National and Autoroute roads

    Args:
    - data_path (pandas DataFrame): path to .xlsx file

    Returns:
    - df (pandas DataFrame): DataFrame
    """
    df = pd.read_excel(data_path, sheet_name='REG')
    df['Routes_tot'] = df['Autoroutes'] + df['Routes nationales']
    df = df[df.columns[[0, 1, 2, 4]]]

    return df


def add_lat_lon_columns(df):
    """
    Adds new columns 'lonD', 'latD', 'lonF', 'latF' to the dataframe with
    corresponding latitude and longitude values based on the 'xD', 'yD', 'xF',
    and 'yF' columns, which are in Lambert-93 projection.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the columns 'xD', 'yD', 'xF', and 'yF'.

    Returns:
    --------
    pandas.DataFrame
        The original dataframe with the new 'lonD', 'latD', 'lonF', and 'latF'
        columns added.
    """
    # Define the input and output projections
    in_proj = pyproj.Proj(init='epsg:2154')  # Lambert-93
    out_proj = pyproj.Proj(init='epsg:4326')  # WGS84

    df['xD'] = df['xD'].str.replace(',', '.')
    df['yD'] = df['yD'].str.replace(',', '.')
    df['xF'] = df['xF'].str.replace(',', '.')
    df['yF'] = df['yF'].str.replace(',', '.')

    # Convert start coordinates to lat-long
    df['lonD'], df['latD'] = pyproj.transform(in_proj, out_proj, df['xD'], df['yD'])

    # Convert end coordinates to lat-long
    df['lonF'], df['latF'] = pyproj.transform(in_proj, out_proj, df['xF'], df['yF'])

    return df


def add_region_column(df):
    # Create a dictionary mapping department codes to regions
    dep_to_region = {
        '01': 'Auvergne-Rhône-Alpes',
        '02': 'Hauts-de-France',
        '03': 'Auvergne-Rhône-Alpes',
        '04': 'Provence-Alpes-Côte d\'Azur',
        '05': 'Provence-Alpes-Côte d\'Azur',
        '06': 'Provence-Alpes-Côte d\'Azur',
        '07': 'Auvergne-Rhône-Alpes',
        '08': 'Grand Est',
        '09': 'Occitanie',
        '1': 'Auvergne-Rhône-Alpes',
        '2': 'Hauts-de-France',
        '3': 'Auvergne-Rhône-Alpes',
        '4': 'Provence-Alpes-Côte d\'Azur',
        '5': 'Provence-Alpes-Côte d\'Azur',
        '6': 'Provence-Alpes-Côte d\'Azur',
        '7': 'Auvergne-Rhône-Alpes',
        '8': 'Grand Est',
        '9': 'Occitanie',
        '10': 'Grand Est',
        '11': 'Occitanie',
        '12': 'Occitanie',
        '13': 'Provence-Alpes-Côte d\'Azur',
        '14': 'Normandie',
        '15': 'Auvergne-Rhône-Alpes',
        '16': 'Nouvelle-Aquitaine',
        '17': 'Nouvelle-Aquitaine',
        '18': 'Centre-Val de Loire',
        '19': 'Nouvelle-Aquitaine',
        '21': 'Bourgogne-Franche-Comté',
        '22': 'Bretagne',
        '23': 'Nouvelle-Aquitaine',
        '24': 'Nouvelle-Aquitaine',
        '25': 'Bourgogne-Franche-Comté',
        '26': 'Auvergne-Rhône-Alpes',
        '27': 'Normandie',
        '28': 'Centre-Val de Loire',
        '29': 'Bretagne',
        '2A': 'Corse',
        '2B': 'Corse',
        '30': 'Occitanie',
        '31': 'Occitanie',
        '32': 'Occitanie',
        '33': 'Nouvelle-Aquitaine',
        '34': 'Occitanie',
        '35': 'Bretagne',
        '36': 'Centre-Val de Loire',
        '37': 'Centre-Val de Loire',
        '38': 'Auvergne-Rhône-Alpes',
        '39': 'Bourgogne-Franche-Comté',
        '40': 'Nouvelle-Aquitaine',
        '41': 'Centre-Val de Loire',
        '42': 'Auvergne-Rhône-Alpes',
        '43': 'Auvergne-Rhône-Alpes',
        '44': 'Pays de la Loire',
        '45': 'Centre-Val de Loire',
        '46': 'Occitanie',
        '47': 'Nouvelle-Aquitaine',
        '48': 'Occitanie',
        '49': 'Pays de la Loire',
        '50': 'Normandie',
        '51': 'Grand Est',
        '52': 'Grand Est',
        '53': 'Pays de la Loire',
        '54': 'Grand Est',
        '55': 'Grand Est',
        '56': 'Bretagne',
        '57': 'Grand Est',
        '58': 'Bourgogne-Franche-Comté',
        '59': 'Hauts-de-France',
        '60': 'Hauts-de-France',
        '61': 'Normandie',
        '62': 'Hauts-de-France',
        '63': 'Auvergne-Rhône-Alpes',
        '64': 'Nouvelle-Aquitaine',
        '65': 'Occitanie',
        '66': 'Occitanie',
        '67': 'Grand Est',
        '68': 'Grand Est',
        '69': 'Auvergne-Rhône-Alpes',
        '70': 'Bourgogne-Franche-Comté',
        '71': 'Bourgogne-Franche-Comté',
        '72': 'Pays de la Loire',
        '73': 'Auvergne-Rhône-Alpes',
        '74': 'Auvergne-Rhône-Alpes',
        '75': 'Île-de-France',
        '76': 'Normandie',
        '77': 'Île-de-France',
        '78': 'Île-de-France',
        '79': 'Nouvelle-Aquitaine',
        '80': 'Hauts-de-France',
        '81': 'Occitanie',
        '82': 'Occitanie',
        '83': 'Provence-Alpes-Côte d\'Azur',
        '84': 'Provence-Alpes-Côte d\'Azur',
        '85': 'Pays de la Loire',
        '86': 'Nouvelle-Aquitaine',
        '87': 'Nouvelle-Aquitaine',
        '88': 'Grand Est',
        '89': 'Bourgogne-Franche-Comté',
        '90': 'Bourgogne-Franche-Comté',
        '91': 'Île-de-France',
        '92': 'Île-de-France',
        '93': 'Île-de-France',
        '94': 'Île-de-France',
        '95': 'Île-de-France',
        '971': 'Guadeloupe',
        '972': 'Martinique',
        '973': 'Guyane',
        '974': 'La Réunion',
        '976': 'Mayotte'}
    
    df['region'] = df['depPrD'].astype(str).map(dep_to_region)

    return df

def create_part1_data(route_region_path, region_df):
    """
    Takes in a data path, load the DF and merges with the previously processed region dataframe

    Args:
    - data_path (pandas DataFrame): path to .xlsx file

    Returns:
    - df (pandas DataFrame): DataFrame with route distances per region
    """
    route_data = distance_road_region(route_region_path)

    # merge region and route to have the AVG TMJA_PL
    df_route_traffic = route_data.merge(region_df, left_on="Région", right_on="region")
    df_route_traffic = df_route_traffic[df_route_traffic.columns[[0, 1, 2, 3, 8]]]
    
    # Calculate the sum of the Avg TMJA_PL column
    tmja_sum = df_route_traffic['Avg TMJA_PL'].sum()

    # Calculate the percentage of traffic in each region
    df_route_traffic['percentage_traffic'] = round(df_route_traffic['Avg TMJA_PL'] / tmja_sum, 2)

    return df_route_traffic


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
        total_hydrogen = hydrogen_dailmer + hydrogen_nikola + hydrogen_daf

        #stations needed
        stations = total_hydrogen / capacity_stations

        # Append region and number of hydrogen stations to results list
        results.append({'Region': row['Région'], 'Hydrogen Stations Needed': math.ceil(stations)})
        
    # Create dataframe from results list and return it
    results_df = pd.DataFrame(results)

    return results_df
    
