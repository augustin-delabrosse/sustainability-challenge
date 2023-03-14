import numpy as np
import pandas as pd 
import geopandas as gpd
import pyproj 
from shapely.geometry import Point
from shapely.geometry import LineString
from features.config import *

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

def create_coordinate_column_station(
        df_stations: pd.DataFrame
):
    '''
    From the Coordinate column: which is a string of coordingates as follow: latitude, longitude
    
    Parameters:
    -----------
    df_stations : pandas.DataFrame
        The dataframe containing the stations coordinates in the column: Coordinates

    Returns:
    --------
    pandas.DataFrame
        The original dataframe with the new Latitude and Longitude columnn.
    '''
    #create Latitude and Longitude coordinates 
    df_stations['Latitude'] = df_stations['Coordinates'].apply(lambda x: x.split(',')[0].strip())
    df_stations['Longitude'] = df_stations['Coordinates'].apply(lambda x: x.split(',')[1].strip())

    #filter out the empty coordinates
    df_stations = df_stations[(df_stations['Longitude'] != '')&(df_stations['Latitude'] != '')]

    return df_stations

def fix_stations(df):
    """Replaces the double commas in the 'Coordinates' column of a dataframe with a single comma.

    Parameters:
    -----------
    df : pandas.DataFrame (or gpd.geodataframe.GeoDataFrame)
        The (geo)dataframe to fix.

    Returns:
    --------
    pandas.DataFrame (or gpd.geodataframe.GeoDataFrame)
        The fixed (geo)dataframe."""

    df['Coordinates'] = df['Coordinates'].apply(lambda x: x.replace(',,', ','))
    return df

def clean_stations(df_stations):
    df_stations = create_coordinate_column_station(df_stations)
    df_stations = df_stations.drop_duplicates(['URL'])
    df_stations['geometry'] = gpd.GeoSeries.from_wkt(df_stations['geometry'])
    return df_stations

p = Parameters_filter_stations()

def filter_stations(
    df_stations,
    max_dist_road = p.max_dist_road,
    max_dist_dense_hub = p.max_dist_dense_hub,
    max_dist_hub = p.max_dist_hub
):  
    df_stations = df_stations[df_stations['distance_to_closest_road']<=max_dist_road]
    df_stations = df_stations[df_stations['distance_to_closest_dense_hub']<=max_dist_dense_hub]
    df_stations = df_stations[df_stations['distance_to_closest_large_hub']<=max_dist_hub]

    return df_stations

def create_region_columns(
        df: gpd.GeoDataFrame,
        data_region: gpd.GeoDataFrame
):
    data_region['geometry'] = data_region['geometry'].to_crs('epsg:2154')

    for i,region_name in enumerate(data_region['nom']):
        df[region_name] = data_region.loc[i,'geometry'].contains(df['geometry'])

    return df