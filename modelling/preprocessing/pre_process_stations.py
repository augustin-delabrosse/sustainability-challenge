import numpy as np
import pandas as pd 
import geopandas as gpd
import pyproj 
from shapely.geometry import Point
from shapely.geometry import LineString

from features.config import *
from preprocessing.helping_functions import *
from preprocessing.pre_process_traffic import *


######################################################################
# Part 1, 2 and 3: Preprocess the data stations

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
