import numpy as np
import pandas as pd 
import geopandas as gpd
import pyproj 
from shapely.geometry import Point
from shapely.geometry import LineString

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


def transform_station_coordinates(
        df_stations: pd.DataFrame
        )-> pd.DataFrame:
    """
    Transform the coordinate column, which is string of latitude, longitude.
    into the corresponding Lambert-93 projection.

    Parameters:
    -----------
    df_stations : pandas.DataFrame
        The dataframe containing the stations coordinates in the column: Coordinates

    Returns:
    --------
    pandas.DataFrame
        The original dataframe with the new coordinate column in Lambert-93 projection: Coordinate_transform.
    """

    # Define the input and output projections
    in_proj = pyproj.Proj(init='epsg:2154')  # Lambert-93: systeme route
    out_proj = pyproj.Proj(init='epsg:4326')  # WGS84: long, lat

    transform_coord = lambda x: pyproj.transform(out_proj, in_proj, float(x.split(',')[0]),float(x.split(',')[1]))

    coord_transform = []
    for x in df_stations['Coordinates'].values:
        coord_transform.append(transform_coord(x))
    
    df_stations['Coordinate_transform'] = coord_transform

    df_stations = df_stations.reset_index(drop=False)

    return df_stations
