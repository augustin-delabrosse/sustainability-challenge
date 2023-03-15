import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm 
from pyproj import Proj, transform

from preprocessing.helping_functions import *
from preprocessing.pre_process_stations import *
from preprocessing.pre_process_traffic import *

from features.financials import *
from features.config import *


def indicate_crs(shp_file: gpd.geodataframe.GeoDataFrame, epsg:str):
    """Sets the coordinate reference system (CRS) of a GeoDataFrame.

    Parameters:
    -----------
    shp_file : gpd.geodataframe.GeoDataFrame
        The GeoDataFrame to set the CRS for.
    epsg : str
        The EPSG code for the target CRS.

    Returns:
    --------
    gpd.geodataframe.GeoDataFrame
        The GeoDataFrame with the CRS set."""

    shp_file.set_crs(epsg, allow_override=True)
    return shp_file

def fix_stations(df):
    df['Coordinates'] = df['Coordinates'].apply(lambda x: x.replace(',,', ','))
    return df


def new_coordinates_creation(
        approximate_nb_of_points, 
        shapefile_tmja):
    """Creates new coordinates along a route by splitting the route into equal
    length segments and interpolating new points along each segment.
​
    Parameters:
    -----------
    approximate_nb_of_points : int
        The approximate number of points to create along each route.
    shapefile_tmja : gpd.geodataframe.GeoDataFrame
        The shapefile to create new coordinates for.

    Returns:
    --------
    gpd.geodataframe.GeoDataFrame
        A GeoDataFrame containing the new coordinates.
    """

    total_distance = shapefile_tmja.geometry.length.sum() # in meters
    distance_between_coordinates = total_distance/approximate_nb_of_points
    points = []
    routes = []

    for idx in tqdm(shapefile_tmja.index):
        line = shapefile_tmja.geometry[idx]
        route = shapefile_tmja.route[idx]
        n_splits = int(line.length/distance_between_coordinates)
        
        if n_splits < 2:
            splitter = [line.interpolate((i/2), normalized=True) for i in range(2)]
        else:
            splitter = [line.interpolate((i/n_splits), normalized=True) for i in range(n_splits)]

        points.append(splitter)
        routes.append([route for i in (range(n_splits) if n_splits >= 2 else range(2))])
        
    routes = np.concatenate(routes)
    
    coordinates = pd.DataFrame([i.wkt.replace('POINT (', '').replace(')', '').split(' ') for i in np.concatenate(points)])
    coordinates.columns = ["easting", "northing"]
    coordinates['route'] = routes
    coordinates['geometry'] = gpd.points_from_xy(x=coordinates.easting, 
                                                 y=coordinates.northing,
                                                 crs=shapefile_tmja.crs)
    
    shp_coordinates = gpd.GeoDataFrame(coordinates)
    
    return shp_coordinates

def station_distances_all(
        df_stations: pd.DataFrame
):
    '''
    Add columns of the distances between each stations 
    '''

    for i in tqdm(df_stations['index']):
        station_index = df_stations.loc[i, 'index']
        df_stations[f'distance_to_point_{station_index}'] = df_stations.loc[i, 'geometry'].distance(df_stations['geometry'])

    return df_stations

def station_distances_station_total(
        df_stations_total: pd.DataFrame,
        df_stations: pd.DataFrame
):
    '''
    Add columns of the distances between each stations 
    '''

    for i in tqdm(df_stations_total['index']):
        station_index = df_stations_total.loc[i, 'index']
        df_stations[f'distance_to_stationtotal_{station_index}'] = df_stations_total.loc[i, 'geometry'].distance(df_stations['geometry'])

    return df_stations

def get_closer_station(
        df_stations_complete: pd.DataFrame
):
    '''
    Create a columns distance_closer_station with the smaller distance between each stations
    '''
    
    columns_distance = [x for x in df_stations_complete.columns if x.startswith('distance_to_point_')==True]
    df_stations_complete['distance_closer_station'] = df_stations_complete[columns_distance].apply(lambda x: min(x[x!=0]), axis=1)

    return df_stations_complete

def get_closer_station_total(
        df_stations_total: pd.DataFrame
):
    '''
    Create a columns distance_closer_station_total with the smaller distance between each total stations in H2 conversion
    '''
    
    columns_distance = [x for x in df_stations_total.columns if x.startswith('distance_to_stationtotal_')==True]
    df_stations_total['distance_closer_station_total'] = df_stations_total[columns_distance].apply(lambda x: min(x[x!=0]), axis=1)

    return df_stations_total

def distance_to_hub(
        hub_denses: pd.DataFrame,
        hub_elargies: pd.DataFrame,
        stations: pd.DataFrame):
    
    '''
    Add 4 columns regarding the distances to the closest hubs, for each point:
    - closest_dense_hub: the index of the closest dense hub 
    - closest_elargie_hub: the index of the closest enlarged hub 
    - distance_to_closest_dense_hub: the distance to the closest dense hub
    - distance_to_closest_elargie_hub: the distance to the closest enlarged hub 
    '''

    dense_hub_list_denses = []
    dense_hub_list_elargies = []
    distance_to_dense_hub_list_denses = []
    distance_to_dense_hub_list_elargies = []

    for idx_station in tqdm(stations.index):
        # for dense hubs
        geodf_denses = pd.DataFrame(hub_denses['geometry'].distance(stations.loc[idx_station,'geometry']))
        min_distance_denses = geodf_denses.min()
        idx_hub = geodf_denses.min().index[0]
        dense_hub_list_denses.append(hub_denses.at[idx_hub, 'e1'])
        distance_to_dense_hub_list_denses.append(min_distance_denses)

        # for enlarged hubs 
        geodf_elargies = pd.DataFrame(hub_elargies['geometry'].distance(stations.loc[idx_station,'geometry']))
        min_distance_elargies = geodf_elargies.min()
        idx_hub = geodf_elargies.min().index[0]
        dense_hub_list_elargies.append(hub_elargies.at[idx_hub, 'e1'])
        distance_to_dense_hub_list_elargies.append(min_distance_elargies)

    stations['closest_dense_hub'] = dense_hub_list_denses
    stations['distance_to_closest_dense_hub'] = distance_to_dense_hub_list_denses

    stations['closest_elargie_hub'] = dense_hub_list_elargies
    stations['distance_to_closest_large_hub'] = distance_to_dense_hub_list_elargies

    stations = stations.reset_index(drop=False)

    return stations