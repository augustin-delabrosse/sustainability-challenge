import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pyproj 
from tqdm import tqdm

from shapely.geometry import Point
from shapely.geometry import LineString

from preprocessing.pre_process_stations import *
from preprocessing.pre_process_traffic import *
from preprocessing.helping_functions import *

from features.config import *
from features.financials import *

from models.question_1 import *
from models.question_2 import *

from models.question_3_genetic_algorithm import *

def plot_results(roads_shapefile:gpd.geodataframe.GeoDataFrame, df_results:pd.core.frame.DataFrame):
    """
    Plot the results of a simulation of H2 station placement on a map.

    Parameters:
        roads_shapefile : GeoDataFrame
            A GeoDataFrame containing the roads' shapefile data.

        df_results : DataFrame
            A DataFrame containing the H2 station placement simulation results.
            It should contain a 'geometry' column containing the stations' location in
            string or Shapely.geometry format, and a 'station_type' column with the
            type of station (i.e., small, medium, large).

    Returns:
        A plot of the roads and H2 stations overlaid on a map.

    """
    roads = gpd.GeoDataFrame({'geometry': roads_shapefile.geometry, 
                              'type': ['route' for i in range(len(roads_shapefile.geometry))],
                              'nom': roads_shapefile.route}, 
                             crs=roads_shapefile.crs)
    
    if type(df_results.geometry[:1].values[0]) == str:
        df_results = convert_str_geometry_to_geometry_geometry(df_results)
        
    stations = gpd.GeoDataFrame({'geometry': df_results.geometry, 
                                 'type': [f'{i} H2 station' for i in df_results["station_type"]],
                                 'nom': [f'H2 station n{i}' for i in range(df_results.shape[0])]}, 
                                crs=roads_shapefile.crs)
    
    shp_file = pd.concat([roads, stations])
    exploration = shp_file.explore(column='type', cmap='tab10')
    
    return exploration

def plotting_installations(df:pd.core.frame.DataFrame):
    """
    Plot the installation dates of H2 stations on a map.

    Parameters:
        df : DataFrame
            A DataFrame containing data on H2 stations' installation, including their
            location and installation date.

    Returns:
        A plot of the H2 station locations overlaid on a map, with the color of each
            station indicating its installation date.

    """

    if type(df.geometry[:1].values[0]) == str:
        df = convert_str_geometry_to_geometry_geometry(df)
        
    shp_file = gpd.GeoDataFrame(df, crs="epsg:2154")
    shp_file = shp_file[['URL', 'nom_region', 'geometry', 'closest_road',
       'closest_large_hub', 'closest_dense_hub', 'TMJA_PL', 'percentage_traffic',
       'Quantity_sold_per_day(in kg)', 'Revenues', 'bool', 'size',
       'Quantity_sold_per_year(in kg)', 'station_type', 'Revenues_day',
       'EBITDA', 'Opex', 'EBIT', 'depreciation', 'date_installation']]
    
    exploration = shp_file.explore(column="date_installation", cmap="Blues")
    
    return exploration


def plot_clusters_(production_sites, routes, stations, cmap:str="gist_ncar"):
    """
    Plot the given production sites, routes, and clustered stations on a map.

    Args:
    - production_sites: a geopandas GeoDataFrame containing the locations of production sites.
    - routes: a geopandas GeoDataFrame containing the routes.
    - stations: a geopandas GeoDataFrame containing the locations of stations clustered by k-means (it needs a column called 'cluster').
    - cmap: a string specifying the name of the matplotlib colormap to use for coloring the map.

    Returns:
    A holoviews Layout object containing the plotted map.

    """
    shp_production_sites = gpd.GeoDataFrame({'geometry': production_sites.geometry, 
                                         'nom': [f'{i+1}th production site' for i in range(production_sites.shape[0])]}, 
                                        crs="epsg:2154")
    
    shp_routes = gpd.GeoDataFrame({'geometry': routes.geometry, 
                                         'nom': ['0 route' for i in range(routes.shape[0])]}, 
                                        crs="epsg:2154")
    
    shp_file = pd.concat([shp_routes, shp_production_sites])

    mask = shp_file['nom'] != "0 route"
    shp_file.loc[mask, 'geometry'] = shp_file.loc[mask, 'geometry'].apply(lambda x: x.buffer(50000))


    shp_stations = gpd.GeoDataFrame({'geometry': stations.geometry, 
                                         'nom': [f'{i+1}th cluster of stations' for i in stations.cluster]}, 
                                        crs="epsg:2154")
    
    shp_file = pd.concat([shp_file, shp_stations])
    
    exploration = shp_file.explore(column='nom', cmap=cmap)
    
    return exploration