import geopandas as gpd
import pandas as pd
import numpy as np

import pyproj
from pyproj import Proj, transform
import reverse_geocoder as rg

from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')


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

    df['xD'] = df['xD'].replace(',', '.')
    df['yD'] = df['yD'].replace(',', '.')
    df['xF'] = df['xF'].replace(',', '.')
    df['yF'] = df['yF'].replace(',', '.')

    # Convert start coordinates to lat-long
    df['lonD'], df['latD'] = pyproj.transform(in_proj, out_proj, df['xD'], df['yD'])

    # Convert end coordinates to lat-long
    df['lonF'], df['latF'] = pyproj.transform(in_proj, out_proj, df['xF'], df['yF'])

    return df

def read_shape_file(path: str):
    """
    Reads a shapefile and returns it as a GeoDataFrame.

    Parameters:
    -----------
    path : str
        The path to the shapefile.

    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing the shapefile data.
        """

    shp = gpd.read_file(path)
    return shp

def fix_tmja(df):
    """Fixes the 'longueur' and 'ratio_PL' columns of a dataframe, adds latitude and longitude columns,
    and adds 'region' and 'departement' columns based on the latitude and longitude.

    Parameters:
    -----------
    df : pandas.DataFrame (or gpd.geodataframe.GeoDataFrame)
        The (geo)dataframe to fix.

    Returns:
    --------
    pandas.DataFrame (or gpd.geodataframe.GeoDataFrame)
        The fixed (geo)dataframe."""

    # Fixing a few columns
    df.longueur = df.longueur.apply(lambda x: float(x.replace(',', '.')))
    df['ratio_PL'] = df['ratio_PL'].apply(lambda x: x if x<=40 else x/10)
    
    # Add lattitude and longitude
    df = add_lat_lon_columns(df)
    
    # Add region and department with lattitude and longitude
    coordinates = [(i[1], i[0]) for i in df[['lonD', 'latD']].values]
    results = rg.search(coordinates)
    df['region'] = [i['admin1'] for i in results]
    df['departement'] = [i['admin2']\
                         .replace('Departement de ', '')\
                         .replace('Departement du ', '')\
                         .replace('Departement des ', '')\
                         .replace("Departement d'", '')\
                         .replace('la ', '')\
                         .replace("l'", "") for i in results]
    
    return df

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

def convert_long_lat_to_easting_northing(df, initial_epsg='epsg:4326', target_epsg='epsg:2154'):
    """Converts latitude and longitude coordinates to easting and northing coordinates
    using the given input and output projections.

    Parameters:
    -----------
    df : pandas.DataFrame (or gpd.geodataframe.GeoDataFrame)
        The dataframe containing the 'Coordinates' column with the latitude and longitude data.
    initial_epsg : str, optional
        The EPSG code for the initial coordinate system. Default is 'epsg:4326' (WGS84).
    target_epsg : str, optional
        The EPSG code for the target coordinate system. Default is 'epsg:2154' (Lambert-93).

    Returns:
    --------
    pandas.DataFrame (or gpd.geodataframe.GeoDataFrame)
        A (Geo)DataFrame containing the converted data."""

    # Define the input and output projections
    in_proj = pyproj.Proj(init='epsg:4326')  # Lambert-93
    out_proj = pyproj.Proj(init='epsg:2154')  # WGS84

    long = df['Coordinates'].apply(lambda x: x.split(',')[1]).values.astype(float)
    lat = df['Coordinates'].apply(lambda x: x.split(',')[0]).values.astype(float)

    # Convert start coordinates to lat-long
    long_transformed, lat_transformed = pyproj.transform(in_proj, out_proj, long, lat)
    
    df.geometry = gpd.points_from_xy(x=long_transformed, 
                                     y=lat_transformed, crs='epsg:2154')
    
    return df

def convert_str_geometry_to_geometry_geometry(df):
    """Converts a geometry column with geometry shapes writen as string to a geometry shapes writen as geometry shapes column

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with the geometry column to convert.

    Returns:
    --------
    pandas.DataFrame
        The converted dataframe."""

    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    return df

def new_coordinates_creation(approximate_nb_of_points, shapefile_tmja: gpd.geodataframe.GeoDataFrame):
    """Creates new coordinates along a route by splitting the route into equal
    length segments and interpolating new points along each segment.

    Parameters:
    -----------
    approximate_nb_of_points : int
        The approximate number of points to create along each route.
    shapefile_tmja : gpd.geodataframe.GeoDataFrame
        The shapefile to create new coordinates for.

    Returns:
    --------
    gpd.geodataframe.GeoDataFrame
        A GeoDataFrame containing the new coordinates."""

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

def compute_distance_to_closest_large_hub(df_stations:gpd.geodataframe.GeoDataFrame, df_hub_elargies:gpd.geodataframe.GeoDataFrame):
    """Compute the closest large hub to each station in a GeoDataFrame of stations.

    Args:
        df_stations (gpd.geodataframe.GeoDataFrame): A GeoDataFrame of stations.
        df_hub_elargies (gpd.geodataframe.GeoDataFrame): A GeoDataFrame of large hubs.

    Returns:
        gpd.geodataframe.GeoDataFrame: A new GeoDataFrame of stations with additional columns 'closest_large_hub' and 'distance_to_closest_large_hub' that indicate the closest large hub to each station and the distance to that hub, respectively.
    """
    
    large_hub_list = []
    distance_to_large_hub_list = []

    for idx_station in tqdm(df_stations.index):
        geodf = gpd.GeoDataFrame(df_stations.iloc[idx_station].geometry.distance(df_hub_elargies.geometry))
        min_distance = df_stations.iloc[idx_station].geometry.distance(df_hub_elargies.geometry).min()
        idx_hub = geodf[geodf['geometry'] == min_distance].index[0]
        # print(stations.loc[idx_station, 'nom'], roads.loc[idx_road, 'nom'], min_distance)
        large_hub_list.append(df_hub_elargies.at[idx_hub, 'e1'])
        distance_to_large_hub_list.append(min_distance)


    df_stations['closest_large_hub'] = large_hub_list
    df_stations['distance_to_closest_large_hub'] = distance_to_large_hub_list
    
    return df_stations


def compute_distance_to_closest_dense_hub(df_stations:gpd.geodataframe.GeoDataFrame, df_hub_denses:gpd.geodataframe.GeoDataFrame):
    """Compute the closest dense hub to each station in a GeoDataFrame of stations.

    Args:
        df_stations (gpd.geodataframe.GeoDataFrame): A GeoDataFrame of stations.
        df_hub_denses (gpd.geodataframe.GeoDataFrame): A GeoDataFrame of dense hubs.

    Returns:
        gpd.geodataframe.GeoDataFrame: A new GeoDataFrame of stations with additional columns 'closest_dense_hub' and 'distance_to_closest_dense_hub' that indicate the closest dense hub to each station and the distance to that hub, respectively.
    """

    dense_hub_list = []
    distance_to_dense_hub_list = []

    for idx_station in tqdm(df_stations.index):
        geodf = gpd.GeoDataFrame(df_stations.iloc[idx_station].geometry.distance(df_hub_denses.geometry))
        min_distance = df_stations.iloc[idx_station].geometry.distance(df_hub_denses.geometry).min()
        idx_hub = geodf[geodf['geometry'] == min_distance].index[0]
        # print(stations.loc[idx_station, 'nom'], roads.loc[idx_road, 'nom'], min_distance)
        dense_hub_list.append(df_hub_denses.at[idx_hub, 'e1'])
        distance_to_dense_hub_list.append(min_distance)

    df_stations['closest_dense_hub'] = dense_hub_list
    df_stations['distance_to_closest_dense_hub'] = distance_to_dense_hub_list

    return df_stations

def compute_distance_to_each_station(geodf):
    """Compute the distance from each station in a GeoDataFrame to all other stations in the same GeoDataFrame.

    Args:
        geodf (gpd.geodataframe.GeoDataFrame): A GeoDataFrame of stations.

    Returns:
        gpd.geodataframe.GeoDataFrame: The input GeoDataFrame of stations with additional columns indicating the distance from each station to all other stations in the same GeoDataFrame.
    """
    for i in tqdm(geodf.index):
        URL = df.loc[i, 'URL']
        geodf[f'distance_to_{URL}'] = geodf.loc[i, 'geometry'].distance(geodf.geometry)
    return geodf

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
        df_results = convert_str_geometry_to_geometry_geometry(results)
        
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