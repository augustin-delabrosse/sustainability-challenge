import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
from pyproj import Proj, transform
from tqdm import tqdm


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