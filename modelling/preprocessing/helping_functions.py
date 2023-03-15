import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
from pyproj import Proj, transform
from tqdm import tqdm

def read_shape_file(path: str):
    shp = gpd.read_file(path)
    return shp

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