import pandas as pd
from preprocessing.helping_functions import *
from preprocessing.pre_process_stations import *
from preprocessing.pre_process_traffic import *
import reverse_geocoder as rg

######################################################################
# Preprocess the traffic data from the tmja 2019 csv

def preprocess_data(df):
    """
    Preprocesses a given DataFrame.
    Args:
    - df (pandas DataFrame): input DataFrame to preprocess
    Returns:
    - df (pandas DataFrame): preprocessed DataFrame
    """

    #fix ratio_PL issue
    condition = df['ratio_PL'] > 40
    df.loc[condition, 'ratio_PL'] /= 10

    #Add density
    df['TMJA_PL'] = round((df['TMJA']*(df['ratio_PL']/100)),2)

    mask = df['TMJA_PL'].notna()

    df = df[mask]

    # Calculate the percentage of traffic in each region
    df['percentage_traffic'] = df['TMJA_PL'] / df['TMJA_PL'].sum()

    return df



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
