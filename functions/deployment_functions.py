from shapely.geometry import Point
import os
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import Point, LineString, MultiPoint
import datetime
from tqdm import tqdm
import math
import random
from functions.helping_functions import *
from functions.Financials import *

## Part 3

def deployment_dates(df_station: pd.DataFrame, year_start: float, year_end: float) -> pd.DataFrame:
    '''
    This function assigns the year of deployment for each functions depending on its revenue
    '''

    sorted_df = df_station.sort_values(by='EBITDA', ascending=False)
    n_stations = len(sorted_df)
    n_years = year_end - year_start + 1

    # calculate the number of stations to deploy each year
    n_first_year = math.ceil(n_stations * 0.3)
    n_remaining_years = n_stations - n_first_year
    n_stations_per_remaining_year = math.ceil(n_remaining_years / (n_years - 1))
    n_stations_per_year = [n_first_year] + [n_stations_per_remaining_year] * (n_years - 1)

    # divide the stations into groups for each year
    groups = []
    for i in range(n_years):
        group_size = n_stations_per_year[i]
        start = sum(n_stations_per_year[:i])
        end = start + group_size
        group = sorted_df.iloc[start:end]
        groups.append(group)

    installation_years = [year_start] + [year_start + i for i in range(1, n_years)]
    installation_dates = {}
    for group, year in zip(groups, installation_years):
        for url in group['geometry']:
            installation_dates[url] = year

    date_df = pd.DataFrame({'geometry': df_station['geometry'], 'date_installation': [installation_dates.get(s, year_end) for s in df_station['geometry']]})

    # Merge the installation dates dataframe with the original dataframe
    merged_df = pd.merge(df_station, date_df, on='geometry')
    
    return merged_df


def scenario_select(df_deployments: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    This function randomly selects a given percentage of each station each year, and outputs a dataframe with all the randomly selected years.

    """
    years = df_deployments['date_installation'].unique()
    
    selected_years = {}
    for year in years:
        year_stations = df_deployments[df_deployments['date_installation'] == year]['geometry']
        n_select = int(round(percentage * len(year_stations)))
        selected_stations = random.sample(list(year_stations), n_select)
        # Store selected years for each station in the dictionary
        for station in selected_stations:
            selected_years[station] = year

    selected_years_df = pd.DataFrame.from_dict(selected_years, orient='index', columns=['random_year'])
    
    # Merge the selected years dataframe with the original dataframe
    output_df = pd.merge(df_deployments, selected_years_df, left_on='geometry', right_index=True)

    return output_df

def deployment_financials(df_station:pd.DataFrame,year_start: float,year_end: float)-> pd.DataFrame:
    '''
    This functions gives a detail financial overview of the cumulatitve stations metrics
    
    '''
    df_station = deployment_dates(df_station,year_start,year_end)
    years = sorted(df_station['date_installation'].unique())
    data = []
    for i, year in enumerate(years):
        cumulative_df = df_station[df_station['date_installation'] <= year]
        cumulative_data = cumulative_df[['Quantity_sold_per_year(in kg)', 'Revenues', 'Opex', 'EBITDA', 'depreciation', 'EBIT', 'small_station', 'medium_station', 'large_station']].sum()
        data.append(cumulative_data)

    df_cumulative_financials = pd.DataFrame(data, index=years, columns=['Quantity_sold_per_year(in kg)', 'Revenues', 'Opex', 'EBITDA', 'depreciation', 'EBIT', 'small_station', 'medium_station', 'large_station'])
    #df_cumulative_financials = df_cumulative_financials.applymap(lambda x: format(x, ',.0f') if isinstance(x, (int, float)) else x)

    return df_cumulative_financials

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

# Part 4
# Assumptions from slides
factory_info = {
    'station_type': ['small', 'large'],
    'capex': [20, 120],
    'depreciation': [0.15,0.15], 
    'opex': [0.03 * 20, 0.03 * 150],
    'Power_usage': [55, 50],  # Wh/kgH2 
    'water_consumption': [10, 10], #L/kgH2

}

df_factory_info = pd.DataFrame(factory_info).reset_index(drop=True)
Transport_by_truck = 0.008
# price KWH France, source: https://www.fournisseur-energie.com/prix-kwh/#:~:text=L%27essentiel%20sur%20les%20prix,6%20kVA%2C%20en%20option%20base.
# price Liter Water, source: https://www.cieau.com/le-metier-de-leau/prix-des-services-deau/#:~:text=le%20prix%20moyen%20des%20services,de%20120%20m3%20consomm%C3%A9s.

kwh_price = 0.2 #euros
L_price = 0.0037 #euros
