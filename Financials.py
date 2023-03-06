
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

# Hydrogen fuel price
#source: https://www.autonews.fr/green/voiture-hydrogene#:~:text=Quant%20au%20plein%20d%27hydrog%C3%A8ne,diminuer%20avec%20son%20d%C3%A9veloppement%20progressif.
# source: https://reglobal.co/cost-of-ownership-of-fuel-cell-hydrogen-trucks-in-europe/

# H2 price can vary between 10 and 15 euros depending on the source. Assuming that in our scenario we have no competitors, an assumption we will relax in part 3, we can
# set the price of H2 to 15e
H2_price_2023 = 15
H2_price_2030 = 7 
H2_price_2040 = 4 

# Refueling time
#source : https://www.pcmag.com/news/volvos-hydrogen-powered-truck-can-travel-620-miles-before-needing-to-refuel
refueling = 10 #min


station_info = {
    'station_type': ['small', 'medium', 'large'],
    'capex': [3, 5, 8],
    'depreciation': [0.15,0.15,0.15],
    'opex': [0.1 * 3, 0.08 * 3, 0.07 * 3],
    'storage': [2, 3, 4],
    'construction_time': [1, 1, 1],
    'footprint': [650, 900, 1200],
    'prof_threshold': [0.9, 0.8, 0.6]
}

df_station_info = pd.DataFrame(station_info).reset_index(drop=True)
total_demand = 122100

def threshold(df:pd.DataFrame)-> float:
    '''
    This functions calculates the necessary amount of sales in T/day to be profitable depending on station's sizes
    '''
    small_prof_threshold = df.loc[df['station_type'] == 'small', 'prof_threshold'].iloc[0] * df.loc[df['station_type'] == 'small', 'storage'].iloc[0]
    medium_prof_threshold = df.loc[df['station_type'] == 'medium', 'prof_threshold'].iloc[0] * df.loc[df['station_type'] == 'medium', 'storage'].iloc[0]
    large_prof_threshold = df.loc[df['station_type'] == 'large', 'prof_threshold'].iloc[0] * df.loc[df['station_type'] == 'large', 'storage'].iloc[0]
    return small_prof_threshold, medium_prof_threshold, large_prof_threshold
# Calculate Demand

def sales(df:pd.DataFrame,year: float)-> pd.DataFrame:
    '''
    This functions calculates the amount sold at each new stations in kg/day.
    '''
    h2_price_dict = {2023: 15, 2030: 7, 2040: 4}
    
    if year not in h2_price_dict:
        raise ValueError('Year can only be 2023, 2030 or 2040')
    
    df['Quantity_sold_per_day(in kg)'] = total_demand * df['TMJA_PL'] * df['percentage_traffic'] *  (1 / df.groupby('route_id')['route_id'].transform('count'))
    df['Revenues'] = df['Quantity_sold_per_day(in kg)']  * h2_price_dict
    return df

def station_type(df:pd.DataFrame, df_info: pd.DataFrame)-> pd.DataFrame:
    '''
    This function allocates the size of station depeding on revenues
    '''

    small_prof_threshold, medium_prof_threshold, large_prof_threshold = threshold(df_info)
    conditions = [    (df['small_station'] == 1),
    (df['medium_station'] == 1),
    (df['large_station'] == 1)]

    values = ['small', 'medium', 'large']


    df['small_station'] = np.where(df['Quantity_sold_per_day(in kg)'] * 0.365 >  small_prof_threshold, 1,0)
    df['medium_station'] = np.where(df['Quantity_sold_per_day(in kg)'] * 0.365 >  medium_prof_threshold, 1,0)
    df['large_station'] = np.where(df['Quantity_sold_per_day(in kg)'] * 0.365 >  large_prof_threshold, 1,0)

    df['station_type'] = np.select(conditions, values, default='other')
    return df

def financials(df:pd.DataFrame, df_info: pd.DataFrame)-> pd.DataFrame:
    '''
    This function provides an overview of the P&L for each station
    '''
    df['EBITDA'] = df['Revenues']- df['station_type'].map(df_station_info.set_index('station_type')['opex'])
    df_info['yearly_depreciation'] = df_info['capex'] * df_info ['depreciation']
    df['EBIT'] = df['Revenues']- df['station_type'].map(df_station_info.set_index('station_type')['opex'])

def financial_summary(df:pd.DataFrame) -> pd.DataFrame:
    '''
    This functions give the consolidated financials of the deployment of all the stations
    '''
    total_revenues, total_EBITDA = df['Revenues'].sum(), df['EBITDA'].sum()
    total_opex = total_revenues - total_EBITDA
    total_EBIT = df['EBIT'].sum()
    summary_df = pd.DataFrame({
        'Total Revenues': [total_revenues],
        'Total Opex': [total_opex],
        'Total EBITDA': [total_EBITDA],
        'Total EBIT': [total_EBIT]
    })
    return summary_df