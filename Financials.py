
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
# Hydrogen fuel price
#source: https://www.autonews.fr/green/voiture-hydrogene#:~:text=Quant%20au%20plein%20d%27hydrog%C3%A8ne,diminuer%20avec%20son%20d%C3%A9veloppement%20progressif.
# source: https://reglobal.co/cost-of-ownership-of-fuel-cell-hydrogen-trucks-in-europe/

# H2 price can vary between 10 and 15 euros depending on the source. Assuming that in our scenario we have no competitors, an assumption we will relax in part 3, we can
# set the price of H2 to 15e
H2_price_2023 = 10
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
total_demand = 1110000

def threshold(df_station_info :pd.DataFrame)-> float:
    '''
    This functions calculates the necessary amount of sales in T/day to be profitable depending on station's sizes
    '''
    df_station_info['threshold'] = df_station_info['prof_threshold']* df_station_info['storage']
    return df_station_info
# Calculate Demand



def preprocess_station(df_station: pd.DataFrame,df_tmja: pd.DataFrame) -> pd.DataFrame:
    '''
    This functions adds traffic information relative to the stations
    '''
    df_station = df_station.drop(['URL','Lavage','Paiement'],axis =1)
    df_tmja_group = df_tmja.groupby('route').agg({'percentage_traffic':'sum','TMJA_PL':'sum'})
    df_tmja_group['route'] = df_tmja_group.index
    df_tmja_group = df_tmja_group.reset_index(drop=True)
    df_station['route_id'] = df_tmja.iloc[df_station['Closer_route_by_index']]['route'].iloc[0]
    df_station_1 = pd.merge(df_station, df_tmja_group, on='route',how ='left')
    
    return df_station

def sales(df_station:pd.DataFrame,year: float)-> pd.DataFrame:
    '''
    This functions calculates the amount sold at each new stations in kg/day.
    '''
    h2_price_dict = {2023: 10, 2030: 7, 2040: 4}
    if year not in h2_price_dict:
        raise ValueError('Year can only be 2023, 2030 or 2040')
    
    #df_station['Quantity_sold_per_day(in kg)'] = total_demand * df_station['percentage_traffic'] *  (1 / df_station.groupby('route_id')['route_id'].transform('count'))
    df_station['Revenues_day'] = df_station['Quantity_sold_per_day(in kg)']  * h2_price_dict[year]
    return df_station

def station_type(df_station:pd.DataFrame, df_station_info: pd.DataFrame)-> pd.DataFrame:
    '''
    This function derives the station size depending on quantity sold and the profitability thresholds
    '''
    info = threshold(df_station_info)
    small_prof_threshold, medium_prof_threshold, large_prof_threshold = info['threshold']
    df_station['Quantity_sold_per_year(in kg)']= df_station['Quantity_sold_per_day(in kg)']*365

    df_station['not_prof'] = (df_station['Quantity_sold_per_year(in kg)']/1000< small_prof_threshold).astype(int)

    df_station['small_station'] = ((df_station['Quantity_sold_per_year(in kg)']/1000 >= small_prof_threshold) &
                                   (df_station['Quantity_sold_per_year(in kg)']/1000 < medium_prof_threshold)).astype(int)
    
    df_station['medium_station'] = ((df_station['Quantity_sold_per_year(in kg)']/1000 >= medium_prof_threshold) &
                                    (df_station['Quantity_sold_per_year(in kg)']/1000 < large_prof_threshold)).astype(int)
    
    df_station['large_station'] = (df_station['Quantity_sold_per_year(in kg)']/1000>= large_prof_threshold).astype(int)
    
    df_station['station_type'] = df_station.apply(lambda row: 
    'not profitable' if row['not_prof'] == 1 
    else 'small' if row['small_station'] == 1 
    else 'medium' if row['medium_station'] == 1 
    else 'large' if row['large_station'] == 1 
    else 'unknown', axis=1)
    return df_station

def financials(df_station:pd.DataFrame, df_station_info: pd.DataFrame,year: float)-> pd.DataFrame:
    '''
    This function provides an overview of the P&L for each station
    '''
    df_station = station_type(df_station, df_station_info)
    df_station = sales(df_station,year)
    df_station['Revenues'] = df_station['Revenues_day'] * 365
    df_station['EBITDA'] = df_station['Revenues']- df_station['station_type'].map(df_station_info.set_index('station_type')['opex']*1000000)
    df_station['Opex'] = df_station['Revenues'] - df_station['EBITDA']

    df_station_info['yearly_depreciation'] = df_station_info['capex'] * df_station_info['depreciation']*1000000 
    df_station['EBIT'] = df_station['Revenues']- df_station['station_type'].map(df_station_info.set_index('station_type')['yearly_depreciation'])
    df_station['depreciation'] = df_station['EBITDA'] - df_station['EBIT']

    fin = ['Revenues','EBITDA','EBIT','depreciation','Opex']
    df_station[fin] = df_station[fin].fillna(0)
    return df_station

def capex(df_station:pd.DataFrame, df_station_info: pd.DataFrame)-> pd.DataFrame:
    '''
    This function provides an overview of the capex needs for each station
    '''
    df_station['CAPEX'] = df_station['Revenues']- df_station['station_type'].map(df_station_info.set_index('station_type')['capex']*1000000)
    return df_station


def financial_summary(df_station:pd.DataFrame,df_station_info: pd.DataFrame,year: float) -> pd.DataFrame:
    '''
    This function gives the consolidated financials of the deployment of all the stations
    '''
    
    df_station = financials(df_station, df_station_info,2030)
    df_station = capex(df_station, df_station_info)

    summary_df = pd.pivot_table(df_station, values=['CAPEX','Revenues','Opex', 'EBITDA', 'depreciation', 'EBIT'], 
                            index=[],
                            columns=['station_type'], 
                            aggfunc=np.sum, fill_value=0)
    summary_df.loc['Total'] = summary_df.sum()
    
    return summary_df



def deployment_dates(df_station:pd.DataFrame,year_start: float,year_end: float)-> pd.DataFrame:
    '''
    This function assigns the year of deployment for each functions depending on its revenue
    '''
    
    sorted_df = df_station.sort_values(by='Revenues', ascending=False)
    n_stations = len(sorted_df)
    n_years = year_end - year_start + 1
    n_stations_per_group = math.ceil(n_stations / n_years)
    groups = []
    for i in range(n_years):
        start = i * n_stations_per_group
        end = (i+1) * n_stations_per_group
        group = sorted_df.iloc[start:end]
        groups.append(group)

    installation_years = [year_start + i for i in range(n_years)]
    installation_dates = {}
    for group, year in zip(groups, installation_years):
        for url in group['geometry']:
            installation_dates[url] = year

    date_df = pd.DataFrame({'geometry': df_station['geometry'], 'date_installation': [installation_dates.get(s, year_end) for s in df_station['geometry']]})

    # Merge the installation dates dataframe with the original dataframe
    merged_df = pd.merge(df_station, date_df, on='geometry')

    return merged_df

def deployment_financials(df_station:pd.DataFrame,year_start: float,year_end: float)-> pd.DataFrame:
 

    df_station = deployment_dates(df_station,year_start,year_end)
    years = sorted(df_station['date_installation'].unique())

    data = {}

    for i, year in enumerate(years):
        prev_years = years[:i+1]
        cumulative_sales = df_station[df_station['date_installation'].isin(prev_years)]['Quantity_sold_per_year(in kg)'].sum()
        cumulative_revenues = df_station[df_station['date_installation'].isin(prev_years)]['Revenues'].sum()
        cumulative_opex = df_station[df_station['date_installation'].isin(prev_years)]['Opex'].sum()
        cumulative_ebitda = df_station[df_station['date_installation'].isin(prev_years)]['EBITDA'].sum()
        cumulative_depreciation = df_station[df_station['date_installation'].isin(prev_years)]['depreciation'].sum()
        cumulative_ebit = df_station[df_station['date_installation'].isin(prev_years)]['EBIT'].sum()
        small_sation = df_station[df_station['date_installation'].isin(prev_years)]['small_station'].sum()
        medium_sation = df_station[df_station['date_installation'].isin(prev_years)]['medium_station'].sum()
        large_sation = df_station[df_station['date_installation'].isin(prev_years)]['large_station'].sum()
        
        data[year] = [cumulative_sales,cumulative_revenues,cumulative_opex, cumulative_ebitda,cumulative_depreciation,cumulative_ebit,small_sation,medium_sation,large_sation]

    # Create a dataframe from the data dictionary, with the years as columns and the financial metrics as rows
    df_cumulative_financials = pd.DataFrame.from_dict(data, orient='index', columns=['sales','revenues','opex' , 'ebitda','depreciation','ebit','small_station','medium_station','large_station'])

    return df_cumulative_financials

def sceanrio_select(df_deployments: pd.DataFrame, percentage: float) -> pd.DataFrame:
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

    data = {}


    for i, year in enumerate(years):
        prev_years = years[:i+1]
        cumulative_sales = df_station[df_station['date_installation'].isin(prev_years)]['Quantity_sold_per_year(in kg)'].sum()
        cumulative_revenues = df_station[df_station['date_installation'].isin(prev_years)]['Revenues'].sum()
        cumulative_opex = df_station[df_station['date_installation'].isin(prev_years)]['Opex'].sum()
        cumulative_ebitda = df_station[df_station['date_installation'].isin(prev_years)]['EBITDA'].sum()
        cumulative_depreciation = df_station[df_station['date_installation'].isin(prev_years)]['depreciation'].sum()
        cumulative_ebit = df_station[df_station['date_installation'].isin(prev_years)]['EBIT'].sum()
        small_sation = df_station[df_station['date_installation'].isin(prev_years)]['small_station'].sum()
        medium_sation = df_station[df_station['date_installation'].isin(prev_years)]['medium_station'].sum()
        large_sation = df_station[df_station['date_installation'].isin(prev_years)]['large_station'].sum()
        
        data[year] = [cumulative_sales,cumulative_revenues,cumulative_opex, cumulative_ebitda,cumulative_depreciation,cumulative_ebit,small_sation,medium_sation,large_sation]

    df_cumulative_financials = pd.DataFrame.from_dict(data, orient='index', columns=['sales','revenues','opex' , 'ebitda','depreciation','ebit','small_station','medium_station','large_station'])

    return df_cumulative_financials


results = pd.read_csv(r"C:\Users\cesar\Dropbox\My PC (LAPTOP-GU3S2J8B)\Downloads\results.csv")
results  = financials(results, df_station_info,2030)
results = deployment_dates(results,2030,2045)
#b = sales(results,2030)
#b = financial_summary(results,df_station_info,2040)

#print(b)