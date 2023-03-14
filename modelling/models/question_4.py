import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pyproj 
from tqdm import tqdm
import reverse_geocoder as rg
from deap import base, creator, tools
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from shapely.geometry import Point
from shapely.geometry import LineString

from preprocessing.pre_process_stations import *
from preprocessing.pre_process_traffic import *
from preprocessing.helping_functions import *

from features.config import *
from features.question_2_financials import *

from models.question_1 import *
from models.question_2 import *
from models.genetic_algorithm_part3_1 import *


def size_of_production_sites_by_cluster(df:pd.core.frame.DataFrame):
    """
    Calculates the number of production sites required to meet the hydrogen demand
    for each cluster based on the cluster size and the daily hydrogen demand.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing information on the clusters, including their size
        and hydrogen demand.

    Returns:
    --------
    productions_site_by_cluster : dict
        A dictionary where the keys are the cluster indices and the values are
        dictionaries containing the number of small and large production sites
        required to meet the hydrogen demand for the cluster.
    """
    
    factory_info = {
        "MW": [5, 100],
        'station_type': ['small', 'large'],
        'capex': [20, 120],
        'depreciation': [0.15,0.15], 
        'opex': [0.03 * 20, 0.03 * 120],
        'Power_usage': [55, 50],  # Wh/kgH2 
        'water_consumption': [10, 10], #L/kgH2
        'h2_production_per_day': [2.181, 48]
        }
    
    df['tpd'] = df.taille.apply(lambda x: 4 if x=="large" else (2 if x=="medium" else 1))

    t_per_day_by_cluster = {}
    for i in range(df.cluster.unique().max()+1):
        cluster = df[df['cluster'] == i]
        demand = cluster.tpd.sum()
        t_per_day_by_cluster[i] = demand


    productions_site_by_cluster = {}
    for i, t in t_per_day_by_cluster.items():
        # print(f'cluster {i}', t)
        productions_site_by_cluster[i] = {}
        n=1
        while n*factory_info['h2_production_per_day'][-1] < t:
            n+=1
            # print(n, n*factory_info['h2_production_per_day'][-1])
        # print('')

        base_t = (n-1)*48
        if np.ceil((t-base_t)/2.181)*factory_info['opex'][0] > np.ceil((t-base_t)/48)*factory_info['opex'][-1]:
            productions_site_by_cluster[i]['small'] = 0 # (n-1) + np.ceil((t-base_t)/2.181)
            productions_site_by_cluster[i]['large'] = n
        else:
            productions_site_by_cluster[i]['small'] = np.ceil((t-base_t)/2.181)
            productions_site_by_cluster[i]['large'] = n-1
    
    return productions_site_by_cluster


def production_sites_localization(productions_site_by_cluster: dict, clusters):
    """
    Compute the locations of hydrogen production sites given the number of small and large production sites
    in each cluster and the clusters' geometries.

    Args:
        productions_site_by_cluster (dict): A dictionary containing the number of small and large hydrogen 
            production sites in each cluster.
        clusters (gpd.GeoDataFrame): A GeoDataFrame representing the clusters in which the hydrogen production 
            sites are located, with one row per cluster and a geometry column containing the polygon representing 
            the cluster.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the locations and names of the hydrogen production 
            sites, with one row per site.
    """

    productions_site_locations = {}
    for i in productions_site_by_cluster.keys():
        n_small_large = [*productions_site_by_cluster[i].values()]
        if np.sum(n_small_large) == 1:
            productions_site_locations[i] = [clusters.at[i, 'geometry']]
        else:
            new_geom = clusters.at[i, 'geometry'].buffer(30000)
            perimeter_length = new_geom.length
            n_splits = np.sum(n_small_large)
            distance_between_points = perimeter_length / n_splits
            cumulative_distance = 0
            cluster_points = []
            for idx, vertex in enumerate(new_geom.exterior.coords[:-1]):
                if idx == 0:
                    cluster_points.append(Point(vertex))
                    continue
                previous_vertex = new_geom.exterior.coords[idx-1]
                segment_length = Point(vertex).distance(Point(previous_vertex))
                cumulative_distance += segment_length
                if cumulative_distance >= distance_between_points:
                    distance_from_previous = cumulative_distance - distance_between_points
                    interpolation_factor = distance_from_previous / segment_length
                    new_point = Point(
                        previous_vertex[0] + interpolation_factor * (vertex[0] - previous_vertex[0]),
                        previous_vertex[1] + interpolation_factor * (vertex[1] - previous_vertex[1])
                    )
                    cluster_points.append(new_point)
                    cumulative_distance = distance_from_previous

            productions_site_locations[i] = cluster_points
    
    points = [item for sublist in [*productions_site_locations.values()] for item in sublist]
    sizes = []
    for value in productions_site_by_cluster.values():
        for idx, i in enumerate(value.values()):
            if idx == 0:
                for j in range(int(i)):
                    sizes.append('small')
            else:
                for j in range(int(i)):
                    sizes.append('large')
                    
    shp_production_sites = gpd.GeoDataFrame({'geometry': points, 
                                             'nom': [f'{i} H2 production plant' for i in sizes]}, 
                                            crs="epsg:2154")
    
    exploration = shp_production_sites.explore(column='nom', cmap="rainbow")
    
    return shp_production_sites, exploration