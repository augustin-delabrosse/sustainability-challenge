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


def clustering_of_stations(results:pd.core.frame.DataFrame, k_max:int=40) -> dict:
    """
    Perform k-means clustering on the given data and return the results.
    Args:
    - results: a pandas DataFrame containing the data to be clustered, with columns 'easting' and 'northing'.
    - k_max: an integer specifying the maximum number of clusters to consider.
    Returns:
    A dictionary containing the clustering results for each value of k between 2 and k_max, inclusive.
    The dictionary has keys representing the number of clusters (k), and values representing the clustering results.
    The clustering results are themselves dictionaries with the following keys:
    - 'inertia': the sum of squared distances of samples to their closest cluster center.
    - 'silhouette': a measure of how similar an object is to its own cluster compared to other clusters.
    - 'labels': the labels of each point in the input data after clustering.
    - 'centroids': the coordinates of the cluster centers.
    """

    results_to_cluster = results[['easting', "northing"]]
    result_of_clustering = {}
    K = range(2, k_max)
    for k in tqdm(K):
        result_of_clustering[k] = {}
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(results_to_cluster)
        result_of_clustering[k]['inertia'] = kmeanModel.inertia_
        result_of_clustering[k]['silhouette'] = silhouette_score(results_to_cluster, kmeanModel.labels_)
        result_of_clustering[k]['labels'] = kmeanModel.labels_
        result_of_clustering[k]['centroids'] = kmeanModel.cluster_centers_
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,16))

    ax1.plot(K, [i["inertia"] for i in result_of_clustering.values()], 'bx-')

    ax2.plot(K, [i["silhouette"] for i in result_of_clustering.values()], 'bx-')

    ax1.set_title('Finding the optimal k', {'fontsize':30})
    ax1.set_ylabel('Inertia')
    ax2.set_ylabel('Silhouette score')

    plt.show()

    return result_of_clustering 

def plot_clusters_(productions_sites, routes, stations, cmap:str="gist_ncar"):
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
    shp_production_sites = gpd.GeoDataFrame({'geometry': productions_sites.geometry, 
                                         'nom': [f'{i+1}th production site' for i in range(productions_sites.shape[0])]}, 
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