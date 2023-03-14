import geopandas as gpd
import pandas as pd
import numpy as np

import pyproj
from pyproj import Proj, transform
import reverse_geocoder as rg
from shapely import Point

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')


def add_lat_lon_columns(df):
    """
    Adds new columns 'lonD', 'latD', 'lonF', 'latF' to the tmja dataframe with
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
    in_proj = pyproj.Proj(init=initial_epsg)  
    out_proj = pyproj.Proj(init=target_epsg)  

    long = df['Coordinates'].apply(lambda x: x.split(',')[1]).values.astype(float)
    lat = df['Coordinates'].apply(lambda x: x.split(',')[0]).values.astype(float)

    # Convert start coordinates to lat-long
    long_transformed, lat_transformed = pyproj.transform(in_proj, out_proj, long, lat)
    
    df.geometry = gpd.points_from_xy(x=long_transformed, 
                                     y=lat_transformed, crs=target_epsg)
    
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

def clustering_of_stations(results:pd.core.frame.DataFrame, k_max:int=40, seed:int=5) -> dict:
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
        kmeanModel = KMeans(n_clusters=k, random_state=seed)
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