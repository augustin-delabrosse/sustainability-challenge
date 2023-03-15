import pandas as pd
import geopandas as gpd

from preprocessing.pre_process_stations import *
from preprocessing.pre_process_traffic import *
from preprocessing.helping_functions import *

from features.config import *
from features.financials import *

from models.question_1 import *
from models.question_2 import *
from models.question_3_genetic_algorithm import *


######################################################################
# Scenario 1 for part 3: Load the data needed

def load_data():
    ##### Load new coordinates
    df_new_points = gpd.read_file(config.PATH+'new_coordinates/new_coordinates.shp')

    ##### Load hub data 
    df_hub_dense = gpd.read_file(config.PATH+'F-aire-logistiques-donnees-detaillees/Aires_logistiques_denses.shp')
    df_hub_enlarged = gpd.read_file(config.PATH+'F-aire-logistiques-donnees-detaillees/Aires_logistiques_elargies.shp')

    ##### Load traffic data
    df_traffic = gpd.read_file(config.PATH+'E-tmja2019-shp/TMJA2019.shp')
    df_traffic = preprocess_data(df_traffic)
    df_traffic = fix_tmja(df_traffic)

    df_new_points = distance_to_hub(df_hub_dense, df_hub_enlarged, df_new_points)

    df_traffic = df_traffic.groupby('route')[['TMJA_PL','percentage_traffic']].sum().reset_index(drop=False)
    df_new_points = df_new_points.merge(df_traffic, how='left', on='route')
    df_new_points = station_distances_all(df_new_points)

    data_region = gpd.read_file(config.PATH+'regions-20180101-shp/regions-20180101.shp')
    df_new_points = create_region_columns(df_new_points,data_region)

    return df_new_points


######################################################################
# Scenario 3: Load the red player H2 stations from data I.

def load_red_player_stations():
    '''
    Load the red player stations with H2 concersion
    '''
    df_stations_red_player = pd.read_csv(config.PATH + 'I-Données_de_stations_improved.csv')
    df_stations_red_player = df_stations_red_player[df_stations_red_player['H2 Conversion']==1]
    df_stations_red_player = df_stations_red_player.groupby(['URL'])[['nom_region','geometry']].first()
    df_stations_red_player = df_stations_red_player.reset_index(drop=False)
    df_stations_red_player = convert_str_geometry_to_geometry_geometry(df_stations_red_player)
    df_stations_red_player = df_stations_red_player.reset_index(drop=False)

    return df_stations_red_player