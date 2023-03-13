# Data Analytics for Sustainability Hydrogen Trucks Charging Stations

## Description

The market for hydrogen as a sustainable energy source has been growing rapidly in recent years, with increasing demand for low-emission transportation solutions. France, in particular, has set ambitious targets for reducing carbon emissions and has identified hydrogen as a key part of its strategy to achieve these goals. The country aims to have 6.5 million low-emission vehicles on the road by 2030, with a significant portion powered by hydrogen fuel cells.

However, to support the widespread adoption of hydrogen-powered vehicles, a robust infrastructure of hydrogen refueling stations is needed. This is where our project comes in. In this project, we aim to size and optimize the network of hydrogen truck charging stations in France, taking into account various factors such as the forecasted number of hydrogen trucks in France and Europe, truck autonomy, driver regulations, and the motorway network in France.

Our project is divided into four parts, each focusing on a specific aspect of the challenge. In the first part, we will analyze the forecasted number of hydrogen trucks in France and Europe and use this information to size the network of hydrogen truck charging stations, including the number of stations and their distribution across different regions of France.

In the second part, we will develop models to identify the exact locations where the hydrogen stations should be implemented. These models will consider factors such as truck traffic per transit axis, the localization of logistic hubs, and the cost of deployment and operations per station.

In the third part, we will apply these models to France and develop a deployment roadmap for 2030-2040, taking into account three competitive scenarios. These scenarios include a single network in France, two players entering simultaneously, and one player entering after an incumbent transforming its oil stations network to hydrogen.

Finally, in the fourth part, we will identify the optimal locations for hydrogen production infrastructure, taking into account factors such as production and transport costs. The output of our project will include geographic repartition models, a deployment roadmap, and the coordinates and types of each station and production plant.


## Getting started with the repository
​
To ensure that all libraries are installed pip install the requirements file:
 
```pip install -r requirements.txt```

You should be at the source of the repository structure (ie. sustainability-challenge) when running the command.

Our repository is structured in the following way:
​
```
|endlessline_eleven
   |--data
   |--notebooks
   |-----features
   |--------config.py
   |--------question_1.py
   |--------question2.py
   |--------financials_part2.py
   |-----preprocessing
   |--------helping_functions.py
   |--------pre-process_stations.py
   |--------pre-process_traffic.py
   |-----Notebook_genetic_algorithm
   |-----Notebook_part_1
   |--README.md
   |--requirements.txt
   |--.gitignore
```

### data 

To properly get the data, one has to download it locally on his/her computer, unzip and put the folder in the repository. One should have a folder like data/ with 5 different elements:
- E-tmja2019-shp
- F-aire-logistique-donees-detaillees
- new-coordinates
- regions-20180101-shp
- E-tmja-2019.csv

### Notebooks

#### features

1) config.py
<<<<<<< HEAD
=======

>>>>>>> 35be8ed0f12536271e0c682d9cec9c692aa9f0f6
This python script is used to combine several classes with parameters and results for Part 1 and financial considerations.

The *Parameters_part_1* class defines various parameters for the sizing of the network of hydrogen truck charging stations in France in 2030 and 2040, including the average speed of the trucks, maximum daily drive time, autonomy of different types of trucks, tank sizes, capacity of stations, and the total number and percentage of each type of truck.

The *Results_part_1* class includes the total demand for hydrogen fuel in France in 2030 and 2040 based on the input parameters.

The *Parameters_financial* class defines the hydrogen fuel price for 2023, 2030, and 2040, assuming no competitors in the market.

The refueling parameter specifies the time it takes to refuel a hydrogen truck, while *Parameters_filter_stations* includes the maximum distance from a road or logistic hub for a station to be considered for implementation.

2) questions_1.py
<<<<<<< HEAD
=======

>>>>>>> 35be8ed0f12536271e0c682d9cec9c692aa9f0f6
This python script contains the code used to creaste functions to preprocess traffic data and calculate the number of hydrogen stations needed for a given scenario year based on the percentage of three brands of trucks. Here is a brief summary of each function:

- grouped_region: This function takes a pandas DataFrame and a shape file as inputs and preprocesses the DataFrame by grouping it by region, summing the total distance of roads, and adding shape file information. It returns a new pandas DataFrame.

- distance_road_region: This function takes a path to an Excel file and loads the DataFrame from the "REG" sheet. It then sums the total distance of National and Autoroute roads and returns a pandas DataFrame.

- create_part1_data: This function takes a path to an Excel file and a previously processed region DataFrame as inputs. It merges the two DataFrames to have the AVG TMJA_PL and calculates the percentage of traffic in each region. It returns a pandas DataFrame.

- calculate_hydrogen_stations: This function takes the input dataframe with the traffic data for each region and the percentage of three brands of trucks as inputs. It calculates the number of hydrogen stations needed based on several assumptions and constraints and returns a pandas DataFrame with the region and the number of hydrogen stations needed.

3) question_2.py
<<<<<<< HEAD
=======

>>>>>>> 35be8ed0f12536271e0c682d9cec9c692aa9f0f6
This python script contains the code used for creating a model to predict Hydrogen demand. The code includes various functions to preprocess data, create new features and calculate distances between points.

- indicate_crs: sets the coordinate reference system (CRS) of a GeoDataFrame.
- fix_stations: fixes some formatting issues in the Coordinates column of a DataFrame.
- new_coordinates_creation: creates new coordinates along a route by splitting the route into equal length segments and interpolating new points along each segment.
station_distances_all: adds columns of the distances between each station.
- get_closer_station: creates a column distance_closer_station with the smaller distance between each station.
distance_to_hub: adds four columns regarding the distances to the closest hubs for each point.

4) financials_part_2.py
<<<<<<< HEAD
=======

>>>>>>> 35be8ed0f12536271e0c682d9cec9c692aa9f0f6
This python script contains the code used to simulate the profitability of a network of hydrogen refueling stations for heavy-duty vehicles. The simulation takes into account the cost of building and operating the stations, the traffic flow in the surrounding area, and the price of hydrogen fuel. The output is a recommendation on the optimal number and size of stations to deploy in a given region to achieve profitability.

The simulation is driven by three user-defined parameters: the year of deployment (2023, 2030, or 2040), the total daily demand for hydrogen fuel in the region, and the profitability thresholds for different station sizes. The code also incorporates real-world data on traffic flows, construction times, and station costs to make the simulation as accurate as possible.

Overall, this code provides a powerful tool for stakeholders interested in deploying a network of hydrogen refueling stations for heavy-duty vehicles. With its flexibility and accuracy, it can help decision-makers evaluate different deployment scenarios and make informed choices about how to build a profitable network of stations.

#### preprocessing

1) helping_functions.py
<<<<<<< HEAD
=======

>>>>>>> 35be8ed0f12536271e0c682d9cec9c692aa9f0f6
This python script contains several functions for working with geographic data in Python using the GeoPandas library.

- add_lat_lon_columns(df) adds new columns to a dataframe containing latitude and longitude values based on the 'xD', 'yD', 'xF', and 'yF' columns, which are in Lambert-93 projection.

- The second function indicate_crs(shp_file: gpd.geodataframe.GeoDataFrame, epsg:str) sets the coordinate reference system (CRS) of a GeoDataFrame.

- convert_long_lat_to_easting_northing(df, initial_epsg='epsg:4326', target_epsg='epsg:2154') converts latitude and longitude coordinates to easting and northing coordinates using the given input and output projections.

- convert_str_geometry_to_geometry_geometry(df) converts a geometry column with geometry shapes written as a string to a geometry shapes column.

2) pre_proccess_stations.py
<<<<<<< HEAD
=======

>>>>>>> 35be8ed0f12536271e0c682d9cec9c692aa9f0f6
This python script contains several functions that are used for processing geospatial data related to stations.

-add_lat_lon_columns function takes a dataframe df with columns xD, yD, xF, and yF, which are in Lambert-93 projection, and adds new columns lonD, latD, lonF, and latF with corresponding latitude and longitude values in WGS84 projection (EPSG:4326).

- indicate_crs function takes a GeoDataFrame shp_file and an EPSG code epsg and sets the CRS of the GeoDataFrame to the specified EPSG code.

- add_region_column function takes a dataframe df with a column dep, which contains department codes, and adds a new column region with the corresponding region names. The mapping of department codes to region names is defined in a dictionary within the function.

3) pre_process_traffic.py
<<<<<<< HEAD
=======

>>>>>>> 35be8ed0f12536271e0c682d9cec9c692aa9f0f6
This python script defines two functions to preprocess traffic data.
- preprocess_data, takes a Pandas DataFrame as input and applies some transformations to it. It fixes an issue with the 'ratio_PL' column where some values are greater than 40, by dividing those values by 10. It then calculates a new column called 'TMJA_PL', which is the product of the 'TMJA' and 'ratio_PL' columns divided by 100, rounded to 2 decimal places. The function then calculates the sum of the 'TMJA_PL' column and uses it to calculate the percentage of traffic in each region, which is saved in a new column called 'percentage_traffic'. The preprocessed DataFrame is then returned.

- fix_tmja, takes a Pandas DataFrame as input and applies some transformations to it. It fixes the 'ratio_PL' column by dividing values greater than 40 by 10. It then adds latitude and longitude columns using a helper function called add_lat_lon_columns, and adds 'region' and 'departement' columns based on the latitude and longitude using the reverse_geocoder library. The resulting DataFrame is returned.

#### Notebook Part 1

This notebook is used for all the analysis related to the part 1 of the project- sizing the hydrogen truck market in France and in Europe over the next couple of years.

#### Notebook Genetic Algorithm

This notebook cotains all of the analysis related to the part 2 of the project. using Genetic Algotithms we identified the exact locations where the hydrogen stations should be implemented. 

## Contacts LinkedIn 
​
If you have any feedback, please reach out to us on LinkedIn!
​
- [Cesar Bareau](https://www.linkedin.com/in/cesar-bareau-457089152/)
- [Augustin de la Brosse](https://www.linkedin.com/in/augustin-de-la-brosse/)
- [Alexandra Sacha Giraud](https://www.linkedin.com/in/alexandra-sacha-giraud-06981813b/)
- [Camille Keisser](https://www.linkedin.com/in/camille-keisser-6074a8173/)
- [Charlotte Simon](https://www.linkedin.com/in/charlottesmn/)
