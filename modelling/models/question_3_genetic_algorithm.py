import random
import pandas as pd
import numpy as np
import geopandas as gpd
from deap import base, creator, tools

from preprocessing.pre_process_stations import *
from preprocessing.pre_process_traffic import *
from preprocessing.helping_functions import *

from features.config import *
from features.financials import *

from models.question_1 import *
from models.question_2 import *

###################################################################################
# Scenario 1: same genetic algorithm but by adding a constraint with total functions

def genetic_algorithm(
        df_stations,
        nb_stations,
        region_name,
        evaluate,
        num_generations,
        population_size
        ):
    
    # Set up the DEAP toolbox
    creator.create("FitnessMax", base.Fitness, weights=[1.0,])
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Register the genetic operators
    toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: np.random.choice(list(df_stations.index.values), size=nb_stations, replace=False),n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Create the initial population
    population = toolbox.population(n=population_size)

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):        
        ind.fitness.values = fit

    # Set up the statistics objects to evaluate results
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("argmin", np.argmin)

    avg_fitness = []
    min_fitness = []
    max_fitness = []
    arg_min_fitness = []

    # Start the evolution process
    for generation in tqdm(range(num_generations)):
    
        population = [x for x in population if type(x)!=int]

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if type(child1[0])==int:
                child1[0] = np.random.choice(list(df_stations['index'].values), size=nb_stations, replace=False)
            elif type(child2[0])==int:
                child2[0] = np.random.choice(list(df_stations['index'].values), size=nb_stations, replace=False)
            else:
                toolbox.mate(child1[0], child2[0])
            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate the new individuals
        fresh_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, fresh_individuals)
        for ind, fit in zip(fresh_individuals, fitnesses):
            ind.fitness.values = fit

        # Add the new individuals to the population
        population[:] = offspring

        # Update the hall of fame and statistics
        record = stats.compile(population)
        avg_fitness.append(record["avg"])
        min_fitness.append(record["min"])
        max_fitness.append(record["max"])
        arg_min_fitness.append(record["argmin"])

    return region_name,population,avg_fitness,min_fitness,max_fitness,arg_min_fitness

def run_ga(
        df_stations: gpd.GeoDataFrame,
        nb_stations_per_region: pd.DataFrame,
        region_name: str,
        year: int = 2030,
        num_generations: int = 30,
        population_size: int = 50):
    
    df_stations = sales(df_stations,year)
        
    nb_stations = nb_stations_per_region[nb_stations_per_region['Region']==region_name][f'Hydrogen Stations Needed {year}'].values[0]

    total_nb_stations = df_stations.shape[0]

    # Define the fitness function that takes a pandas DataFrame as input and returns a fitness score
    def fitness(X,nb_stations=nb_stations,data: pd.DataFrame=df_stations):
        index = X[0]

        if type(index)==int:
            index = np.random.choice(list(df_stations['index'].values), size=nb_stations, replace=False)

        columns_distance = [c for c in data.columns if c.startswith('distance_to_point_')==True]
        if data[data['index'].isin(list(index))].shape[0]>0:
            columns_distance_drop = [columns_distance[i] for i in index]
            data_sub = get_closer_station(data[data['index'].isin(list(index))].drop(columns=columns_distance_drop))
            # constraint make sure the closest station is closer than 120km (for trucks with autonomy of 150km)
            data_sub = data_sub[data_sub['distance_closer_station'] < 120000]
            fit = 150 - (data_sub['distance_to_closest_large_hub'].mean() + data_sub['distance_to_closest_dense_hub'].mean())/10 + data_sub['distance_closer_station'].mean() + data_sub['Revenues_day'].mean()

        else:
            fit = -10e3
        
        return fit
    
    # Define the evaluation function
    def evaluate(individual):
        return fitness(individual),
    
    region_name,population,avg_fitness,min_fitness,max_fitness,arg_min_fitness = genetic_algorithm(
        df_stations,
        nb_stations,
        region_name,
        evaluate,
        num_generations=num_generations,
        population_size=population_size
        )
    
    return region_name,population,avg_fitness,min_fitness,max_fitness,arg_min_fitness


def plot_fitness_functions(avg_fitness,region_name):
    fig = plt.figure(figsize=(10,6))
    plt.plot(np.arange(len(avg_fitness)),avg_fitness,label='mean')
    plt.xlabel('Generations')
    plt.ylabel('Average fitness function')
    plt.legend()
    plt.title(f'Evolution of fitness function for region {region_name}')
    plt.show()

def extract_restults(
        population,
        max_fitness,
        df_stations
):
    best_gen = np.argmax(max_fitness)
    best_solution = population[-1][0]
    i=-1
    while type(best_solution) == int:
        best_solution = population[i-1][0]
    chosen_stations = pd.concat([df_stations.iloc[best_solution,:11],df_stations.iloc[best_solution,-2:]],axis=1)

    return chosen_stations

###################################################################################
# Scenario 3: same genetic algorithm but by adding a constraint with total functions

def run_ga_3_3(
        df_stations: gpd.GeoDataFrame,
        nb_stations_per_region: pd.DataFrame,
        region_name: str,
        year: int = 2030,
        num_generations: int = 30,
        population_size: int = 50):
    
    df_stations = sales(df_stations,year)
        
    nb_stations = nb_stations_per_region[nb_stations_per_region['Region']==region_name][f'Hydrogen Stations Needed {year}'].values[0]

    # Define the fitness function that takes a pandas DataFrame as input and returns a fitness score
    def fitness_3_3(X,nb_stations=nb_stations,data: pd.DataFrame=df_stations):
        index = X[0]

        if type(index)==int:
            index = np.random.choice(list(df_stations['index'].values), size=nb_stations, replace=False)

        columns_distance = [c for c in data.columns if c.startswith('distance_to_point_')==True]
        if data[data['index'].isin(list(index))].shape[0]>0:
            columns_distance_drop = [columns_distance[i] for i in index]
            data_sub = get_closer_station(data[data['index'].isin(list(index))].drop(columns=columns_distance_drop))
            # constraint make sure the closest station is closer than 120km (for trucks with autonomy of 150km)
            data_sub = data_sub[data_sub['distance_closer_station'] < 120000]
            data_sub = data_sub[data_sub['distance_closer_station_total'] < 5000]
            fit = 150 - (data_sub['distance_to_closest_large_hub'].mean() + data_sub['distance_to_closest_dense_hub'].mean())/10 + data_sub['distance_closer_station'].mean() - 2*data_sub['distance_closer_station_total'].mean() + data_sub['Revenues_day'].mean()

        else:
            fit = -10e3
        
        return fit
    
    # Define the evaluation function
    def evaluate(individual):
        return fitness_3_3(individual),
    
    region_name,population,avg_fitness,min_fitness,max_fitness,arg_min_fitness = genetic_algorithm(
        df_stations,
        nb_stations,
        region_name,
        evaluate,
        num_generations=num_generations,
        population_size=population_size
        )
    
    return region_name,population,avg_fitness,min_fitness,max_fitness,arg_min_fitness

