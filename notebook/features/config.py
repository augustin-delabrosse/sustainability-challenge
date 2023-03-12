# For Part 1
class Parameters_part_1():
    avg_speed = 80  # km/h
    max_drive_time_daily = 9  # hours
    break_time = 0.33 # hours
    autonomy_daimler = 1000  # km
    autonomy_nikola = 400  # km
    autonomy_daf = 150  # km
    tank_size_daimler = 80
    tank_size_nikola = 32
    tank_size_daf = 30
    capacity_stations = 3*1000

    total_trucks = 10000
    daimler_perc = 0.25
    nikola_perc = 0.25
    daf_perc = 0.50

# Result part 1
class Results_part_1():
    total_demand = 122100

# Hydrogen fuel price
#source: https://www.autonews.fr/green/voiture-hydrogene#:~:text=Quant%20au%20plein%20d%27hydrog%C3%A8ne,diminuer%20avec%20son%20d%C3%A9veloppement%20progressif.
# source: https://reglobal.co/cost-of-ownership-of-fuel-cell-hydrogen-trucks-in-europe/

# H2 price can vary between 10 and 15 euros depending on the source. Assuming that in our scenario we have no competitors, an assumption we will relax in part 3, we can
# set the price of H2 to 15e
class Parameters_financial():
    H2_price_2023 = 15
    H2_price_2030 = 7 
    H2_price_2040 = 4 

# Refueling time
#source : https://www.pcmag.com/news/volvos-hydrogen-powered-truck-can-travel-620-miles-before-needing-to-refuel
refueling = 10 #min

class Parameters_filter_stations():
    max_dist_road = 1000
    max_dist_dense_hub = 50000
    max_dist_hub = 1000