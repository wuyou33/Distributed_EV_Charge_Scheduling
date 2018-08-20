# ''' Decentralized Charging Control of Large Populations of Plug-in Electric Vehicles '''
# ''' Z. Ma, D. S. Callaway and I. A. Hiskens, "Decentralized Charging Control of Large Populations of Plug-in
#     Electric Vehicles," in IEEE Transactions on Control Systems Technology, vol. 21, no. 1, pp. 67-78, Jan. 2013 '''
# ''' URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6081962&isnumber=6384889 '''

# ''' System '''
# Objective : Minimizing charging cost while achieving Valley filling
# Control Architecture : Decentralized Type(2)

# ''' How to execute '''
# Go to line 33, and define the number of EVs (N)
# Run the script

from cvxpy import Minimize, Variable, sum_squares, Problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import csv
import random
import time
from . import constants

# Time Horizon
# -------------

# Real
# 12, 13, 14, 15, 16, 17, ..., 23,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  11
# Following values are shown in the ev.csv as well as in the calculation
# 0 ,  1,  2,  3,  4,  5, ..., 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 , 23

# Number of households
H = constants.HOUSES

# Number of EVs
N = int(H * constants.PENETRATION * constants.EVS_PER_HOUSEHOLD)

# Time horizon
T = constants.TIME_HORIZON


#  ---- Start Functions ---- #

def generate_random_value(mean, standard_deviation):
    """ Generate a random value in a normal distribution.
    Keyword arguments:
        mean : Mean of the distribution
        standard_deviation : Standard deviation of the Gaussian (normal) distribution
    Returns:
        null
    """
    norm1 = sp.norm(loc=mean, scale=standard_deviation)
    generated_number = round(norm1.rvs(), 2)
    # To prevent negative values
    if generated_number < 0:
        generated_number = (-1) * generated_number
    return generated_number


def generate_ev_data(no_of_records):
    """ Generate an EV record (Maximum power,Plug-in time,Plug-out time,SOC at arrival,SOC at departure,Battery capacity,
    Charging efficiency)
    and write to the csv file.
    Keyword arguments:
        no_of_records : Number of EV records to generate
    Returns:
        null
    """

    # Values taken from M. Yilmaz and P. T. Krein, "Review of Battery Charger Topologies, Charging Power Levels, and " \
    # "Infrastructure for Plug-In Electric and Hybrid Vehicles," in IEEE Transactions on Power Electronics, vol. 28,
    # no. 5, pp. 2151-2169, May 2013
    # Toyota Prius PHEV: (3.8kW, 4.4kWh)
    # Chevrolet PHEV: (3.8kW, 16kWh)
    # Mitsubishi i-MiEV EV: (3kW, 16kWh)
    # Nissan Leaf EV: (3.3kW, 24kWh)

    # EV Id
    ev_id = [i for i in range(1, no_of_records + 1)]

    # 2) Maximum charging rate of EVs
    maximum_power = []
    for n in range(no_of_records):
        maximum_power.append(random.choice([3.8, 3.8, 3, 3.3]))

    # Plug-in times and Plug-out times of EVs are assumed to be Gaussian.
    # It is assumed that EVs leave home at 7 am (7 -> 19) with standard deviation of 1h and
    #   come back home at 5 pm (17 -> 5) with standard deviation of 2h
    # 3) Plug-in times of EVs
    plug_in_times = []
    for i in range(no_of_records):
        plug_in_times.append(int(round(generate_random_value(5, 1))))
    # 4) Plug-out times of EVs
    plug_out_times = []
    for i in range(no_of_records):
        plug_out_times.append(int(round(generate_random_value(19, 1))))

    # Required SoC at plug-out time is assumed be 90% of the total capacity to avoid premature aging
    # 6) SoC of EVs departure
    soc_at_departure = [0.9] * no_of_records

    # SoC at the plug-in time is assumed Gaussian with a mean of o.3 and a standard deviation of 0.1
    # 5) SoC of EVs at arrival
    soc_at_arrival = []
    for i in range(no_of_records):
        # To ensure that soc at arrival is always less than soc at departure
        while True:
            soc_in = generate_random_value(0.3, 0.1)
            if soc_in < soc_at_departure[i]:
                break
        soc_at_arrival.append(soc_in)

    # 7) Battery capacities of EVs
    capacities = []
    for i in maximum_power:
        if i == 3.8:
            capacities.append(random.choice([4.4, 16]))
        elif i == 3:
            capacities.append(16)
        elif i == 3.3:
            capacities.append(24)

    # 8) Charging efficiency is assumed 85%
    efficiencies = [0.85] * no_of_records

    # Write to data/ev.csv
    # --------------------

    # Open the file for writing
    csv_out = open('../data/ev.csv', 'w')
    # Create the csv writer object
    mywriter = csv.writer(csv_out)
    # Write the header
    mywriter.writerow(["Maximum power", "Plug-in time", "Plug-out time", "SOC at arrival", "SOC at departure",
                       "Battery capacity", "Charging efficiency"])
    # Write all rows at once
    rows = zip(ev_id, maximum_power, plug_in_times, plug_out_times, soc_at_arrival, soc_at_departure, capacities,
               efficiencies)
    mywriter.writerows(rows)
    # Close the file
    csv_out.close()


#  ---- End Functions ---- #


# Main Script
# -----------

''' Base load '''

# Data is in MW, convert to kW
df_base_load = pd.read_csv('../data/base_load_NSW.csv', sep=',')
base_load = np.array(df_base_load['TOTALDEMAND']) * 1000

''' EV Information '''

# Generate EV data
# Sets the random seed
np.random.seed(1234)
generate_ev_data(N)
# Extract EV data
df_ev = pd.read_csv('../data/ev.csv', sep=',')
# SOC of EVs by arrival
soc_arr = np.array(df_ev['SOC at arrival'])
# Desired SOC of EVs by departure
soc_dep = np.array(df_ev['SOC at departure'])
# Battery capacity of EVs (in kWh)
cap = np.array(df_ev['Battery capacity'])
# Amount of power required by EVs (in kWh)
power_req = cap * (soc_dep - soc_arr)
# Maximum charging rate of EVs (in kW)
p_max = np.array(df_ev['Maximum power'])
# Plug-in time of EVs
t_plug_in = np.array(df_ev['Plug-in time'].astype(int))
# Plug-out time of EVs
t_plug_out = np.array(df_ev['Plug-out time'].astype(int))
# Charging efficiency of EVs
efficiency = np.array(df_ev['Charging efficiency'])

# Check feasibility of charging
for n in range(N):
    if (t_plug_out[n] - t_plug_in[n]) < (power_req[n] / p_max[n]):
        print('Solution is not feasible')
        break;

# Generation capacity
# Generation of NSW taken from
#   https://www.aemo.com.au/Electricity/National-Electricity-Market-NEM/Planning-and-forecasting/Generation-information
#   17080 MW
generation = 17080 * 1000
average_generation = generation / N

# Average demand
average_demand = base_load / N

# Check if a Nash equilibrium exists
# -----------------------------------
# r_min = min(t) {d(t)} / c
_r_min = np.amin(average_demand) / average_generation

# r_max = ( max(t) {d(t)} + gamma ) / c
# gamma : average energy requirement
# gamma = (beta / alpha) (1 - xo)
gamma = np.sum((cap / efficiency) * (soc_dep - soc_arr)) / N
_r_max = (np.amax(average_demand) + gamma) / average_generation

r_max = max(_r_min, _r_max)
r_min = min(_r_min, _r_max)

# Price function
# P(r) = 0.15 * r^1.5
# d p(r) / dr = 0.15 * 1.5 r = 0.225 r
# Left bound:
#   (1/2c) max { d p(r) / dr } = (1/2c) max { 0.225 r }
left_bound = (0.225 * r_max) / (2 * average_generation)
print('L', left_bound)
# Right bound:
#   (a/c) min { d p(r) / dr } = (a/c) min { 0.225 r } = a * (min { 0.225 r } / c)
right_bound = (0.225 * r_min) / average_generation
print('R', right_bound)

# Possible values for a 0.5 < a < 1
# left_bound <= delta <= a * right_bound
# Check if (left_bound/right_bound) <= a
RANGE_LEFT = 0.5
RANGE_RIGHT = 1
# ratio = left_bound / right_bound
# if ratio < RANGE_RIGHT and ratio > RANGE_LEFT:
#     a = random.uniform(RANGE_LEFT, RANGE_RIGHT)
#     delta = random.uniform(left_bound, a * right_bound)
#     print(a, delta)
# else:
#     print('a does not lie in between 0.5 and 1')

delta_left = max(left_bound, RANGE_LEFT * right_bound)
delta_right = RANGE_RIGHT * right_bound

print(delta_left, delta_right)

if delta_left < delta_right:
    delta = random.uniform(delta_left, delta_right)
    print(delta)
else:
    print('a does not lie in between 0.5 and 1')
