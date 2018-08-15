# ''' Scalable Real-Time Electric Vehicles Charging With Discrete Charging Rates (srtevcwdcr) '''
# ''' G. Binetti, A. Davoudi, D. Naso, B. Turchiano and F. L. Lewis, "Scalable Real-Time Electric Vehicles Charging
#       With Discrete Charging Rates," in IEEE Transactions on Smart Grid, vol. 6, no. 5, pp. 2211-2220, Sept. 2015 '''

# ''' System '''
# Objective : Valley filling
# Control Architecture : Decentralized Type(2)

# ''' How to execute '''
# Go to line 40, and define the number of EVs (N)
# Run the script

# ''' Output '''
# Charging schedules of EVs are defined by the starting time for charging each EV
#   : optimal_start_time [n]


from cvxpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import csv
import random
import time
import math as mth

#  ---- Start Functions ---- #

# Time Horizon
# -------------

# Real
# 12, 13, 14, 15, 16, 17, ..., 23,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  11
# Following values are shown in the ev.csv as well as in the calculation
# 0 ,  1,  2,  3,  4,  5, ..., 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 , 23

# Number of households
H = 10000

# Number of EVs
# 50% penetration level
# Vehicles per household is 1.86
N = int(H * 0.5 * 1.86)

# Time horizon
T = 24


def roundoff(number, multiple):
    """ Round the given number to nearest given multiple.
    Keyword arguments:
        number : Number to round
        multiple : Round to this nearest value
    Returns:
        Rounded off value
    """

    # Smaller multiple
    a = (number // multiple) * multiple
    # Larger multiple
    b = a + multiple
    # Return of closest of two
    return (b if number - a > b - number else a)


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

    # Required SoC at plug-out time is assumed be 80% of the total capacity to avoid premature aging
    # 6) SoC of EVs departure
    soc_at_departure = [0.8] * no_of_records

    # SoC at the plug-in time is assumed Gaussian with a mean of o.5 and a standard deviation of 0.1
    # 5) SoC of EVs at arrival
    soc_at_arrival = []
    for i in range(no_of_records):
        # To ensure that soc at arrival is always less than soc at departure
        while True:
            soc_in = generate_random_value(0.5, 0.1)
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

    # 8) Charging efficiency is assumed 100%
    efficiencies = [1] * no_of_records

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

# Data  is taken from https://www.sce.com/005_regul_info/eca/DOMSM11.DLP (07/06/2018)
df_base_load = pd.read_csv('../data/base_load_southern_california_edison.csv', sep=',')
base_load = np.array(df_base_load['BASE_LOAD_PER_HOUSEHOLD']) * H
# Save for final graphical output
base_load_original = base_load.copy()

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
# Battery capacity of EVs (in MWh)
cap = np.array(df_ev['Battery capacity'])
# Charging efficiency of EVs
efficiency = np.array(df_ev['Charging efficiency'])
# Amount of power required by EVs (in kWh)
power_req = (cap / efficiency) * (soc_dep - soc_arr)
# Maximum charging rate of EVs (in MW)
p_max = np.array(df_ev['Maximum power'])
# Plug-in time of EVs
t_plug_in = np.array(df_ev['Plug-in time'].astype(int))
# Plug-out time of EVs
t_plug_out = np.array(df_ev['Plug-out time'].astype(int))

# Check feasibility of charging
for n in range(N):
    if (t_plug_out[n] - t_plug_in[n]) < (power_req[n] / p_max[n]):
        print('Solution is not feasible')
        break;

''' Algorithm '''

# Get the starting time of the algorithm
start_time = time.time()

# Calculate the time required for each EV to charge (number of time slots required)
t_length = np.ceil(power_req / (efficiency * p_max)).astype(int)

# Initialize the starting time for charging the EVs
optimal_start_time = np.zeros(N).astype(int)

# Schedule EVs sequentially
for n in range(N):

    # Starting criterion for optimal aggregate load check
    optimal_total_load = mth.inf

    # Iterate through all possible time values to find the optimal start time that minimizes
    # the variance of the total load
    for t_start in range(t_plug_in[n], t_plug_out[n]):

        # Objective function: Variance of ev : Summation (t=1..T) [ B(t) + P(t)] ^ 2
        # Total load (non-EV load + EV load)
        total_load = 0
        for t in range(0, T):
            # Before charging starts
            if t < t_start:
                total_load += (base_load[t] ** 2)
            # During charging
            elif t_start <= t < (t_start + t_length[n]):
                total_load += ((p_max[n] + base_load[t]) ** 2)
            # After charging completes
            else:
                total_load += (base_load[t] ** 2)

        if total_load < optimal_total_load:
            optimal_total_load = total_load
            optimal_start_time[n] = t_start

    # Update the base load
    for t in range(optimal_start_time[n], (optimal_start_time[n] + t_length[n] -1)):
        base_load[t] += p_max[n]

# Get the finishing time of the algorithm
end_time = time.time()

'''   Output result summary   '''

result = np.around(optimal_start_time)
print('\n\nEV   In   Out   Max_power Power  Schedule t=(0,...,', T - 1, ')')
for n in range(N):
    print(n + 1, '  ', (t_plug_in[n] + 12), '  ', (t_plug_out[n] - 12), '  ', p_max[n], '     ',
          np.around(power_req[n], decimals=2), '  ', (result[n] - 12), ':', t_length[n])

print('Base load: ', base_load_original)
print('Aggregate load: ', base_load)

'''   Graphical output   '''

# Charging rate is kept constant during each time interval, so first value of the array is repeated
# Initial Base load
initial_load = np.insert(base_load_original, 0, base_load_original[0])
# Aggregate load
aggregate_load = np.insert(base_load, 0, base_load[0])

# Draw graph
# ----------

# Plot step graphs
# plt.step(np.arange(0, T + 1, 1), aggregate_load, label='Aggregate load')
# plt.step(np.arange(0, T + 1, 1), initial_load, label='Initial load')
# Plot smoother graphs
plt.plot(np.arange(0, T + 1, 1), aggregate_load, label='Aggregate load')
plt.plot(np.arange(0, T + 1, 1), initial_load, label='Initial load')

# Grid lines
plt.grid()
plt.xticks(np.arange(25), ('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '0', '1', '2', '3',
                           '4', '5', '6', '7', '8', '9', '10', '11', '12'))
# Decorate graph
plt.legend(loc='best')
plt.xlabel("Time (12 pm to 12 am)")
plt.ylabel("Load (kW)")
plt.title("Scalable Real-Time Electric Vehicles Charging \n"
          "With Discrete Charging Rates")
# Save figure
plt.savefig('../figures/srtevcwdcr.png')
# Show graph
plt.show()
# Close
plt.close()

''' Time spent for the execution of algorithm '''

print('\nTime spent for the execution of the algorithm: ', (end_time - start_time), 'seconds.')
