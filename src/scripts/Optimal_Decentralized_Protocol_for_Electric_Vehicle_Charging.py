# ''' Optimal Decentralized Protocol for Electric Vehicle Charging '''
# ''' Lingwen Gan, Ufuk Topcu and Steven H. Low '''

# ''' System '''
# Objective : Minimizing charging cost while achieving Valley filling
# Control Architecture : Decentralized Type(2)

# ''' How to execute '''
# Go to line 33, and define the number of EVs (N)
# Run the script

from cvxpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import csv
import random
import time


#  ---- Start Functions ---- #

# Time Horizon
# -------------

# Real
# 12, 13, 14, 15, 16, 17, ..., 23,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  11
# Following values are shown in the ev.csv as well as in the calculation
# 0 ,  1,  2,  3,  4,  5, ..., 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 , 23

# Number of EVs
N = 90

# Time horizon
T = 24

def calculate_charging_schedule(price, maximum_charging_rate, power_req, plug_in_time,
                                plug_out_time, previous_schedule):
    """ Calculate the optimal charging schedule for an EV using Quadratic Optimization.

    Keyword arguments:
        price : Electricity price
        number_of_time_slots : Charging duration
        maximum_charging_rate : Maximum allowable charging rate of the EV
        power_req : Total amount of power required by the EV
        plug_in_time : Plug-in time of EV
        plug_out_time : Plug-out time of EV
            Note: If plug-out time is x, EV can't be charged at time x
        previous_schedule : Charging profile of earlier nth iteration
    Returns:
        new_schedule : Charging profile of (n+1)th iteration (Updated charging rates during each time slot)

    Optimization:
        At nth iteration,
        Find x(n+1) that,
            Minimize  Σ(Charging cost + penalty term) for t=1,....,number_of_time_slots
            Minimize  Σ {<p(n), (previous_schedule)> + 1/2(new_schedule - previous_schedule)²}

    Assumptions:
        All EVs are available for negotiation at the beginning of scheduling period
    """

    # Define variables for new charging rates during each time slot
    new_schedule = Variable(T)

    # Define Objective function
    objective = Minimize(sum(price * new_schedule) + 0.5 * sum_squares(new_schedule - previous_schedule))

    # Define constraints list
    constraints = []
    # Constraint for charging rate limits
    constraints.append(0.0 <= new_schedule)
    constraints.append(new_schedule <= maximum_charging_rate)
    # Constraint for total amount of power required
    constraints.append(sum(new_schedule) == power_req)
    # Constraint for specifying EV's arrival & departure times
    if plug_in_time != 0:
        constraints.append(new_schedule[:plug_in_time] == 0)
    if plug_out_time == T - 1:
        constraints.append(new_schedule[plug_out_time] == 0)
    elif plug_out_time != T:
        constraints.append(new_schedule[plug_out_time:] == 0)

    # Solve the problem
    prob = Problem(objective, constraints)
    prob.solve()

    # Solution
    result = (new_schedule.value).tolist()
    # Uncomment:
    # print('Solution status: ', prob.status)
    # print('     Charging schedule: ', np.around(result, decimals=2))
    # print('     Objective value:', objective.value)

    # Return updated charging shedule
    return result


def calculate_price_signal(base_load, charging_schedules):
    """ Calculate the price signal based on updated charging schedules of EVs.
    Price during time t is modeled as a function of total demand during time t
    Keyword arguments:
        N : Number of EVs
        T : Time horizon
        base_load : Non-EV load
        charging_schedule : Updated charging schedules of EVs
    Returns:
        new_price : Control signal (price) for next iteration

    Price function:
        U = x²/2
        U' = x
        B (beta) = 1
        Y (gamma) = 1/(NB) = 1/N
        p(t) = YU'( base_load(t) + Σ charging_schedule ) ; n = 1,...,N   t=1,...,T
        p(t) = (1/N)( base_load(t) + Σ charging_schedule )
    """

    # Calculate gamma
    Y = 1 / N

    # Calculate total charging load of EVs at time t
    ev_load = np.zeros(T)
    for t in range(T):
        ev_load[t] = 0
        for n in range(N):
            ev_load[t] += charging_schedules[n][t]

    # Calculate total demand at time t (Base load + EV-load)
    total_load = base_load[:T] + ev_load

    # Calculate price signal
    price = Y * (total_load)

    # Return price signal
    return price


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
        generated_number = (-1)*generated_number
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
        plug_in_times.append(int(round(generate_random_value(5, 2))))
    # 4) Plug-out times of EVs
    plug_out_times = []
    for i in range(no_of_records):
        plug_out_times.append(int(round(generate_random_value(19, 1))))

    # SoC at the plug-in time is assumed Gaussian with a mean of o.5 and a standard deviation of 0.1
    # 5) SoC of EVs at arrival
    soc_at_arrival = []
    for i in range(no_of_records):
        soc_at_arrival.append(generate_random_value(0.5, 0.1))

    # Required SoC at plug-out time is assumed be 80% of the total capacity to avoid premature aging
    # 6) SoC of EVs departure
    soc_at_departure = [0.8] * no_of_records

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

    # bus_numbers = ['101', '102', '36', '40']
    # bus_names = ['NUC_A', 'NUC_B', 'CATDOG', 'HYDRO_A']
    # voltage = [.99, 1.02, 1.01, 1.00]

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

df_base_load = pd.read_csv('../data/base_load_NSW.csv', sep=',')
base_load = np.array(df_base_load['TOTALDEMAND'])

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
# Battery capacity of EVs
cap = np.array(df_ev['Battery capacity'])
# Amount of power required by EVs
power_req = cap * (soc_dep - soc_arr)
# Maximum charging rate of EVs
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

# Starting time of the algorithm
start_time = time.time()

# Step i
#   Initialize charging schedules of N EVs from t = 1,...,T
charging_schedules = np.zeros(shape=(N, T))

# Repeat steps ii, iii & iv until stopping criterion is met
previous_price = np.zeros(T)
k = 0
while True:
    # Uncomment:
    # print('\nIteration ', k)
    # print('----------------------------------------')

    # Step ii
    #   Utility calculates the price control signal and broadcasts to all EVs
    price = np.zeros(T)
    price = calculate_price_signal(base_load, charging_schedules)

    # Step iii
    # Each EV locally calculates a new charging profile by  solving the optimization problem
    # and reports new charging profile to utility
    new_charging_schedules = np.zeros(shape=(N, T))
    for n in range(N):
        # Uncomment:
        # print('For EV ', n)
        new_charging_schedules[n] = calculate_charging_schedule(price, p_max[n], power_req[n], t_plug_in[n],
                                                                t_plug_out[n], charging_schedules[n])

    # Stopping criterion
    # sqrt{(p(k) - p(k-1))²} <= 0.001, for t=1,...,T
    stop = True
    for t in range(T):
        deviation = np.sqrt((price[t] - previous_price[t]) ** 2)
        if deviation > 0.001:
            stop = False
            break

    if stop:
        break
    else:
        # Step iV
        # Go back to Step ii
        charging_schedules = new_charging_schedules
        previous_price = price
        k += 1

# Remove negative 0 values from output
charging_schedules[charging_schedules < 0] = 0

# Finishing time of the algorithm
end_time = time.time()

'''   Output result summary   '''

result = np.around(charging_schedules, decimals=2)
print('\n\nEV   In   Out   Max_power Power  Schedule t=(0,...,', T - 1, ')')
for n in range(N):
    print(n + 1, '  ', t_plug_in[n] + 12, '  ', t_plug_out[n] - 12, '  ', p_max[n], '     ',
          np.around(power_req[n], decimals=2), '  ', result[n])

'''   Graphical output   '''

# Plot
fig = plt.figure()
time_horizon = base_load.shape[0]
charging_horizon = result.shape[1]

# Change of base load
aggregate_load = np.zeros(time_horizon)
aggregate_load += base_load

for t in range(charging_horizon):
    for n in range(N):
        aggregate_load[t] += result[n][t]
for t in range(charging_horizon, time_horizon):
    aggregate_load[t] = base_load[t]

# Charging rate is kept constant during each time interval, so first value of the array is repeated
# Initial Base load
base_load = np.insert(base_load, 0, base_load[0])
# Aggregate load
aggregate_load = np.insert(aggregate_load, 0, aggregate_load[0])

# Draw graph
# ----------

# Round off to nearest multiple of 200
lower_bound = roundoff(np.amin(base_load), 200)
upper_bound = roundoff(np.amax(base_load), 200)

# Plot
plt.step(np.arange(0, time_horizon + 1, 1), aggregate_load, label='Aggregate load')
plt.step(np.arange(0, time_horizon + 1, 1), base_load, label='Initial load')

# Grid lines
plt.xticks(np.arange(25), ('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '0', '1', '2', '3',
                           '4', '5', '6', '7', '8', '9', '10', '11', '12'))
plt.yticks(np.arange(lower_bound, upper_bound, 200))

plt.legend(loc='best')
plt.xlabel("Time (12 pm to 12 am)")
plt.ylabel("Load")
plt.title("Variation of Total Load")
plt.grid()
plt.show()
plt.close()

''' Time spent for the execution of algorithm '''

print('\nTime spent for the execution of the algorithm: ', (end_time - start_time), 'seconds.')
print('Number of iterations for convergence: ', k)