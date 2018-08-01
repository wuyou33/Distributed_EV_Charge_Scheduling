# ''' Optimal Decentralized Protocol for Electric Vehicle Charging '''
# ''' Lingwen Gan, Ufuk Topcu and Steven H. Low '''

# ''' System '''
# Objective : Minimizing charging cost while achieving Valley filling
# Control Architecture : Decentralized Type(2)

from cvxpy import *
import numpy as np
import pandas as pd


#  ---- Start Functions ---- #

def calculate_charging_schedule(price, number_of_time_slots, maximum_charging_rate, power, previous_schedule):
    """ Calculate the optimal charging schedule for an EV using Quadratic Optimization.

    Keyword arguments:
        price : Electricity price
        number_of_time_slots : Charging duration
        maximum_charging_rate : Maximum allowable charging rate of the EV
        power : Total amount of power required by the EV
        previous_schedule : Charging profile of earlier nth iteration
    Returns:
        new_schedule : Charging profile of (n+1)th iteration (Updated charging rates during each time slot)

    Optimization:
        At nth iteration,
        Find x(n+1) that,
            Minimize  Σ(Charging cost + penalty term) for t=1,....,number_of_time_slots
            Minimize  Σ {<p(n), (previous_schedule)> + 1/2(new_schedule - previous_schedule)²}
    """

    # Define variables for new charging rates during each time slot
    new_schedule = Variable(number_of_time_slots)

    # Define Objective function
    objective = Minimize(sum(price * new_schedule) + 0.5 * sum_squares(new_schedule - previous_schedule))

    # Define constraints for charging rate limits and total amount of power required
    constraints = [0.0 <= new_schedule, new_schedule <= maximum_charging_rate, sum(new_schedule) == power]

    # Solve the problem
    prob = Problem(objective, constraints)
    prob.solve()

    # Solution
    result = (new_schedule.value).tolist()
    # print('Solution status: ', prob.status)
    print('     Charging schedule: ', np.around(result, decimals=2))
    print('     Objective value:', objective.value)

    # Return updated charging shedule
    return result


def calculate_price_signal(N, T, base_load, charging_schedules):
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
    total_load = base_load + ev_load

    # Calculate price signal
    price = Y * (total_load)

    # Return price signal
    return price


#  ---- End Functions ---- #


''' Base load '''
df_base_load = pd.read_csv('../data/base_load.csv', sep=',')
extract_base_load = df_base_load['Load']
base_load = extract_base_load.values
# Time horizon
T = df_base_load.shape[0]


''' EV Information '''
df_ev = pd.read_csv('../data/ev.csv', sep=',')
# Number of EVs
N = df_ev.shape[0]
# SOC of EVs by arrival
soc_arr = df_ev['SOC at arrival']
# Desired SOC of EVs by departure
soc_dep = df_ev['SOC at departure']
# Battery capacity of EVs
cap = df_ev['Battery capacity']
# Amount of power required by EVs
power_req_series = cap * (soc_dep - soc_arr)
power_req = power_req_series.values
# Maximum charging rate of EVs
p_max_series = df_ev['Maximum power']
p_max = p_max_series.values


''' Algorithm '''
# Step i
#   Initialize charging schedules of N EVs from t = 1,...,T
charging_schedules = np.zeros(shape=(N, T))

# Repeat steps ii, iii & iv until stopping criterion is met
previous_price = np.zeros(T)
k = 0
while True:
    print('\nIteration ', k)
    print('----------------------------------------')
    # Step ii
    #   Utility calculates the price control signal and broadcasts to all EVs
    price = np.zeros(T)
    price = calculate_price_signal(N, T, base_load, charging_schedules)

    # Step iii
    # Each EV locally calculates a new charging profile by  solving the optimization problem
    # and reports new charging profile to utility
    new_charging_schedules = np.zeros(shape=(N, T))
    for n in range(N):
        print('For EV ', n)
        new_charging_schedules[n] = calculate_charging_schedule(price, T, p_max[n], power_req[n], charging_schedules[n])

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
