# ''' Optimal Decentralized Protocol for Electric Vehicle Charging '''
# ''' Lingwen Gan, Ufuk Topcu and Steven H. Low '''

# ''' System '''
# Objective : Minimizing charging cost while achieving Valley filling
# Control Architecture : Decentralized Type(2)

from cvxpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#  ---- Start Functions ---- #

def calculate_charging_schedule(price, number_of_time_slots, maximum_charging_rate, power_req, plug_in_time, plug_out_time, previous_schedule):
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
    new_schedule = Variable(number_of_time_slots)

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
    if plug_out_time == T-1:
        constraints.append(new_schedule[plug_out_time] == 0)
    elif plug_out_time != T :
        constraints.append(new_schedule[plug_out_time:] == 0)

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
    total_load = base_load[:T] + ev_load

    # Calculate price signal
    price = Y * (total_load)

    # Return price signal
    return price


#  ---- End Functions ---- #


''' Base load '''
df_base_load = pd.read_csv('../data/base_load.csv', sep=',')
extract_base_load = df_base_load['Load']
base_load = extract_base_load.values

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
# Plug-in time of EVs
t_plug_in_series = df_ev['Plug-in time'].astype(int)
t_plug_in = t_plug_in_series.values
# Deadline of EVs
t_plug_out_series = df_ev['Deadline'].astype(int)
t_plug_out = t_plug_out_series.values
# Time horizon
#   Note: If maximum plug_out time is x, Time horizon goes from 1,...,x-1
T = np.max(t_plug_out)

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
        new_charging_schedules[n] = calculate_charging_schedule(price, T, p_max[n], power_req[n], t_plug_in[n], t_plug_out[n], charging_schedules[n])

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

'''   Output result summary   '''
result = np.around(charging_schedules, decimals=2)
print('\n\nEV   In   Out  Power  Schedule t=(0,...,', T-1,')')
for n in range(N):
    print(n+1, '  ', t_plug_in[n], '  ', t_plug_out[n], '  ', np.around(power_req[n], decimals=2), '  ', result[n])


'''   Graphical output   '''
# Plot
fig = plt.figure()

# Change of base load
aggregate_load = np.zeros(base_load.shape)
aggregate_load += base_load

for t in range(result.shape[1]):
    for n in range(N):
        aggregate_load[t] += result[n][t]
for t in range(result.shape[1], T+1):
        aggregate_load[t] = base_load[t]

# Charging rate is kept constant during each time interval, so first value of the array is repeated
# Initial Base load
base_load = np.insert(base_load, 0, base_load[0])
# Aggregate load
aggregate_load = np.insert(aggregate_load, 0, aggregate_load[0])

# Draw graph
# Grid lines
ax = fig.gca()
ax.set_xticks(np.arange(0, T+2, 1))
ax.set_yticks(np.arange(0, np.amax(base_load), 10))
# Plot
plt.step(np.arange(0,T+2,1), base_load, label='Initial load')
plt.step(np.arange(0,T+2,1), aggregate_load, label='Aggregate load')
plt.legend(loc='best')
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Variation of Total Load")
plt.grid()
plt.show()