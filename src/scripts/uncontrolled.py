# ''' Decentralized optimal demand-side management for PHEV charging in a smart grid '''

# ''' System '''
# Objective : Valley filling
# Control Architecture : Decentralized Type(2)

from cvxpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


#  ---- Start Functions ---- #

# Time Horizon
# -------------

# Real
# 12, 13, 14, 15, 16, 17, ..., 23,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  11
# Following values are shown in the ev.csv as well as in the calculation
# 0 ,  1,  2,  3,  4,  5, ..., 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 , 23

N = 100

# Time horizon
T = 24


# Main Script
# -----------

''' Base load '''

df_base_load = pd.read_csv('../data/base_load_southern_california_edison.csv', sep=',')
base_load = np.array(df_base_load['BASE_LOAD_PER_HOUSEHOLD']) * N * 2

''' EV Information '''

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

aggregate_load_6 = np.copy(base_load)
power_required = np.copy(power_req)

charging_schedules = np.zeros(shape=(N, T))

# Schedule EVs sequentially
for n in range(N):
    for t in range(t_plug_in[n], t_plug_out[n]):
        if (power_required[n] < p_max[n]):
            charging_schedules[n][t] = power_required[n]
            break
        else:
            charging_schedules[n][t] = p_max[n]
            power_required[n] = power_required[n] - p_max[n]
    aggregate_load_6 += charging_schedules[n]

result = np.around(charging_schedules, decimals=2)
print('\n\nEV   In   Out   Max_power Power  Schedule t=(0,...,', T - 1, ')')
for n in range(N):
    print(n + 1, '  ', t_plug_in[n] + 12, '  ', t_plug_out[n] - 12, '  ', p_max[n], '     ',
          np.around(power_req[n], decimals=2), '  ', result[n])

print('Base load: ', base_load)
print('Aggregate load: ', aggregate_load_6)


'''   Comparison Parameters   '''

print("\n--- Performance of the Algorithm (6) ---")

# 1. Peak
peak = np.amax(aggregate_load_6)
print("PEAK: ", peak, 'kW')

# 2. Maximum variance
variance = 0
average_aggregate_load = np.mean(aggregate_load_6)
for t in range(T):
    new_variance = (aggregate_load_6[t] - average_aggregate_load) ** 2
    if new_variance > variance:
        variance = new_variance
print("VARIANCE: ", variance, 'KW²')

# 3. PAR
par = peak / average_aggregate_load
print("PAR: ", par)


'''   Graphical output   '''

initial_load = np.insert(base_load, 0, base_load[0])
aggregate_load_6 = np.insert(aggregate_load_6, 0, aggregate_load_6[0])

# Draw graph
# ----------

# Plot step graphs
# plt.step(np.arange(0, T + 1, 1), aggregate_load, label='Aggregate load')
# plt.step(np.arange(0, T + 1, 1), initial_load, label='Initial load')
# Plot smoother graphs
l1=plt.plot(np.arange(0, T + 1, 1), aggregate_load_6, label='Aggregate load', linewidth=1)
l2=plt.plot(np.arange(0, T + 1, 1), initial_load, '--', label='Initial load', linewidth=2)
# Grid lines
plt.grid()
plt.xticks(np.arange(25), ('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '0', '1', '2', '3',
                           '4', '5', '6', '7', '8', '9', '10', '11', '12'))
plt.yticks(fontname="Times new roman", fontsize=9)
# Decorate graph
plt.legend(loc='lower left')
plt.xlabel("Time (12 pm to 12 am)")
plt.ylabel("Load (kW)")
plt.title("Water filling")
# Save figure
# Show graph
plt.show()
# Close
plt.close()



