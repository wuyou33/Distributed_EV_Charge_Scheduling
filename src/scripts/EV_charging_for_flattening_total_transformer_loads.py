import numpy as np
import pandas as pd
import math as mth
import matplotlib.pyplot as plt

"Nomenclature"
# ''' System '''
# N : Number of EVs
# T : Number of time slots
# m : Number of transformers, starts from 1
# ev[] : Set of EVs, starts from 1
# t[] : Time horizon, starts from 0
# delta : Duration of a time interval
#
# ''' Grid '''
# base[distribution_transformer][time] : Base load
# cap_transformer[distribution_transformer] : Capacity of distribution transformer
# price[time] : TOU energy cost
#
# ''' EV '''
# p_max[ev] : Maximum charging power rate
# efficiency[ev] : Charging efficiency
# cap[ev] : Battery capacity of EV
# soc_arr[ev] : State of charge of EV by arrival
# soc_dep[ev] : Desired state of charge of EV by departure

# ''' EV Specification '''
# t_plugin[ev] : Time at which EV is plugged in
# t_start[ev] : Time at which EV starts charging
# t_finish[ev] : Time at which EV finishes charging
# t_length[ev] : Length of the charging period of EV
# t_dead[ev] : Expected deadline for finish charging the EV
# t_availability[ev] : Available time for the EV to be charged

# ''' DataFrames '''
# df_tou : DataFrame that holds TOU rates of the time intervals
# df_ev : DataFrame that holds data about EVs
# df_schedule : DataFrame that holds the charging power rates of EVs
#               in different time intervals (CHARGE SCHEDULE)
# df_base_load_collection : DataFrame that holds forecasted base loads of all the transformers
# df_base_load[transformer][time] : A DataFrame matrix that holds base load of each transformer
#                                   in different time intervals


""
" Objective Function 1: Minimization of Load Variance"
""



''' TOU rates information '''

df_tou = pd.read_csv('../data/tou.csv', sep=',')
price = np.array(df_tou)

# Set the number of time intervals
T = price.size

# Assume each time interval is an hour
delta = 1



''' Base load information '''

df_base_load_collection = pd.read_csv('../data/base_load_at_transformers.csv', sep=',')

# Get the number of transformers
m = len(df_base_load_collection['Transformer'].unique())

# Create a new dataframe to hold the base loads of transformers separately
# Row : Transformer (starts with 1, access starts by 0)   ,  Column : Time (starts with 0)
index= range(1,m+1,1)
columns = range(0,T,1)
df_base_load = pd.DataFrame(index=index, columns=columns)

for trans in range(0,m):
    value_set = df_base_load_collection.loc[(trans*T):((trans+1)*T - 1), 'Load'].values
    df_base_load.iloc[trans] = value_set

# Keep a copy of the original base load profile of transformers separately
df_base_load_before = df_base_load.copy()

# Create a dataframe to hold the total base load across all transformers
index = range(0,T,1)
columns = ['Total load']
df_total_base_load = pd.DataFrame(index=pd.Index(index,name='Time'), columns=columns)

for t in range(0,T):
    df_total_base_load.loc[t] = 0
    for trans in range(0, m):
        df_total_base_load.loc[t,'Total load'] +=  df_base_load.iloc[trans, t]

# Keep a copy of the original total base load profile
df_total_base_load_before = df_total_base_load.copy()



''' EV information '''

df_ev = pd.read_csv('../data/ev.csv', sep=',')
# Set the number of EVs
N = df_ev.shape[0]

# Set the transformer of EVs
transformer = df_ev['Transformer']

# Set the maximum power rate of EVs
p_max = df_ev['Maximum power']

# Set the plug-in time of EVs
t_plugin = df_ev['Plug-in time'].astype(int)

# Set the deadline of EVs
t_dead = df_ev['Deadline'].astype(int)

# Set the battery capacity of EVs
cap = df_ev['Battery capacity']

# Set the SOC of EVs by arrival
soc_arr = df_ev['SOC at arrival']

# Set the desired SOC of EVs by departure
soc_dep = df_ev['SOC at departure']

# Set the Charging efficiency of EVs
efficiency = df_ev['Charging efficiency']

# Add a new column to the df_ev dataframe to store the optimal starting time of each EV
df_ev['Optimal start time'] = ""

# Initialize charging power of EVs
df_schedule = pd.DataFrame(0, index=np.arange(1, N + 1), columns=np.arange(T))

# Set charging power of EVs to maximum value
for ev in range(1, N + 1):
    for time_slot in range(0, T):
        df_schedule.ix[ev, time_slot] = p_max[ev]

# Calculate the required amount of energy for each EV
# df_ev['Energy required'] = df_ev['Battery capacity']*(df_ev['SOC at departure']-df_ev['SOC at arrival'])
energy_req = cap * (soc_dep - soc_arr)

# Calculate the time required for each EV to charge (number of time slots required)
t_length = np.ceil(energy_req / (efficiency * p_max * delta)).astype(int)



''' Check the feasibility of charging EVs as specified by the EV owner '''

t_availability = t_dead - t_plugin
for ev in range(1, N + 1):
    if t_length[ev] > t_availability[ev]:
        print('Charging EV', ev, 'is not feasible')
        print('Optimization aborted')
        exit()



''' Calculate the optimal starting time for each EV that maximizes the Objective Function 1 '''


# Find the optimal starting time for each EV, that minimizes the variance of the charging load
for ev in range(1, N + 1):

    optimal_total_load = mth.inf

    # Iterate through all possible time values to find the optimal start time that minimizes
    # the variance of the total load
    for t_start in range(t_plugin[ev], t_dead[ev] - t_length[ev] + 1):

        # Objective function: Variance of ev : Summation (t=1..T) [ B(t) + P(t)] ^ 2
        # B(t) : df_base_load[transformer[ev]].iloc[t]['Load']
        # P(t) : p_max[ev]
        total_load = 0
        total_load_before_plugin = 0
        total_load_while_plugin = 0
        total_load_after_plugin = 0

        for t in range(0, T):
            if t < t_start:
                total_load_before_plugin = (df_total_base_load.loc[t,'Total load']) ** 2
            elif t_start <= t <= (t_start + t_length[ev] - 1):
                total_load_while_plugin = (p_max[ev] + df_total_base_load.loc[t,'Total load']) ** 2
            elif t >= (t_start + t_length[ev]):
                total_load_after_plugin = (df_total_base_load.loc[t,'Total load']) ** 2

        total_load += (total_load_before_plugin + total_load_while_plugin + total_load_after_plugin)

        if total_load < optimal_total_load:
            optimal_total_load = total_load
            df_ev.loc[ev,'Optimal start time'] = t_start

    # Update the base load
    for t in range(df_ev.loc[ev,'Optimal start time'], t_start + t_length[ev] - 1):
        df_base_load.iloc[transformer[ev]-1, t] += p_max[ev]
        df_total_base_load.loc[t, 'Total load'] += p_max[ev]

# Extract the optimal starting time of EVs to an array
optimal_start_time = df_ev["Optimal start time"]

print('Optimal starting time for each EV:')
print(optimal_start_time)



''' Print results '''
new_index = ['[{0}-{1}]'.format(x, x+1) for x in range(0,T)]
before_load = np.array(df_total_base_load_before['Total load'])
after_load = np.array(df_total_base_load['Total load'])
d = {'Initial load': before_load, 'Aggregate load': after_load}
result_df = pd.DataFrame(data=d, index=pd.Index(new_index,name='Time range'))
print(result_df)



''' Plot graphs for load variation at transformers separately '''
after = np.concatenate([[after_load[0]],after_load])
before = np.concatenate([[before_load[0]],before_load])
plt.title('Total Load Variation', fontsize=20)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Aggregate load', fontsize=14)
plt.step(np.arange(0,T+1,1), before, label='Initial load')
plt.step(np.arange(0,T+1,1), after, label='Aggregated load')
plt.legend()
plt.show()

