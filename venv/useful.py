p = np.zeros((N,T),dtype=np.int16)
for ev in range(0, p.shape[0]):
    for time_slot in range(0, p.shape[1]):
        p[ev][time_slot] = p_max[ev]
        
        
        
df_ev.loc[ev,'Optimal start time'] = t_start
Name: Optimal start time, dtype: int64

df_ev.at[ev,'Optimal start time'] = t_start
Name: Optimal start time, dtype: object

Use iloc if your columns are not labeled:

df.iloc[3,4]
This will return the cell located at the intersection of the third row with the fourth column.

If you want to access the cell based on the column and row labels, use at:

df.at["Year","Temperature"]
This will return the cell intersected by the row "Year" and the column "Temperature".