import pandas as pd
import matplotlib as mpl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
mpl.use('Agg')

def find_local_minima(df, col, window_size):  # possible que ca foire sur le 1e calcul
    # Si on a 2 min ou max consécutif : prendre la 2e valeur (pour repartir et trouver un autre min) et
    # ajouter 30 mn dans le time
    local_minima = []
    local_maxima = []
    time_local_min, time_local_max = [], []
    i = 0
    while i < len(df[col]):
        window = df[col].loc[i:i + window_size]
        # print(window)
        if len(window) < 10:  # To manage the last cycles
            break
        local_max_idx = window.idxmax()
        # print('local max idx', local_max_idx)
        local_maxima.append(window[local_max_idx])
        # if df[col].loc[local_max_idx] == df[col].loc[local_max_idx + 1]:
        #     # print('2 consecutives MAX, we assume that the peak is +30mn')
        #     time_local_max.append(df['Datetime'].loc[local_max_idx] + timedelta(minutes=30))
        # elif df[col].loc[local_max_idx] == df[col].loc[local_max_idx + 1] == df[col].loc[local_max_idx + 2]:
        #     # print('3 consecutives MAX §§')
        #     time_local_min.append(df['Datetime'].loc[local_max_idx + 1])
        # elif df[col].loc[local_max_idx] == df[col].loc[local_max_idx + 1] == df[col].loc[local_max_idx + 2] \
        #         == df[col].loc[local_max_idx + 3]:
        #     # print('4 consecutives MAX §§')
        #     time_local_min.append(df['Datetime'].loc[local_max_idx + 1] + timedelta(minutes=30))
        # else:
        #     time_local_max.append(df['Datetime'].loc[local_max_idx])
        time_local_max.append(df['Datetime'].loc[local_max_idx])

        window2 = df[col].loc[local_max_idx:local_max_idx + window_size]  # shift the window studied in order to find
        # another local min after the max
        local_min_idx = window2.idxmin()
        local_minima.append(window2[local_min_idx])
        # We do not take in account the values identical in this loop.
        # if df[col].loc[local_min_idx] == df[col].loc[local_min_idx + 1]:
        #     # print('2 consecutives MIN, we assume that the peak is +30mn')
        #     time_local_min.append(df['Datetime'].loc[local_min_idx] + timedelta(minutes=30))
        # elif df[col].loc[local_min_idx] == df[col].loc[local_min_idx + 1] == df[col].loc[local_min_idx + 2]:
        #     # print('3 consecutives MIN **')
        #     time_local_min.append(df['Datetime'].loc[local_min_idx + 1])
        # else:
        #     time_local_min.append(df['Datetime'].loc[local_min_idx])
        time_local_min.append(df['Datetime'].loc[local_min_idx])
        i = local_min_idx + 1
    #time_local_min = [dt.replace(microsecond=0) for dt in time_local_min]
    #time_local_max = [dt.replace(microsecond=0) for dt in time_local_max]
    # 2d loop to check the min and max are well detected
    local_minima2 = []
    local_maxima2 = []
    time_local_min2, time_local_max2 = [], []
    for i in range(len(local_minima)-1):
        min1 = df[col].loc[df['Datetime'] == time_local_min[i]].values[0]
        time_min1 = time_local_min[i]
        min1_2d = -9999
        time_min1_2d = -9999
        #print('min1 ', min1, time_min1)
        max1 = df[col].loc[df['Datetime'] == time_local_max[i]].values[0]
        time_max1 = time_local_max[i]
        max1_2d = -9999
        time_max1_2d = -9999
        for window_size in range(-5, 5):
            test_min = df[col].loc[df['Datetime'] == time_local_min[i] + timedelta(hours=window_size)].values[0]
            test_max = df[col].loc[df['Datetime'] == time_local_max[i] + timedelta(hours=window_size)].values[0]
            if test_min < min1:
                min1 = test_min
                time_min1 = time_local_min[i] + timedelta(hours=window_size)
            elif test_min == min1 and window_size !=0:
                # print(" on a 2 extrema consécutif : que fait on ? ")
                min1_2d = test_min
                time_min1_2d = time_local_min[i] + timedelta(hours=window_size)
            if test_max > max1:
                max1 = test_max
                time_max1 = time_local_max[i] + timedelta(hours=window_size)
            elif test_max == max1 and window_size !=0:
                max1_2d = test_max
                time_max1_2d = time_local_max[i] + timedelta(hours=window_size)
        if min1 == min1_2d :
            time_min1 = time_min1 + (time_min1_2d - time_min1)/2
        if max1 == max1_2d :
            time_max1 = time_max1 + (time_max1_2d - time_max1)/2
        local_minima2.append(min1)
        time_local_min2.append(time_min1)
        local_maxima2.append(max1)
        time_local_max2.append(time_max1)

    return np.array(local_minima2), np.array(local_maxima2), np.array(time_local_min2), np.array(time_local_max2)



file = '/home/penicaud/Documents/Data/Décharge_waterlevel/Data_2021-2022.xlsx'

columns_to_load = list(range(25))
df = pd.read_excel(file, sheet_name='sea_level-HonDau_2021-2022', usecols=columns_to_load, skiprows=4)
df = df.rename(columns={'Unnamed: 0': 'Date'})
melted_df = pd.melt(df, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df['Datetime'] = pd.to_datetime(melted_df['Date']) + pd.to_timedelta(melted_df['Hour'], unit='h')
melted_df.sort_values("Datetime", inplace=True)
melted_df = melted_df.dropna()
data_to_comp = melted_df.loc[(melted_df['Datetime'].dt.year == 2022 ) & (melted_df['Datetime'].dt.month >= 5) &
                             (melted_df['Datetime'].dt.month < 9)].dropna().reset_index()
data_to_comp['Value'] = data_to_comp['Value']/100


file2 = '/home/penicaud/Documents/Data/Décharge_waterlevel/tide_DoSon_from_May2022.xlsx'

df2 = pd.read_excel(file2, sheet_name='Sea level Do Son 5.2022', usecols=['hour', 'Tide'], skiprows=1)
df2 = df2.dropna()
df2 = df2.rename(columns={'time': 'Datetime'})
df2['Datetime'] = [dt.replace(microsecond=0) for dt in df2['Datetime']]
df2 = df2.loc[(df2['Datetime'].dt.month <9)]

fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Comparison prediction vs data at HD')
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=20)
ax.set_xlabel('Time', fontsize = 20)
ax.plot(df2['Datetime'], df2['Tide'], color = 'gray', label = 'Prediction')
ax.plot(data_to_comp['Datetime'], data_to_comp['Value'], color = 'black', label = 'Data')
ax.legend()
fig.savefig('Check_prediction_data_HD.png', format = 'png')

# Find local min and max
local_min_pred, local_max_pred, time_local_min_pred, time_local_max_pred = find_local_minima(df2, 'Tide', 17)
local_min_data, local_max_data, time_local_min_data, time_local_max_data = find_local_minima(data_to_comp, 'Value', 17)

s = 20
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Comparison prediction vs data at HD')
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=20)
ax.set_xlabel('Time', fontsize = 20)
ax.plot(df2['Datetime'], df2['Tide'], color = 'gray', label = 'Prediction')
ax.plot(data_to_comp['Datetime'], data_to_comp['Value'], color = 'black', label = 'Data')
ax.legend()
ax.scatter(time_local_min_pred, local_min_pred, marker = 'o', s = s, color = 'red')
ax.scatter(time_local_max_pred, local_max_pred, marker = 'o', s = s, color = 'red')
ax.scatter(time_local_min_data, local_min_data, marker = 'o', s = s, color = 'orange')
ax.scatter(time_local_max_data, local_max_data, marker = 'o', s = s, color = 'orange')
ax.set_xlim(datetime(2022,6,15), datetime(2022,6,20))
fig.savefig('Check_prediction_data_HD_zoomjunesurvey.png', format = 'png')