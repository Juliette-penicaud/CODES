# 08/08/2024 : Je charge tous les df avant

# 08/08 : changement de la méthode de concaténation de ebb and flood pour éviter le décalage dans les df et donc
# faux ajustements
import pandas as pd
import matplotlib as mpl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import cmcrameri as cmc
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import cmocean
from scipy import stats
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator

mpl.use('Agg')

# last one,  from model
def find_HW_LW_duration(ds, col):
    ################
    # POURRAIT ETRE OPTIMISEE
    # This function aims at finding the min and max of a timeseries waterlevel in a ds, adding a 2 boolean column correponding to min and max,
    # adding the mean ammplitude over one cycle (from low tide to low tide), and the duration of each
    ################
    window_length = 7 # 15
    polyorder = 2
    prominence = 12 #0.2
    # NECESSAIRE de faire car le tableau de T% a été déterminé avec cette méthode de lissage
    # Je lisse la courbe pour bien détecter les min et max.
    ds['water_level_smooth'] = savgol_filter(ds[col], window_length=window_length, polyorder=polyorder)
    max, _ = find_peaks(ds['water_level_smooth'], prominence=prominence)
    min, _ = find_peaks(-ds['water_level_smooth'], prominence=prominence)
    # max, _ = find_peaks(ds['ssh_w'], prominence= prominence)
    # min, _ = find_peaks(-ds['ssh_w'], prominence = prominence)

    figure = False
    if figure :
        for month in np.arange(1, 13, 1):
            print(month)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ds['Datetime'], ds[col], lw=0.5)
            ax.scatter(ds['Datetime'][max].values, ds[col][max].values, c='red', zorder=5, s=7)  # -timedelta(hours=5)
            ax.scatter(ds['Datetime'][min].values, ds[col][min].values, c='k', zorder=5, s=7)  # -timedelta(hours=5)
            # plt.scatter(time_local_max_data, local_maxima_data, c='k', zorder=5, s=7)
            if month == 12:
                ax.set_xlim(datetime(2017, month, 1), datetime(2017, month, 31))
            else:
                ax.set_xlim(datetime(2017, month, 1), datetime(2017, month + 1, 1))
            fig.savefig('test_water_level_' + str(month) + '.png')

    ##########################
    ##########################
    # PART1 : je rajoute 2 colonnes boolean pour chaque HW et LW, afin de pouvoir calculer facilement les lags.
    # Initialize the new columns with NaN
    # ds['HW'] = ('time', np.nan * np.ones(ds.ssh_w.shape))
    # ds['LW'] = ('time', np.nan * np.ones(ds.ssh_w.shape))
    # ds['HW'][max] = True
    # ds['LW'][min] = True

    ##########################
    ##########################
    # calcul des TR et durées
    time_local_min_data = ds['Datetime'][min].values
    local_minima_data = ds[col][min].values
    local_maxima_data = ds[col][max].values
    time_local_max_data = ds['Datetime'][max].values

    Ebb = pd.DataFrame(
        time_local_max_data)  # the starting datetime is the beginning of the ebb i.e : max water levels at TT
    Flood = pd.DataFrame(time_local_min_data)
    Ebb = Ebb.rename(columns={0: 'Datetime ebb'})
    Flood = Flood.rename(columns={0: 'Datetime flood'})
    if time_local_max_data[0] > time_local_min_data[0]:  # To know which one we need to substract
        print('The first extremum is the minimum data, so it is the flood')
        Flood['Duration flood'] = time_local_max_data - time_local_min_data
        Flood['Amplitude flood'] = local_maxima_data - local_minima_data
        Ebb['Duration ebb'] = np.roll(time_local_min_data, shift=-1) - time_local_max_data
        Ebb['Amplitude ebb'] = np.roll(local_minima_data, shift=-1) - local_maxima_data
        Ebb.loc[len(Ebb) - 1, 'Duration ebb'] = np.nan
        Ebb.loc[len(Ebb) - 1, 'Amplitude ebb'] = np.nan
        combined_df = pd.concat([Flood, Ebb], axis=1)
    else:
        print('The first extremum is the MAX data, so it is the Ebb')
        Flood['Duration flood'] = np.roll(time_local_max_data, shift=-1) - time_local_min_data
        Flood['Amplitude flood'] = np.roll(local_maxima_data, shift=-1) - local_minima_data
        Flood.loc[len(Flood) - 1, 'Duration flood'] = np.nan
        Flood.loc[len(Flood) - 1, 'Amplitude flood'] = np.nan
        Ebb['Duration ebb'] = time_local_min_data - time_local_max_data
        Ebb['Amplitude ebb'] = local_minima_data - local_maxima_data
        combined_df = pd.concat([Ebb, Flood], axis=1)

    combined_df['Amplitude ebb'] = abs(combined_df['Amplitude ebb'])

    df = ds[[col, 'Datetime', 'Q (m3/s)']].reset_index()
    df = df.rename(columns={'Q (m3/s)':'Q'})


    # merged_df = df_ssh.merge(combined_df[['Datetime ebb', 'Amplitude ebb']], how='left', left_on='time', right_on='Datetime ebb')
    merged_df = df.merge(combined_df[['Datetime ebb', 'Duration ebb', 'Amplitude ebb']], how='left', left_on='Datetime',
                         right_on='Datetime ebb')

    # Merge df_ssh with df2 based on datetime_flood
    merged_df = merged_df.merge(combined_df[['Datetime flood', 'Duration flood', 'Amplitude flood']], how='left',
                                left_on='Datetime', right_on='Datetime flood')

    # fillna the duration of ebb and flood
    merged_df['Duration'] = merged_df['Duration ebb'].fillna(merged_df['Duration flood'])

    # Combine amplitude_ebb and amplitude_flood into tidal_range using fillna
    merged_df['Tidal range init'] = merged_df['Amplitude ebb'].fillna(merged_df['Amplitude flood'])

    ##########################
    ##########################
    # je rajoute 2 colonnes boolean pour chaque HW et LW, afin de pouvoir calculer facilement les lags.
    # Initialize the new columns with NaN
    merged_df['HW'] = False  # pd.Series(np.nan, dtype='boolean')
    merged_df['LW'] = False  # pd.Series(np.nan, dtype='boolean')
    merged_df.loc[max, 'HW'] = True
    merged_df.loc[min, 'LW'] = True

    ###########
    ###########
    # Do the mean on the tidal cycle (from 0 T% to 0T%)
    #cycle_id = (merged_df['T%'] == 0).cumsum()  # before it was high value, but I want to count from the low tide.
    #mean_in_cycles = merged_df.groupby(cycle_id)['Tidal range init'].mean()

    #high_tide_indices, low_tide_indices = find_high_low_cycles(merged_df)

    #mean_tidal_range = pd.Series(np.nan, index=merged_df.index)
    #mean_in_cycles = merged_df.groupby(cycle_id)['Tidal range init'].mean()
    #mean_in_cycles = mean_in_cycles[1::]  # put at size 348
    #mean_tidal_range[low_tide_indices] = mean_in_cycles.values
    #merged_df['Water level amplitude'] = mean_tidal_range

    # Drop unnecessary columns
    #merged_df.drop(
    #    ['Datetime ebb', 'Amplitude ebb', 'Datetime flood', 'Amplitude flood', 'Duration flood', 'Duration ebb'],
     #   axis=1, inplace=True)  # 'Tidal range init'
    # fill the nan with the previous value
    #merged_df['Water level amplitude'] = merged_df['Water level amplitude'].ffill()
    # merged_df.iloc['T%', 18580:] = np.nan
    #merged_df.set_index('time', inplace=True)

    return merged_df


def find_local_minima(df, col, window_size, interp=False):  # possible que ca foire sur le 1e calcul
    # Si on a 2 min ou max consécutif : prendre la 2e valeur (pour repartir et trouver un autre min) et
    # ajouter 30 mn dans le time
    local_minima = []
    local_maxima = []
    time_local_min, time_local_max = [], []
    time_local_SW, local_SW = [], []
    i = 0
    len_max = 10
    if interp == True:
        len_max = 0.7*window_size
    while i < len(df[col]):
        window = df[col].loc[i:i + window_size].astype(float)
        if len(window) < len_max:  # To manage the last cycles
            print('len is too small')
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
        # 17/01 : I add the SW on both set of windows because we have 2 times more SW than HT or LT
        local_SW_idx = abs(window).idxmin()
        local_SW_idx_inf = local_SW_idx if local_SW_idx == 0 else local_SW_idx - 1
        local_SW_idx_sup = local_SW_idx if local_SW_idx == len(df[col])-1 else local_SW_idx + 1
        if df[col].loc[local_SW_idx_inf] * df[col].loc[local_SW_idx_sup] < 0:
            #18/05 : je rajoute une condition: il doit y avoir eu un changement de signe pour valider le SW
            local_SW.append(window[local_SW_idx])
            time_local_SW.append(df['Datetime'].loc[local_SW_idx])

        window2 = df[col].loc[local_max_idx:local_max_idx + window_size].astype(float)  # shift the window studied in order to find
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
        # 17/01 : I add the SW
        local_SW_idx = abs(window2).idxmin() # Attention, cela prend en compte le minimum de la valeur absolue,
        # et pas vraiment quand on est à 0, donc il est possible que des fois, la valeur n'est pas 0 car il n'y a
        # pas de passage à 0
        local_SW_idx_inf = local_SW_idx if local_SW_idx == 0 else local_SW_idx - 1
        local_SW_idx_sup = local_SW_idx if local_SW_idx == len(df[col])-1 else local_SW_idx + 1
        if df[col].loc[local_SW_idx_inf] * df[col].loc[local_SW_idx_sup] < 0:
            #18/05 : je rajoute une condition: il doit y avoir eu un changement de signe pour valider le SW
            local_SW.append(window2[local_SW_idx])
            time_local_SW.append(df['Datetime'].loc[local_SW_idx])

        i = local_min_idx + 1

    # 2d loop to check the min and max are well detected
    local_minima2 = []
    local_maxima2 = []
    time_local_min2, time_local_max2 = [], []
    for i in range(len(local_minima) - 1):
        min1 = df[col].loc[df['Datetime'] == time_local_min[i]].values[0]
        time_min1 = time_local_min[i]
        min1_2d = -9999
        time_min1_2d = -9999
        # print('min1 ', min1, time_min1)
        max1 = df[col].loc[df['Datetime'] == time_local_max[i]].values[0]
        time_max1 = time_local_max[i]
        max1_2d = -9999
        time_max1_2d = -9999
        for window_size in range(-5, 5):
            # Calculate time_val_min and time_val_max with the added window_size
            time_val_min = time_local_min[i] + timedelta(hours=window_size)
            time_val_max = time_local_max[i] + timedelta(hours=window_size)

            # Ensure that time_val_min and time_val_max are not before the earliest datetime in df
            start_time = df['Datetime'][0]
            time_val_min = max(time_val_min, start_time)
            time_val_max = max(time_val_max, start_time)

            test_min = df[col].loc[df['Datetime'] == time_val_min].values[0]
            test_max = df[col].loc[df['Datetime'] == time_val_max].values[0]
            if test_min < min1:
                min1 = test_min
                time_min1 = time_local_min[i] + timedelta(hours=window_size)
            elif test_min == min1 and window_size != 0:
                # print(" on a 2 extrema consécutif : que fait on ? ")
                min1_2d = test_min
                time_min1_2d = time_local_min[i] + timedelta(hours=window_size)
            if test_max > max1:
                max1 = test_max
                time_max1 = time_local_max[i] + timedelta(hours=window_size)
            elif test_max == max1 and window_size != 0:
                max1_2d = test_max
                time_max1_2d = time_local_max[i] + timedelta(hours=window_size)
        if min1 == min1_2d:
            time_min1 = time_min1 + (time_min1_2d - time_min1) / 2
        if max1 == max1_2d:
            time_max1 = time_max1 + (time_max1_2d - time_max1) / 2
        local_minima2.append(min1)
        time_local_min2.append(time_min1)
        local_maxima2.append(max1)
        time_local_max2.append(time_max1)

    return np.array(local_minima2), np.array(local_maxima2), np.array(time_local_min2), np.array(time_local_max2),\
           np.array(local_SW), np.array(time_local_SW)


fontsize=15
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 4
s = 25

rep = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
file = rep + 'Data_HD_TT_2015-2022.xlsx'

columns_to_load = list(range(27))
df2 = pd.read_excel(file, sheet_name='HonDau_08-20', usecols=columns_to_load, skiprows=2)
df2 = df2.rename(columns={'Unnamed: '+str(i): str(i-3) for i in range(3,27)})
df2 = df2.rename(columns={'Nam':'Year', 'thang':'Month', 'Ngày':'Day'})
df2['Date'] = pd.to_datetime(df2[['Year', 'Month', 'Day']])
df2.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

melted_df2 = pd.melt(df2, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df2['Datetime'] = pd.to_datetime(melted_df2['Date']) + pd.to_timedelta(melted_df2['Hour'].astype(int), unit='h')
melted_df2.sort_values("Datetime", inplace=True)
melted_df2 = melted_df2.rename(columns={'Value': 'Water level HD'})
melted_df2.drop(['Hour'], axis=1, inplace=True)
melted_df2['Water level HD'] = melted_df2['Water level HD']/100
df_all_HD = melted_df2.copy()
df_all_HD = df_all_HD.reset_index().drop(['index'], axis = 1)

columns_to_load = list(range(25))
list_month = np.arange(1, 13)
nrows = 31
skip = 4  # Correspond à l'en tête
list_skip2 = [2,1]  # correspond au nombre de ligne entre les tableaux

# Traitement spécifique de HD 2021 et 2022 qui sont dans des feuilles à part
i = 0
for y in [21, 22]:
    print('i', i)
    skip2 = list_skip2[i]  # correspond au nombre de ligne entre les tableaux
    print('skip2', skip2)
    skip_new = 4  # 1364 si on commence en 2012
    for month in list_month :
        print('month', month)
        df = pd.read_excel(file, sheet_name='HonDau'+str(y), skiprows=skip_new, nrows=nrows, usecols=columns_to_load)
        #print(df[0:2], df[-3:])
        df = df.rename(columns={'Ngày': 'Day'})  # Je renomme les colonnes
        df = df.rename(columns={'Unnamed: '+str(i): str(i-1) for i in range(0,25)})  # Y compris celles qui sont en mois en chiffre
        df['Year'] = 2000 + y
        df['Month'] = month
        #print(df[0:2], df[-3:])

        melted_df = pd.melt(df, id_vars=['Year', 'Month', 'Day'], value_vars=[f'{i}' for i in range(24)],
                            var_name='Hour', value_name='Water level HD')

        # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
        melted_df = melted_df.dropna(subset=['Water level HD'])
        # Sort the values by Year, Month, and Day to maintain chronological order
        melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
        melted_df['Datetime'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day', 'Hour']])
        melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])
        melted_df = melted_df[['Datetime', 'Date', 'Water level HD']]
        melted_df['Water level HD'] = melted_df['Water level HD']/100

        df_all_HD = pd.concat([df_all_HD, melted_df], ignore_index=True)
        skip_new = skip_new + skip + nrows + skip2 + 1
    i = i+1

# TT Water level
print('TT water level df loading')
df_all_TT = pd.DataFrame(columns=['Datetime', 'Date', 'Water level TT'])
list_year = np.arange(2015, 2023, 1)
#list_year = [2017]
for year in list_year:
    #df_TT = pd.DataFrame(columns=['Datetime', 'Date', 'Water level TT'])
    list_month = np.arange(1, 13)
    nrows = 31
    skip = 4  # Correspond à l'en tête
    skip_new = 4  # 1364 si on commence en 2012
    skip2 = 2  # correspond au nombre de ligne entre les tableaux
    columns_to_load = list(range(25))
    for month in list_month :
        print('month', month)
        df = pd.read_excel(file, sheet_name='H_'+str(year), skiprows=skip_new, nrows=nrows, usecols=columns_to_load)
        #print(month, '\n', df[0:2]) #, df[-3:])
        df = df.rename(columns={'Ngày': 'Day'})  # Je renomme les colonnes
        df = df.rename(columns={'Unnamed: '+str(i): str(i-1) for i in range(0,25)})  # Y compris celles qui sont en mois en chiffre
        df['Year'] = year
        df['Month'] = month
        print(month)#, '\n', df[0:2], df[-3:])

        melted_df = pd.melt(df, id_vars=['Year', 'Month', 'Day'], value_vars=[f'{i}' for i in range(24)],
                            var_name='Hour', value_name='Water level TT')

        # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
        melted_df = melted_df.dropna(subset=['Water level TT'])
        # Sort the values by Year, Month, and Day to maintain chronological order
        melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
        melted_df['Datetime'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day', 'Hour']])
        melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])
        melted_df = melted_df[['Datetime', 'Date', 'Water level TT']]
        melted_df['Water level TT'] = melted_df['Water level TT']/100

        df_all_TT = pd.concat([df_all_TT, melted_df], ignore_index=True)
        skip_new = skip_new + skip + nrows + skip2 + 1

# DISCHARGE
print('Discharge file loading')
df_all_Q = pd.DataFrame(['Date', 'Datetime', 'Q'])
for year in list_year:
    print('year ', year)
    if year == 2017 :
        print('year = 2017, Specific treatment')
        df_Q = pd.read_excel(file, sheet_name='Q_' + str(year),
                           usecols=['Date', 'Time (hours)', 'Q (m3/s)'], skiprows=3)  # (m3/s)'
        df_Q['Datetime'] = df_Q['Date'] + pd.to_timedelta(df_Q['Time (hours)'].astype(int), unit='h')
        df_Q.drop(['Time (hours)'], axis=1, inplace=True)
        df_Q= df_Q.rename(columns={'Q (m3/s)': 'Q'})
        df_all_Q = pd.concat([df_all_Q, df_Q], ignore_index=True)
    else :
        df_Q = pd.DataFrame(columns=['Datetime', 'Date', 'Q'])
        list_month = np.arange(1, 13)
        nrows = 31
        skip = 4  # Correspond à l'en tête
        skip_new = 4  # 1364 si on commence en 2012
        skip2 = 2  # correspond au nombre de ligne entre les tableaux
        columns_to_load = list(range(25))
        for month in list_month :
            print('month', month)
            df = pd.read_excel(file, sheet_name='Q_'+str(year), skiprows=skip_new, nrows=nrows, usecols=columns_to_load)
            #print(month, '\n', df[0:2]) #, df[-3:])
            df = df.rename(columns={df.columns[0]: 'Day'})  # Je renomme les colonnes
            df = df.rename(columns={'Unnamed: '+str(i): str(i-1) for i in range(0,25)})  # Y compris celles qui sont en mois en chiffre
            df['Year'] = year
            df['Month'] = month
            #print(df[0:2], df[-3:])

            melted_df_Q = pd.melt(df, id_vars=['Year', 'Month', 'Day'], value_vars=[f'{i}' for i in range(24)],
                                var_name='Hour', value_name='Q')

            # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
            melted_df_Q = melted_df_Q.dropna(subset=['Q'])
            # Sort the values by Year, Month, and Day to maintain chronological order
            melted_df_Q = melted_df_Q.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
            melted_df_Q['Datetime'] = pd.to_datetime(melted_df_Q[['Year', 'Month', 'Day', 'Hour']])
            melted_df_Q['Date'] = pd.to_datetime(melted_df_Q[['Year', 'Month', 'Day']])
            melted_df_Q = melted_df_Q[['Datetime', 'Date', 'Q']]

            df_all_Q = pd.concat([df_all_Q, melted_df_Q], ignore_index=True)
            skip_new = skip_new + skip + nrows + skip2 + 1

daily_mean = df_all_Q.resample('D', on='Datetime').mean()
yearly_mean = df_all_Q.resample('Y', on='Datetime').mean()
print('Ok for loading files ')

print('Beginning of Ebb_and_flood construction ')
### Ebb_and_flood
window_size = 17
local_minima, local_maxima, time_local_min, time_local_max, a, b = \
    find_local_minima(df_all_TT, 'Water level TT', window_size)

# Amplitude à TT
Ebb = pd.DataFrame(time_local_max)  # the starting datetime is the beginning of the ebb i.e : max water levels at TT
Flood = pd.DataFrame(time_local_min)
Ebb = Ebb.rename(columns={0: 'Datetime ebb TT'})
Ebb['Datetime'] = Ebb['Datetime ebb TT'].copy()
Flood = Flood.rename(columns={0: 'Datetime flood TT'})
Flood['Datetime'] = Flood['Datetime flood TT'].copy()
if time_local_max[0] > time_local_min[0]:  # To know which one we need to substract
    print('The first extremum is the minimum data, so it is the flood')
    Flood['Duration'] = time_local_max - time_local_min
    Flood['Amplitude'] = local_maxima - local_minima
    Ebb['Duration'] = np.roll(time_local_min, shift=-1) - time_local_max
    Ebb['Amplitude'] = np.roll(local_minima, shift=-1) - local_maxima
    Ebb.loc[len(Ebb) - 1, 'Duration'] = np.nan
    Ebb.loc[len(Ebb) - 1, 'Amplitude'] = np.nan
else:
    print('The first extremum is the MAX data, so it is the Ebb')
    Flood['Duration'] = np.roll(time_local_max, shift=-1) - time_local_min
    Flood['Amplitude'] = np.roll(local_maxima, shift=-1) - local_minima
    Flood.loc[len(Flood) - 1, 'Duration'] = np.nan
    Flood.loc[len(Flood) - 1, 'Amplitude'] = np.nan
    Ebb['Duration'] = time_local_min - time_local_max
    Ebb['Amplitude'] = local_minima - local_maxima

# Ajout du débit
result_value_Q = []
for index1, row1 in Ebb.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1['Datetime']))  # Find the closest datetime index
    result_value_Q.append(daily_mean['Q'].iloc[closest_index])
Ebb['Q'] = result_value_Q
result_value_Qf = []
for index1, row1 in Flood.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1['Datetime']))  # Find the closest datetime index
    result_value_Qf.append(daily_mean['Q'].iloc[closest_index])
Flood['Q'] = result_value_Qf

print('Ok for Ebb and Flood at TT ')

# AT HD : to have the tidal amplitude without damping
df_HD_2015_2022 =    df_all_HD.loc[df_all_HD['Date'].dt.year >= 2015].reset_index().drop(['index'], axis = 1)

local_minima_HD, local_maxima_HD, time_local_min_HD, time_local_max_HD, a, b = \
    find_local_minima(df_HD_2015_2022, 'Water level HD', window_size)

Ebb_HD = pd.DataFrame(
    time_local_max_HD)  # the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
Flood_HD = pd.DataFrame(time_local_min_HD)
Ebb_HD = Ebb_HD.rename(columns={0: 'Datetime ebb HD'})
Ebb_HD['Datetime'] = Ebb_HD['Datetime ebb HD'].copy()
Flood_HD = Flood_HD.rename(columns={0: 'Datetime flood HD'})
Flood_HD['Datetime'] = Flood_HD['Datetime flood HD'].copy()
if time_local_max_HD[0] > time_local_min_HD[0]:  # To know which one we need to substract
    print('The first extremum is the minimum data, so it is the flood_HD')
    Flood_HD['Duration HD'] = time_local_max_HD - time_local_min_HD
    Flood_HD['Amplitude HD'] = local_maxima_HD - local_minima_HD
    Ebb_HD['Duration HD'] = np.roll(time_local_min_HD, shift=-1) - time_local_max_HD
    Ebb_HD['Amplitude HD'] = np.roll(local_minima_HD, shift=-1) - local_maxima_HD
    Ebb_HD.loc[len(Ebb_HD) - 1, 'Duration HD'] = np.nan
    Ebb_HD.loc[len(Ebb_HD) - 1, 'Amplitude HD'] = np.nan
else:
    print('The first extremum is the MAX data, so it is the Ebb_HD')
    Flood_HD['Duration HD'] = np.roll(time_local_max_HD, shift=-1) - time_local_min_HD
    Flood_HD['Amplitude HD'] = np.roll(local_maxima_HD, shift=-1) - local_minima_HD
    Flood_HD.loc[len(Flood_HD) - 1, 'Duration HD'] = np.nan
    Flood_HD.loc[len(Flood_HD) - 1, 'Amplitude HD'] = np.nan
    Ebb_HD['Duration HD'] = time_local_min_HD - time_local_max_HD
    Ebb_HD['Amplitude HD'] = local_minima_HD - local_maxima_HD

print('Ok for Ebb and Flood at HD ')
# Vmin and Vmax
# Min and Max Q i.e. Vmax and Vmin.
local_minima_Q, local_maxima_Q, time_local_min_Q, time_local_max_Q, a, b = \
    find_local_minima(df_all_Q, 'Q', window_size)

####### ICI je créé les df Ebb_df et Flood_df qui regroupent à la fois TT et HD
#Ebb_df = pd.concat([Ebb_HD_copy, Ebb], axis=1)
Ebb_df = pd.merge_asof(Ebb_HD, Ebb, on='Datetime', direction='nearest')
Ebb_df_abs = Ebb_df.copy()
Ebb_df_abs[['Amplitude', 'Amplitude HD']] = Ebb_df[['Amplitude', 'Amplitude HD']].abs()

#Flood_df = pd.concat([Flood_HD, Flood], axis=1)
Flood_df = pd.merge_asof(Flood_HD, Flood, on='Datetime', direction='nearest')

Ebb_and_flood = pd.concat([Ebb_df_abs, Flood_df], axis=0)

print('Ok for Ebb_and_flood !!  ')
Ebb_and_flood_clean = Ebb_and_flood.dropna(how='any', subset=['Amplitude', 'Q', 'Amplitude HD'])
z_clean = Ebb_and_flood_clean['Amplitude HD'].values
x_clean = Ebb_and_flood_clean['Q'].values
amplification_clean = Ebb_and_flood_clean['Amplitude'].values / Ebb_and_flood_clean['Amplitude HD'].values
y_clean = amplification_clean
slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(z_clean, y_clean)
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(z_clean, x_clean)


# Analyse des corrélations
print('Tablea_correlation building')
filter = True # corrélation et pentes filtrées <1000 m3/s
table_yearly_mean = yearly_mean.copy()
list_corr_Q = [-0.62, -0.45, -0.83, -0.78, -0.45, -0.56, -0.22, -0.66]
list_slope_Q = [-2.01, -2.32, -2.17, -2.16, -1.53, -1.76, -1.07, -1.74]
if not filter :
    list_corr_TR  = [-0.57, -0.49, -0.47, -0.45, -0.71, -0.61, -0.86, -0.54]
    list_slope_TR = [-1.01, -1.32, -1.03, -0.85, -0.88, -0.83, -1.02, -0.68]
else :
    list_corr_TR  = [-0.77, -0.50, -0.84, -0.86, -0.84, -0.83, -0.87, -0.76]
    list_slope_TR = [-1.11, -1.31, -1.17, -1.05, -0.97, -0.99, -1.02, -0.80]
table_yearly_mean['corr_Q'] = list_corr_Q
table_yearly_mean['corr_TR'] = list_corr_TR
table_yearly_mean['slope_Q'] = list_slope_Q
table_yearly_mean['slope_TR'] = list_slope_TR
slope, intercept, r_value, p_value, std_err = stats.linregress(table_yearly_mean['Q'], table_yearly_mean['corr_Q'])
print('r_value p_value', r_value, p_value)

val_quantile = 0.90
spring_tides_per_year_2015 = Ebb_and_flood.groupby(Ebb_and_flood['Datetime'].dt.year)['Amplitude HD'].quantile(
    [val_quantile])
slope, intercept, r_value, p_value, std_err = stats.linregress(spring_tides_per_year_2015.values, table_yearly_mean['slope_TR'])
print('r_value p_value', r_value, p_value)
figure_corr = True
save = True
if figure_corr :
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid('both', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel('Tidal amplitude of the 10% highest spring tides ', fontsize=fontsize)
    ax.set_ylabel('Slope value', fontsize=fontsize)
    im = ax.scatter(spring_tides_per_year_2015.values, table_yearly_mean['slope_TR'], lw=0.5, s=s, c='k')#, vmin=0, vmax=4, s=s)
    p_print = "{:.2e}".format(p_value) if p_value > 0.01 else "p<0.01"
    label = 'r = ' + str(np.round(r_value, 2)) + ' , ' + p_print
    label = label + "\ny = {:.2e}".format(slope) + ' x + ' + "{:.2e}".format(intercept)
    x = np.arange(2,5)
    ax.plot(x, slope * x + intercept, color='gray', lw=5, zorder=0, label=label)
    if save:
        outfile = 'corr_vs_spring_tide'
        fig.savefig(outfile)

# from several thesis
test = False
per_year = False
if test :
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    if per_year : # pas ok statistiquement parlant
        # Valeurs moy annuelles du débit à SOn Tay :
        list_Q_ST = [378.17430137, 408.4570765 , 568.79646575, 560.5660274 ,
               378.78846575, 428.78639344, 379.66958904, 486.21876712]

        df = pd.DataFrame(list_Q_ST)
        df = df.rename(columns={0: 'pure_discharge'})
        #df['pure_discharge'] = list_Q_ST
        df['spring_tide'] = spring_tides_per_year_2015.values
        df['discharge_influenced'] = table_yearly_mean['Q'].values
        df["interaction"] = df['pure_discharge'] * df["spring_tide"]

    else :
        # Je charge les valeurs de débit journaliers à ST
        rep = '/home/penicaud/Documents/Data/Décharge_waterlevel/'
        file = rep + 'SonTay_alldata.xlsx'

        df_all = pd.read_excel(file, sheet_name='Q_1975-2022', skiprows=2)
        df_2015_2022 = df_all[df_all['Date'].dt.year >= 2015]

        # Valeurs marnages :
        Ebb_and_flood['Date'] = Ebb_and_flood['Datetime'].dt.date

        df = pd.DataFrame(df_2015_2022['Discharge'].values)
        df = df.rename(columns={0: 'pure_discharge'})





    X = df[['pure_discharge', 'spring_tide', 'interaction']]
    y = df["discharge_influenced"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print(coeffs)


    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Discharge Influenced')
    plt.ylabel('Predicted Discharge Influenced')
    plt.title('Actual vs Predicted Discharge Influenced')
    plt.savefig('test_multilinear')
#######################################
########   FIGURE All years  ##########
#######################################
cmap = cmocean.cm.thermal
s = 25
fontsize = 35
save = True

figure_attenuation = True
if figure_attenuation :
    Ebb_and_flood_clean = Ebb_and_flood.dropna(how='any', subset=['Amplitude', 'Q', 'Amplitude HD'])
    z_clean = Ebb_and_flood_clean['Amplitude HD'].values
    x_clean = Ebb_and_flood_clean['Q'].values
    amplification_clean = Ebb_and_flood_clean['Amplitude'].values / Ebb_and_flood_clean['Amplitude HD'].values
    y_clean = amplification_clean

    z = Ebb_and_flood['Amplitude HD'].values
    x = Ebb_and_flood['Q'].values
    amplification = Ebb_and_flood['Amplitude'].values / Ebb_and_flood['Amplitude HD'].values
    y = amplification

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(z_clean, y_clean)
    p_print = "{:.2e}".format(p_value) if p_value > 0.01 else "p<0.01"
    p_print2 = "{:.2e}".format(p_value2) if p_value > 0.01 else "p<0.01"
    label = 'r = ' + str(np.round(r_value, 2)) + ' , ' + p_print
    label = label + "\ny = {:.2e}".format(slope) + ' x + ' + "{:.2e}".format(intercept)
    #label = label + '\nCorrelation with TR: r = ' + str(np.round(r_value2, 2)) + ' , ' + p_print2
    #label = label + "\ny = {:.2e}".format(slope2) + ' x + ' + "{:.2e}".format(intercept2)

    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle('2015-2022', fontsize=fontsize)
    ax.grid('both', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    im = ax.scatter(x, y, lw=0.5, c=z, cmap=cmap, vmin=0, vmax=4, s=s)
    # im = ax.scatter(x, y, lw=0.5, c=localisation[sta]['color'], marker=list_marker[i], label=sta)
    x_discharge = np.arange(0, 3600, 100)
    ax.plot(x_discharge, slope * x_discharge + intercept, color='gray', lw=5, zorder=0, label=label)
    ax.set_xlim(0, 3200)
    ax.set_ylim(0, 2)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(1))  # Set minor tick every 0.2 units on the x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_aspect(1200)
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontsize(fontsize=fontsize - 7)  # Set the desired font size
    ax.set_xlabel('Discharge (m³/s)', fontsize=fontsize)
    # ax.set_ylabel('Amplification', fontsize=fontsize)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar = fig.colorbar(im, ax=axs[1])
    cbar.ax.tick_params(labelsize=fontsize, width=0.5)
    cbar.set_label(label='Tidal amplitude at HD (m)', fontsize=fontsize)
    cbar.outline.set_linewidth(0.05)

    if save:
        outfile = 'Attenuation_vs_discharge_'
        outfile = outfile + 'all_years' + '.png'
        fig.savefig(outfile)


figure_attenuation_TR = True
filter = True
if figure_attenuation_TR :
    Ebb_and_flood_clean = Ebb_and_flood.dropna(how='any', subset=['Amplitude', 'Q', 'Amplitude HD'])
    seuil_Q = 1000
    condition = Ebb_and_flood_clean['Q']<=seuil_Q
    z_clean = Ebb_and_flood_clean['Amplitude HD'].loc[condition].values
    x_clean = Ebb_and_flood_clean['Q'].loc[condition].values
    amplification_clean = Ebb_and_flood_clean['Amplitude'].loc[condition].values / \
                          Ebb_and_flood_clean['Amplitude HD'].loc[condition].values
    y_clean = amplification_clean


    z = Ebb_and_flood['Amplitude HD'].values
    x = Ebb_and_flood['Q'].values
    amplification = Ebb_and_flood['Amplitude'].values / Ebb_and_flood['Amplitude HD'].values
    y = amplification

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(z_clean, y_clean)
    p_print = "{:.2e}".format(p_value) if p_value > 0.01 else "p<0.01"
    p_print2 = "{:.2e}".format(p_value2) if p_value > 0.01 else "p<0.01"
    #label = 'Correlation with Q: r = ' + str(np.round(r_value, 2)) + ' , ' + p_print
    #label = label + "\ny = {:.2e}".format(slope) + ' x + ' + "{:.2e}".format(intercept)
    label = 'r = ' + str(np.round(r_value2, 2)) + ' , ' + p_print2
    label = label + "\ny = {:.2e}".format(slope2) + ' x + ' + "{:.2e}".format(intercept2)
    label = label + "\nN= " + str(len(x_clean))

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title('2015-2022', fontsize=fontsize)
    ax.grid('both', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    im = ax.scatter(z, y, lw=0.5, c=x, cmap=cmap, vmin=0, vmax=3200, s=s)
    # im = ax.scatter(x, y, lw=0.5, c=localisation[sta]['color'], marker=list_marker[i], label=sta)
    x_TR = np.arange(0, 4.5, 0.1)
    ax.plot(x_TR, slope2 * x_TR + intercept2, color='gray', lw=5, zorder=0, label=label)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))  # Set minor tick every 0.2 units on the x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_aspect(1.700)
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontsize(fontsize=fontsize - 7)  # Set the desired font size
    plt.tight_layout(rect=[0, 0.1, 0.9, 1])
    ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize)
    # ax.set_ylabel('Amplification', fontsize=fontsize)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar = fig.colorbar(im, ax=axs[1])
    cbar.ax.tick_params(labelsize=fontsize, width=0.5)
    cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize)
    cbar.outline.set_linewidth(0.05)

    if save:
        outfile = 'Attenuation_vs_TR_'
        if filter :
            outfile = outfile + 'filterd_'
        outfile = outfile + 'all_years' + '.png'
        fig.savefig(outfile)

cmap = cmc.cm.batlow
vmax = 2500
figure_amplitude = True
interp = 'polyfit'
deb = 1000
if figure_amplitude :
    fig, ax = plt.subplots(figsize=(13, 10))

    ax.grid(which='both', alpha=0.5)
    fig.suptitle('2015-2022', fontsize=fontsize)
    ax.set_aspect("equal")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    p1 = ax.scatter(abs(Ebb_and_flood['Amplitude HD']),
                    abs(Ebb_and_flood['Amplitude']),
                    c=Ebb_and_flood['Q'], cmap=cmap, vmin=0, vmax=vmax, alpha=0.7)
    plt.tight_layout(rect=[0, 0.1, 0.9, 1])
    cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
    cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize - 1)
    cbar.outline.set_linewidth(0.05)
    #ax.set_ylabel('Tidal amplitude at TT (m)', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize - 2)

    if interp == 'polyfit':
        cond = Ebb_and_flood_clean['Q'] < deb
        Q = Ebb_and_flood_clean.loc[cond]
        QX = Q['Amplitude HD'].dropna()
        QY = Q['Amplitude'].dropna()
        print(interp)
        coefficients = np.polyfit(QX, QY, 2)
        y_pred = np.polyval(coefficients, QX.sort_values())
        r_value, _ = pearsonr(y_pred, QY.sort_values())
        label1 = str(str(np.round(coefficients[0], 2)) + ' x${²}$ + ' +
                     str(np.round(coefficients[1], 2)) + ' x + ' + str(np.round(coefficients[2], 2)) + '\nr=' +
                     "{:.3f}".format(r_value) + ', N=' + str(QX.count()))  # str(np.round(r_value, 3)) #" m$³$/s, "
        # label = str(label_title[c] + ' m$³$/s, r=' + "{:.3f}".format(r_value) + ' N=' + str(QX.count()))
        ax.plot(QX.sort_values(), y_pred, alpha=0.7, lw=5, color='k', zorder=1, label=label1)
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontsize(fontsize=fontsize - 7)  # Set the desired font size

    identity = np.arange(0, 5, 1)
    ax.plot(identity, identity, c='gray')
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if save:
        outfile = 'Amplitude_TT_vs_HD_Amplitude_'
        outfile = outfile + interp + '_all_year'
        outfile = outfile + '.png'
        fig.savefig(outfile, format='png')

###
# Figure all year spring tides at HD
figure_tidal_range_HD = False
if figure_tidal_range_HD:
    s = 50
    # 1. Je fais un tableau avec seulement Ebb and flood HD :
    local_minima_HD, local_maxima_HD, time_local_min_HD, time_local_max_HD, a, b = \
        find_local_minima(df_all_HD, 'Water level HD', window_size)

    Ebb_HD = pd.DataFrame(
        time_local_max_HD)  # the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
    Flood_HD = pd.DataFrame(time_local_min_HD)
    Ebb_HD = Ebb_HD.rename(columns={0: 'Datetime ebb HD'})
    Ebb_HD['Datetime'] = Ebb_HD['Datetime ebb HD'].copy()
    Flood_HD = Flood_HD.rename(columns={0: 'Datetime flood HD'})
    Flood_HD['Datetime'] = Flood_HD['Datetime flood HD'].copy()
    if time_local_max_HD[0] > time_local_min_HD[0]:  # To know which one we need to substract
        print('The first extremum is the minimum data, so it is the flood_HD')
        Flood_HD['Duration HD'] = time_local_max_HD - time_local_min_HD
        Flood_HD['Amplitude HD'] = local_maxima_HD - local_minima_HD
        Ebb_HD['Duration HD'] = np.roll(time_local_min_HD, shift=-1) - time_local_max_HD
        Ebb_HD['Amplitude HD'] = np.roll(local_minima_HD, shift=-1) - local_maxima_HD
        Ebb_HD.loc[len(Ebb_HD) - 1, 'Duration HD'] = np.nan
        Ebb_HD.loc[len(Ebb_HD) - 1, 'Amplitude HD'] = np.nan
    else:
        print('The first extremum is the MAX data, so it is the Ebb_HD')
        Flood_HD['Duration HD'] = np.roll(time_local_max_HD, shift=-1) - time_local_min_HD
        Flood_HD['Amplitude HD'] = np.roll(local_maxima_HD, shift=-1) - local_minima_HD
        Flood_HD.loc[len(Flood_HD) - 1, 'Duration HD'] = np.nan
        Flood_HD.loc[len(Flood_HD) - 1, 'Amplitude HD'] = np.nan
        Ebb_HD['Duration HD'] = time_local_min_HD - time_local_max_HD
        Ebb_HD['Amplitude HD'] = local_minima_HD - local_maxima_HD

    Ebb_HD_abs = Ebb_HD.copy()
    Ebb_HD_abs['Amplitude HD'] = Ebb_HD_abs['Amplitude HD'].abs()
    Ebb_and_flood_HD = pd.concat([Ebb_HD_abs, Flood_HD], axis=0)

    # Calcul des spring tides per year : quantile 0.95, je groupe par year, et je vois l'amplitude à HD.
    val_quantile = 0.90
    spring_tides_per_year = Ebb_and_flood_HD.groupby(Ebb_and_flood_HD['Datetime'].dt.year)['Amplitude HD'].quantile(
        [val_quantile])
    amplitude1 = spring_tides_per_year.max() - spring_tides_per_year.min()

    val_quantile2 = 0.95
    spring_tides_per_year2 = Ebb_and_flood_HD.groupby(Ebb_and_flood_HD['Datetime'].dt.year)['Amplitude HD'].quantile(
        [val_quantile2])
    amplitude2 = spring_tides_per_year2.max() - spring_tides_per_year2.min()

    val_quantile3 = 0.99
    spring_tides_per_year3 = Ebb_and_flood_HD.groupby(Ebb_and_flood_HD['Datetime'].dt.year)['Amplitude HD'].quantile(
        [val_quantile3])
    amplitude3 = spring_tides_per_year3.max() - spring_tides_per_year3.min()

    x = spring_tides_per_year.index.get_level_values(0)
    y = spring_tides_per_year.values
    x2 = spring_tides_per_year2.index.get_level_values(0)
    y2 = spring_tides_per_year2.values
    x3 = spring_tides_per_year3.index.get_level_values(0)
    y3 = spring_tides_per_year3.values

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Year', fontsize=fontsize, labelpad=20)
    ax.set_ylabel('Tidal amplitude at HD (m)', fontsize=fontsize, labelpad=20)
    ax.plot(x, y, marker='o', c='royalblue',
            label='10% highest spring tides, amplitude = ' + str(np.round(amplitude1, 2)) + ' m')
    ax.plot(x2, y2, marker='D', c='orangered',
            label='5% highest spring tides, amplitude = ' + str(np.round(amplitude2, 2)) + ' m')
    ax.plot(x3, y3, marker='v', c='olivedrab',
            label='1% highest spring tides, amplitude = ' + str(np.round(amplitude3, 2)) + ' m')

    ax.set_ylim(2.5,4)
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_fontsize(fontsize - 10)  # Set the desired font size

    ax.tick_params(axis='both', which='major', labelsize=fontsize - 5)
    plt.tight_layout()
    outfile = 'Spring_tide_per_year_HD_' + str(val_quantile) + '_and_' + str(val_quantile2) + '_' + str(
        val_quantile3) + '.png'
    fig.savefig(outfile, format='png')

#######################################
########### FIGURE ZOOM 1 YEAR ########
# Zoom 2015 2016 vives-eaux identiques
year = 2015
zoom = True
save = True
figure_zoom_wl = True
if figure_zoom_wl :
    print('HD water level df loading')
    df_HD = df_all_HD.loc[df_all_HD['Date'].dt.year == year].reset_index().drop(['index'], axis=1)
    fig, ax = plt.subplots(figsize=(20, 8))
    fig.suptitle(year, fontsize=fontsize)

    ax.grid('both', alpha=0.5)
    # ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.plot(df_HD['Datetime'], df_HD['Water level HD'], label='hourly data', color='grey',
            zorder=0.1)
    ax.scatter(time_local_min_HD, local_minima_HD, label='min values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max_HD, local_maxima_HD, label='max values', marker='o', color='red', zorder=1)
    ax.set_ylim(0, 4.5)
    ax.yaxis.set_major_locator(MultipleLocator(2))  # Set minor tick every 0.2 units on the x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 5)
    legend = fig.legend()
    for label in legend.get_texts():
        label.set_fontsize(fontsize - 5)  # Set the desired font size
    ax.set_ylabel('Water level (m)', fontsize=fontsize - 5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    date_form = DateFormatter("%d/%m")  # Define the date format
    if zoom:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
        plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax.set_xlim(datetime(year, 5, 1), datetime(year, 8, 1))
    ax.yaxis.set_major_locator(MultipleLocator(2))  # Set minor tick every 0.2 units on the x-axis
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(date_form)
    if save:
        outfile = 'water_level_HD_'
        if zoom:
            outfile = outfile + 'zoom_'
        outfile = outfile + str(year) + '.png'
        fig.savefig(outfile, format='png')

#######################################
#######################################
####        FIGURE PER YEAR      ######
#######################################
#######################################
#######################################
list_year = np.arange(2015, 2023, 1)
df_all_TT['Datetime'] = pd.to_datetime(df_all_TT['Datetime'])
df_all_Q['Date'] = pd.to_datetime(df_all_Q['Date'])
#list_year = [2017]
for year in list_year:
    print('year ', year)
    # HON DAU
    df_HD = df_all_HD.loc[df_all_HD['Date'].dt.year == year].reset_index().drop(['index'], axis = 1)

    # TT Water level
    print('TT water level df loading')
    df_TT = df_all_TT.loc[df_all_TT['Datetime'].dt.year == year].reset_index().drop(['index'], axis = 1)

    # TT Discharge
    print('TT water discharge df loading')
    df_Q = df_all_Q.loc[df_all_Q['Date'].dt.year == year].reset_index().drop(['index'], axis = 1)

    # Ebb_and_flood selection
    Ebb_and_flood_selected = Ebb_and_flood.loc[Ebb_and_flood['Datetime'].dt.year == year].reset_index().drop(['index'], axis = 1)
    Ebb_and_flood_selected_clean = Ebb_and_flood_selected.dropna(how='any', subset=['Amplitude', 'Q', 'Amplitude HD'])

    #####################################################
    #####################################################
    #### FIGURE
    #####################################################
    #####################################################
    # FIGURE 1 : données et min max
    fontsize = 35
    save = True
    zoom = False
    to_combine = True
    figure1 = False
    only_daily_discharge = True
    if figure1 :
        fig, axs = plt.subplots(figsize=(20, 12), nrows=3, sharex=True)
        fig.suptitle(year, fontsize=fontsize)

        ax = axs[0]
        ax.set_title('Trung Trang', fontsize=fontsize - 5)
        ax.grid('both', alpha=0.5)
        # ax.set_xlabel('Time', fontsize=fontsize - 5)
        ax.plot(daily_mean.index, daily_mean['Q'], label='daily mean', color='sandybrown', lw=3)
        if not only_daily_discharge :
            ax.plot(df_Q['Datetime'], df_Q['Q'], color='dodgerblue', zorder=0.1)
            ax.scatter(time_local_min_Q, local_minima_Q, label='min values', marker='o', color='black', zorder=1)
            ax.scatter(time_local_max_Q, local_maxima_Q, label='max values', marker='o', color='red', zorder=1)

        ax.set_ylim(-2000, 3500)
        ax.yaxis.set_major_locator(MultipleLocator(2000))  # Set minor tick every 0.2 units on the x-axis
        ax.yaxis.set_minor_locator(MultipleLocator(1000))
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 5)
        if to_combine:
            if year in [2015, 2016, 2017, 2018]:
                ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 7)
        else:
            legend = fig.legend()
            for label in legend.get_texts():
                label.set_fontsize(fontsize - 5)  # Set the desired font size
            ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 7)

        ax = axs[1]
        ax.set_title('Trung Trang', fontsize=fontsize - 5)
        ax.grid('both', alpha=0.5)
        # ax.set_xlabel('Time', fontsize=fontsize - 5)
        ax.plot(df_TT['Datetime'], df_TT['Water level TT'], label='hourly data', color='grey',
                zorder=0.1)
        ax.scatter(time_local_min, local_minima, label='min values', marker='o', color='black', zorder=1)
        ax.scatter(time_local_max, local_maxima, label='max values', marker='o', color='red', zorder=1)
        ax.set_ylim(-1, 3)
        ax.yaxis.set_major_locator(MultipleLocator(2))  # Set minor tick every 0.2 units on the x-axis
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 5)
        if to_combine:
            if year == 2015:
                legend = fig.legend()
                for label in legend.get_texts():
                    label.set_fontsize(fontsize - 5)  # Set the desired font size
            if year in [2015, 2016, 2017, 2018]:
                ax.set_ylabel('Water level (m)', fontsize=fontsize - 7)
        else:
            ax.set_ylabel('Water level (m)', fontsize=fontsize - 7)

        ax = axs[2]
        ax.set_title('Hon Dau', fontsize=fontsize - 5)
        ax.grid('both', alpha=0.5)
        ax.plot(df_HD['Datetime'], df_HD['Water level HD'], label='hourly water level',
                color='grey', zorder=0.1)
        ax.scatter(time_local_min_HD, local_minima_HD, label='min height values', marker='o', color='black', zorder=1)
        ax.scatter(time_local_max_HD, local_maxima_HD, label='max height values', marker='o', color='red', zorder=1)
        ax.set_ylim(0, 4.5)
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 5, pad=10)
        if to_combine:
            if year in [2015, 2016, 2017, 2018]:
                ax.set_ylabel('Water level (m)', fontsize=fontsize - 7)
            if year in [2018, 2022]:
                ax.set_xlabel('Time', fontsize=fontsize - 7)
        else:
            ax.set_ylabel('Water level (m)', fontsize=fontsize - 7)
            ax.set_xlabel('Time', fontsize=fontsize - 5)

        date_form = DateFormatter("%d/%m")  # Define the date format
        if zoom:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
            plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=5))
            ax.set_xlim(datetime(year, 5, 1), datetime(year, 8, 1))

        else:
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 1))

        ax.yaxis.set_major_locator(MultipleLocator(2))  # Set minor tick every 0.2 units on the x-axis
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.xaxis.set_major_formatter(date_form)
        fig.align_labels()
        if save:
            outfile = 'min_max_data_'
            if to_combine:
                outfile = outfile + 'tocombine_'
            if zoom:
                outfile = outfile + 'zoom_'
            outfile = outfile + str(year) + '.png'
            fig.savefig(outfile, format='png')

    fontsize = 35
    save = True
    zoom = False
    to_combine = True
    figure_onlydischarge = False
    only_daily_discharge = True
    if figure_onlydischarge :
        fig, ax = plt.subplots(figsize=(20, 8))
        fig.suptitle(year, fontsize=fontsize)

        #ax.set_title('Trung Trang', fontsize=fontsize - 5)
        ax.grid('both', alpha=0.5)
        # ax.set_xlabel('Time', fontsize=fontsize - 5)
        ax.plot(daily_mean.index, daily_mean['Q'], label='daily mean', color='sandybrown', lw=3)
        if not only_daily_discharge :
            ax.plot(df_Q['Datetime'], df_Q['Q'], color='dodgerblue', zorder=0.1)
            ax.scatter(time_local_min_Q, local_minima_Q, label='min values', marker='o', color='black', zorder=1)
            ax.scatter(time_local_max_Q, local_maxima_Q, label='max values', marker='o', color='red', zorder=1)
        if to_combine:
            if year in [2015, 2016, 2017, 2018]:
                ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
            if year in [2018, 2022]:
                ax.set_xlabel('Time', fontsize=fontsize - 5)
        else:
            ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
            ax.set_xlabel('Time', fontsize=fontsize - 5)

        date_form = DateFormatter("%d/%m")  # Define the date format
        if zoom:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
            plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=5))
            ax.set_xlim(datetime(year, 5, 1), datetime(year, 8, 1))
        else:
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 1))
        ax.tick_params(axis='both', which='major', labelsize=fontsize - 7)
        ax.yaxis.set_major_locator(MultipleLocator(1000))  # Set minor tick every 0.2 units on the x-axis
        ax.yaxis.set_minor_locator(MultipleLocator(500))
        ax.xaxis.set_major_formatter(date_form)
        fig.align_labels()
        if save:
            outfile = 'daily_discharge_'
            if to_combine:
                outfile = outfile + 'tocombine_'
            if zoom:
                outfile = outfile + 'zoom_'
            outfile = outfile + str(year) + '.png'
            fig.savefig(outfile, format='png')

    #################
    #### FIGURE 2 : TT vs HD Amplitude
    cmap = cmc.cm.batlow
    save = True
    to_combine = True
    vmax = 2500
    figure2 = False
    interp = 'polyfit' # or ''
    deb = 1000 # valeur plafond sous laquelle on veut calculer le polyfit

    if figure2:
        print("I beggin FIG 2")
        fig, ax = plt.subplots(figsize=(13, 10))
        fig.suptitle(year, fontsize=fontsize)

        ax.grid(which='both', alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        p1 = ax.scatter(abs(Ebb_and_flood_selected_clean['Amplitude HD']),
                        abs(Ebb_and_flood_selected_clean['Amplitude']),
                        c=Ebb_and_flood_selected_clean['Q'], cmap=cmap, vmin = 0, vmax=vmax, alpha=0.7)
        plt.tight_layout(rect=[0, 0.1, 0.9, 1])
        if to_combine :
            if year in [2021,2022] :#[2019,2020,2021,2022]:
                cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
                cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
                cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize - 1)
                cbar.outline.set_linewidth(0.05)
            if year in [2015,2016,2017]: #,2018]:
                ax.set_ylabel('Tidal amplitude at TT (m)', fontsize=fontsize - 2)
            if year in [2017,2020]: # [2018,2022]:
                ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize - 2)
        else :
            cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
            cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
            cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize - 1)
            cbar.outline.set_linewidth(0.05)
            ax.set_ylabel('Tidal amplitude at TT (m)', fontsize=fontsize - 2)
            ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize - 2)

        if interp == 'polyfit':
            cond = Ebb_and_flood_selected_clean['Q'] < deb
            Q = Ebb_and_flood_selected_clean.loc[cond]
            QX = Q['Amplitude HD'].dropna()
            QY = Q['Amplitude'].dropna()
            print(interp)
            coefficients = np.polyfit(QX, QY, 2)
            y_pred = np.polyval(coefficients, QX.sort_values())
            r_value, _ = pearsonr(y_pred, QY.sort_values())
            label1 = str(str(np.round(coefficients[0], 2)) + ' x${²}$ + ' +
                         str(np.round(coefficients[1], 2)) + ' x + ' + str(np.round(coefficients[2], 2)) + '\nr=' +
                         "{:.3f}".format(r_value) + ' , N=' + str(QX.count()))  # str(np.round(r_value, 3)) #" m$³$/s, "
            #label = str(label_title[c] + ' m$³$/s, r=' + "{:.3f}".format(r_value) + ' N=' + str(QX.count()))
            ax.plot(QX.sort_values(), y_pred, alpha=0.7, lw=5, color='k', zorder=1, label=label1)
        legend = ax.legend()
        for label in legend.get_texts():
            label.set_fontsize(fontsize=fontsize - 7)  # Set the desired font size

        identity = np.arange(0, 5, 1)
        ax.plot(identity, identity, c='gray')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        #plt.tight_layout(rect=[0, 0.01, 0.99, 1])
        #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
        if save:
            outfile = 'Amplitude_TT_vs_HD_Amplitude_'
            outfile = outfile + str(year) + '_' + interp
            if to_combine:
                outfile = outfile + 'to_combinev2'
            outfile = outfile + '.png'
            fig.savefig(outfile, format='png')

    ###############################
    #### FIGURE 3 : Amplification
    cmap = cmocean.cm.thermal
    to_combine = True
    s = 25
    figure3 = False
    if figure3 :
        print("My name is FIG 3")
        z = Ebb_and_flood_selected['Amplitude HD'].values
        x = Ebb_and_flood_selected['Q'].values
        amplification = Ebb_and_flood_selected['Amplitude'].values / Ebb_and_flood_selected['Amplitude HD'].values
        y = amplification

        z_clean = Ebb_and_flood_selected_clean['Amplitude HD'].values
        x_clean = Ebb_and_flood_selected_clean['Q'].values
        amplification_clean = Ebb_and_flood_selected_clean['Amplitude'].values / Ebb_and_flood_selected_clean['Amplitude HD'].values
        y_clean = amplification_clean

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(z_clean, y_clean)
        p_print = "{:.2e}".format(p_value) if p_value > 0.01 else "p<0.01"
        p_print2 = "{:.2e}".format(p_value2) if p_value > 0.01 else "p<0.01"
        label = 'r = ' + str(np.round(r_value, 2)) + ' , ' + p_print
        label = label + "\ny = {:.2e}".format(slope) + ' x + ' + "{:.2e}".format(intercept)
        #label = label + '\nCorrelation with TR: r = ' + str(np.round(r_value2, 2)) + ' , ' + p_print2
        #label = label + "\ny = {:.2e}".format(slope2) + ' x + ' + "{:.2e}".format(intercept2)

        fig, ax = plt.subplots(figsize=(15, 10))  # , ncols=2, gridspec_kw={'width_ratios': [25, 1]})
        fig.suptitle(year, fontsize=fontsize)
        # ax = axs[0]
        ax.grid('both', alpha=0.5)
        # ax.set_title('2017 ' + a + str(month), fontsize = fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        im = ax.scatter(x_clean, y_clean, lw=0.5, c=z_clean, cmap=cmap, vmin=0, vmax=4, s=s)
        # im = ax.scatter(x, y, lw=0.5, c=localisation[sta]['color'], marker=list_marker[i], label=sta)
        x_discharge = np.arange(0, 3600, 100)
        ax.plot(x_discharge, slope * x_discharge + intercept, color='gray', lw=5, zorder=0, label=label)
        ax.set_xlim(0, 3200)
        ax.set_ylim(0, 2)
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(500))
        ax.yaxis.set_major_locator(MultipleLocator(1))  # Set minor tick every 0.2 units on the x-axis
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.set_aspect(1200)
        legend = ax.legend()
        for label in legend.get_texts():
            label.set_fontsize(fontsize=fontsize - 7)  # Set the desired font size
        #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))

        if to_combine:
            if year in [2021,2022]:#[2019, 2020, 2021, 2022]:
                cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                cbar = fig.colorbar(im, cax=cbar_ax)
                # cbar = fig.colorbar(im, ax=axs[1])
                cbar.ax.tick_params(labelsize=fontsize, width=0.5)
                cbar.set_label(label='Tidal amplitude at HD (m)', fontsize=fontsize)
                cbar.outline.set_linewidth(0.05)

            if year in [2015, 2016, 2017] :#, 2018]:
                # axs[1].axis('off')
                ax.set_ylabel('Amplification', fontsize=fontsize)
            if year in [2017, 2020]:
                ax.set_xlabel('Discharge (m³/s)', fontsize=fontsize)
        else:
            # axs[1].axis('off')
            ax.set_xlabel('Discharge (m³/s)', fontsize=fontsize)
            ax.set_ylabel('Amplification', fontsize=fontsize)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar = fig.colorbar(im, ax=axs[1])
            cbar.ax.tick_params(labelsize=fontsize, width=0.5)
            cbar.set_label(label='Tidal amplitude at HD (m)', fontsize=fontsize)
            cbar.outline.set_linewidth(0.05)

        if save:
            outfile = 'Attenuation_vs_discharge_'
            if to_combine:
                outfile = outfile + 'to_combine_v2_'
            outfile = outfile + str(year) + '.png'
            fig.savefig(outfile)

    figure_attenuation_TR_year = True
    to_combine = True
    filter = True # calcule l'atténuation que pour les débits < à un seuil (<1000)
    if figure_attenuation_TR_year:
        print("My name is FIG 3")
        z = Ebb_and_flood_selected['Amplitude HD'].values
        x = Ebb_and_flood_selected['Q'].values
        amplification = Ebb_and_flood_selected['Amplitude'].values / Ebb_and_flood_selected['Amplitude HD'].values
        y = amplification

        if filter :
            seuil_Q = 1000
            condition = Ebb_and_flood_selected_clean['Q'] <= seuil_Q
            z_clean = Ebb_and_flood_selected_clean['Amplitude HD'].loc[condition].values
            x_clean = Ebb_and_flood_selected_clean['Q'].loc[condition].values
            amplification_clean = Ebb_and_flood_selected_clean['Amplitude'].loc[condition].values / \
                                  Ebb_and_flood_selected_clean['Amplitude HD'].loc[condition].values
            y_clean = amplification_clean
        else:
            z_clean = Ebb_and_flood_selected_clean['Amplitude HD'].values
            x_clean = Ebb_and_flood_selected_clean['Q'].values
            amplification_clean = Ebb_and_flood_selected_clean['Amplitude'].values / Ebb_and_flood_selected_clean['Amplitude HD'].values
            y_clean = amplification_clean

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(z_clean, y_clean)
        p_print = "{:.2e}".format(p_value) if p_value > 0.01 else "p<0.01"
        p_print2 = "{:.2e}".format(p_value2) if p_value > 0.01 else "p<0.01"
        # label = 'Correlation with Q: r = ' + str(np.round(r_value, 2)) + ' , ' + p_print
        # label = label + "\ny = {:.2e}".format(slope) + ' x + ' + "{:.2e}".format(intercept)
        label = 'r = ' + str(np.round(r_value2, 2)) + ' , ' + p_print2
        label = label + "\ny = {:.2e}".format(slope2) + ' x + ' + "{:.2e}".format(intercept2)
        label = label + "\nN= " + str(len(x_clean))

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(year, fontsize=fontsize)
        ax.grid('both', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        im = ax.scatter(z, y, lw=0.5, c=x, cmap=cmap, vmin=0, vmax=3200, s=s)
        # im = ax.scatter(x, y, lw=0.5, c=localisation[sta]['color'], marker=list_marker[i], label=sta)
        x_TR = np.arange(0, 4.5, 0.1)
        ax.plot(x_TR, slope2 * x_TR + intercept2, color='gray', lw=5, zorder=0, label=label)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(1))  # Set minor tick every 0.2 units on the x-axis
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.set_aspect(1.700)
        legend = ax.legend()
        for label in legend.get_texts():
            label.set_fontsize(fontsize=fontsize - 7)  # Set the desired font size

        plt.tight_layout(rect=[0, 0.05, 0.9, 1])

        if to_combine:
            if year in [2021,2022]:#[2019, 2020, 2021, 2022]:
                cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
                cbar = fig.colorbar(im, cax=cbar_ax)
                # cbar = fig.colorbar(im, ax=axs[1])
                cbar.ax.tick_params(labelsize=fontsize, width=0.5)
                cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize)
                cbar.outline.set_linewidth(0.05)
            if year in [2015, 2016, 2017] :#, 2018]:
                # axs[1].axis('off')
                ax.set_ylabel('Amplification', fontsize=fontsize)
            if year in [2017, 2020]:
                ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize)
        else:
            # axs[1].axis('off')
            ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize)
            ax.set_ylabel('Amplification', fontsize=fontsize)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax)
            # cbar = fig.colorbar(im, ax=axs[1])
            cbar.ax.tick_params(labelsize=fontsize, width=0.5)
            cbar.set_label(label='Tidal amplitude at HD (m)', fontsize=fontsize)
            cbar.outline.set_linewidth(0.05)



        if save:
            outfile = 'Attenuation_vs_TR_'
            if filter :
                outfile = outfile + 'filtered_'
            outfile = outfile + str(year) + '.png'
            fig.savefig(outfile)


print('ok gogole')


########################
#########################"

# Je regarde si la variation de débit est corrélée avec la variation du signal résidule à HD
import utide
# Sur le notebook exemple de Utide, ajout de l'anomalie
df_HD_2015_2022["anomaly"] = df_HD_2015_2022["Water level HD"] - df_HD_2015_2022["Water level HD"].mean()
df_HD_2015_2022["anomaly"] = df_HD_2015_2022["anomaly"].interpolate() # normalement je n'ai quasi pas de np.nan
merged_df_HD_Q = df_HD_2015_2022.merge(df_all_Q, how='left',
                            left_on='Datetime', right_on='Datetime')
merged_df_HD_Q = merged_df_HD_Q.dropna(subset=['anomaly', 'Q'])

coef_mc = utide.solve(
    merged_df_HD_Q['Datetime'],
    merged_df_HD_Q["anomaly"],
    lat=20.66,
    method="ols", # ols ordinary least square ou wls weighted ls
    conf_int="MC", # Monte Carlo to calculate the confidence interval
    MC_n=1000, # necessary to explicit the number of realisation
    verbose=True,
)


# Reconstruit le signal de marée
tide = utide.reconstruct(merged_df_HD_Q['Datetime'], coef_mc, verbose=False)
print(tide.keys())

x = merged_df_HD_Q['anomaly'] - tide.h
y = merged_df_HD_Q['Q']
correlation_coefficient, p_value = stats.pearsonr(x, y)


"""
# Pour l'année 2009# file = rep + 'HonDau_water_level_en.xlsx'
file = rep + 'HonDau_water_level_en.xlsx'

df_all_HD = pd.DataFrame(columns=['Datetime', 'Date', 'Water level HD'])
list_month = np.arange(1, 13)
nrows = 31
skip = 4  # Correspond à l'en tête
skip_new = 4  # 1364 si on commence en 2012
skip2 = 2  # correspond au nombre de ligne entre les tableaux
columns_to_load = list(range(25))
for month in list_month:
    df = pd.read_excel(file, sheet_name='09', skiprows=skip_new, nrows=nrows, usecols=columns_to_load)
    # print(month, '\n', df[0:2]) #, df[-3:])
    df = df.rename(columns={'date': 'Day'})  # Je renomme les colonnes
    df = df.rename(columns={'Unnamed: ' + str(i): str(i - 1) for i in
                            range(0, 25)})  # Y compris celles qui sont en mois en chiffre
    df['Year'] = year
    df['Month'] = month
    # print(month, '\n', df[0:2], df[-3:])

    melted_df = pd.melt(df, id_vars=['Year', 'Month', 'Day'], value_vars=[f'{i}' for i in range(24)],
                        var_name='Hour', value_name='Water level HD')

    # Drop rows with NaN values (these correspond to days that do not exist in shorter months)
    melted_df = melted_df.dropna(subset=['Water level HD'])
    # Sort the values by Year, Month, and Day to maintain chronological order
    melted_df = melted_df.sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)
    melted_df['Datetime'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day', 'Hour']])
    melted_df['Date'] = pd.to_datetime(melted_df[['Year', 'Month', 'Day']])
    melted_df = melted_df[['Datetime', 'Date', 'Water level HD']]

    df_all_HD = pd.concat([df_all_HD, melted_df], ignore_index=True)
    skip_new = skip_new + skip + nrows + skip2 + 1


### TRUNG TRANG
file = rep + 'TrungTrang_Q_SPM_2008-2016_en.xlsx'

columns_to_load = ['Month', 'Date', 'hour', 'H']
# Water level at Hon Dau
df = pd.read_excel(file, sheet_name='Q-2009', usecols=columns_to_load, skiprows=3, nrows=1492)
df = df.rename(columns={'Unnamed: 0': 'Date'})
melted_df = pd.melt(df, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df['Datetime'] = pd.to_datetime(melted_df['Date']) + pd.to_timedelta(melted_df['Hour'], unit='h')
melted_df.sort_values("Datetime", inplace=True)
melted_df = melted_df.rename(columns={'Value': 'Water level Trung Trang'})
melted_df.drop(['Date', 'Hour'], axis=1, inplace=True)
"""

