# 02/08/2024 : pour 2017 pour comparer au modèle
# 29/05/24 : Je fais une nouvelle page propre dédiée aux lags, donc je ne calcule que les HW, LW, HWS, LWS, Vmin , Vmax.

import numpy as np
import csv
import pandas as pd
import numpy as np
import cmcrameri as cmc
import seaborn as sns
import xarray as xr
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib as mpl
from openpyxl import load_workbook
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
mpl.use('Agg')


# COnvention : 100% is HT, 0% is LT, + from L to HT and - from H to LT.
# convention 1 : From HT to LT, we begin by -99% near to HT, -1% near to LT.  CIRCLE
# Trouble convention 1 : representation on graphs ..

# Functions :*
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
        window = df[col].loc[i:i + window_size]
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
            test_min = df[col].loc[df['Datetime'] == time_local_min[i] + timedelta(hours=window_size)].values[0]
            test_max = df[col].loc[df['Datetime'] == time_local_max[i] + timedelta(hours=window_size)].values[0]
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


# Calculation of the lag correlation
def calculate_corr(data1, data2, year_constraint, year, month_constraint, month, datetime=True):
    correlations = []
    if datetime:
        max_lag = 5  # Maximum lag value to test
        if year_constraint and month_constraint:
            df1 = data1[(data1['Datetime'].dt.year == year) & (data1['Datetime'].dt.month == month)]
            df2 = data2[(data2['Datetime'].dt.year == year) & (data2['Datetime'].dt.month == month)]
        elif year_constraint:
            df1 = data1[(data1['Datetime'].dt.year == year)]
            df2 = data2[(data2['Datetime'].dt.year == year)]
        elif month_constraint:
            df1 = data1[data1['Datetime'].dt.month == month]
            df2 = data2[data2['Datetime'].dt.month == month]
        else:
            print('no constraint on year or month, I take the whole series')
            df1 = data1
            df2 = data2
    else:
        max_lag = 50 # Si série interpolée à 1mn, 200 pour calculer le lag renverse des courants, max de hauteur,
        # 50 pour
        if year_constraint and month_constraint:
            df1 = data1[(data1.index.year == year) & (data1.index.month == month)]
            df2 = data2[(data2.index.year == year) & (data2.index.month == month)]
        elif year_constraint:
            df1 = data1[(data1.index.year == year)]
            df2 = data2[(data2.index.year == year)]
        elif month_constraint:
            df1 = data1[data1.index.month == month]
            df2 = data2[data2.index.month == month]
        else:
            print('no constraint on year or month, I take the whole series')
            df1 = data1
            df2 = data2
    for lag in range(-max_lag, max_lag + 1):
        shifted_df2 = df2.shift(periods=lag)
        if datetime:
            m2 = pd.concat([df1.reset_index().drop('index', axis=1), shifted_df2.reset_index().drop('index', axis=1)],
                           axis=1)
            # correlation = m2['Q'].corr(m2['Q (m$^3$/s)'])
        else:
            m2 = pd.concat([df1, shifted_df2], axis=1)
        correlation = m2.corr().iloc[0, 1]
        correlations.append((lag, correlation))
    # Find the lag with the highest correlation
    best_lag, best_correlation = max(correlations, key=lambda x: abs(x[1]))
    # print(f"Best lag: {best_lag}")
    # print(f"Best correlation: {best_correlation}")
    return best_lag, best_correlation
    # ON WATER LEVEL :
    # For the whole temporal serie  Best lag: 2 Best correlation: 0.9403780905327238
    # For 2022 : Best lag: 2 Best correlation: 0.9231507853916974
    # Monthly : worst for May to Sept, but still > 0.87
    # For january, Best lag: 2 Best correlation: 0.974414815770983 febr,Lag: 2 Corr: 0.9841633877568988 ,  march, Lag: 2 Corr: 0.98359689458487
    # april, Lag: 2 Corr: 0.985 may, Lag: 2 Corr: 0.940 june, Lag: 2 Corr: 0.874 july lag: 2 correlation: 0.8816781512215971
    # Aug, Lag: 2 Corr: 0.881 Sept, Lag: 2 Corr: 0.957 Oct, Lag: 2 Corr: 0.953  Nov  Lag: 2 Corr: 0.979 Dec, Lag: 2 Corr: 0.980


def calculate_corr2(data1, data2, year_constraint, year, month_constraint, month, datetime=True):
    correlations = []
    if datetime:
        max_lag = 5  # Maximum lag value to test
        if year_constraint and month_constraint:
            df1 = data1[(data1['Datetime'].dt.year == year) & (data1['Datetime'].dt.month == month)]
            df2 = data2[(data2['Datetime'].dt.year == year) & (data2['Datetime'].dt.month == month)]
        elif year_constraint:
            df1 = data1[(data1['Datetime'].dt.year == year)]
            df2 = data2[(data2['Datetime'].dt.year == year)]
        elif month_constraint:
            df1 = data1[data1['Datetime'].dt.month == month]
            df2 = data2[data2['Datetime'].dt.month == month]
        else:
            print('no constraint on year or month, I take the whole series')
            df1 = data1
            df2 = data2
    else:
        max_lag = 100 # Si série interpolée à 1mn, 200 pour calculer le lag renverse des courants, max de hauteur,
        # 50 pour
        if year_constraint and month_constraint:
            df1 = data1[(data1.index.year == year) & (data1.index.month == month)]
            df2 = data2[(data2.index.year == year) & (data2.index.month == month)]
        elif year_constraint:
            df1 = data1[(data1.index.year == year)]
            df2 = data2[(data2.index.year == year)]
        elif month_constraint:
            df1 = data1[data1.index.month == month]
            df2 = data2[data2.index.month == month]
        else:
            print('no constraint on year or month, I take the whole series')
            df1 = data1
            df2 = data2
    for lag in range(-max_lag, max_lag):
        shifted_df2 = df2.shift(periods=lag)
        if datetime:
            m2 = pd.concat([df1.reset_index().drop('index', axis=1), shifted_df2.reset_index().drop('index', axis=1)],
                           axis=1)
            # correlation = m2['Q'].corr(m2['Q (m$^3$/s)'])
        else:
            m2 = pd.concat([df1, shifted_df2], axis=1)
        correlation = m2.corr().iloc[0, 1]
        correlations.append((lag, correlation))
    # Find the lag with the highest correlation
    best_lag, best_correlation = max(correlations, key=lambda x: (x[1]))
    # print(f"Best lag: {best_lag}")
    # print(f"Best correlation: {best_correlation}")
    return best_lag, best_correlation
    # ON WATER LEVEL :
    # For the whole temporal serie  Best lag: 2 Best correlation: 0.9403780905327238
    # For 2022 : Best lag: 2 Best correlation: 0.9231507853916974
    # Monthly : worst for May to Sept, but still > 0.87
    # For january, Best lag: 2 Best correlation: 0.974414815770983 febr,Lag: 2 Corr: 0.9841633877568988 ,  march, Lag: 2 Corr: 0.98359689458487
    # april, Lag: 2 Corr: 0.985 may, Lag: 2 Corr: 0.940 june, Lag: 2 Corr: 0.874 july lag: 2 correlation: 0.8816781512215971
    # Aug, Lag: 2 Corr: 0.881 Sept, Lag: 2 Corr: 0.957 Oct, Lag: 2 Corr: 0.953  Nov  Lag: 2 Corr: 0.979 Dec, Lag: 2 Corr: 0.980


def calculate_categories_mean(df):
    cond1 = (df['Q'] <= 500)
    cond2 = (500 < df['Q']) & (df['Q'] <= 1000)
    cond3 = (1000 < df['Q']) & (df['Q'] <= 1500)
    cond4 = (df['Q'] > 1500)

    condA = (abs(df['Amplitude HD']) < 100)
    condB = (100 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 200)
    condC = (200 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 300)
    condD = (abs(df['Amplitude HD']) > 300)

    list_val_TT, list_val_HD, list_len = [], [], []
    situation_debit = [cond1, cond2, cond3, cond4]
    situation_amplitude = [condA, condB, condC, condD]
    for deb in situation_debit:
        for amp in situation_amplitude:
            mean_TT = np.nanmean(df['Amplitude'].loc[deb & amp])
            mean_TT = np.round(mean_TT / 100, 2)  # Mean in meters
            list_val_TT.append(mean_TT)
            mean_HD = np.nanmean(df['Amplitude HD'].loc[deb & amp])
            mean_HD = np.round(mean_HD / 100, 2)
            list_val_HD.append(mean_HD)
            list_len.append(len(df['Amplitude HD'].loc[deb & amp]))  # number of values corresponding to the conds
    return list_val_HD, list_val_TT, list_len


def calculate_quartiles_mean(df):
    condA = (abs(df['Amplitude HD']) < 100)
    condB = (100 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 200)
    condC = (200 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 300)
    condD = (abs(df['Amplitude HD']) > 300)

    list_val_TT, list_val_HD, list_len, list_debit = [], [], [], []
    situation_amplitude = [condA, condB, condC, condD]
    for amp in situation_amplitude:
        deb = df.loc[amp].quantile([0.25, 0.5, 0.75])['Q'].values
        cond1 = (df['Q'] < deb[0])
        cond2 = (deb[0] < df['Q']) & (df['Q'] <= deb[1])
        cond3 = (deb[1] < df['Q']) & (df['Q'] <= deb[2])
        cond4 = (df['Q'] > deb[2])
        for i in range(len(deb)):
            list_debit.append(np.round(deb[i], 2))
        list_debit.append(np.nan)  # In order to have the same length as other lists
        for debs in [cond1, cond2, cond3, cond4]:
            mean_TT = np.nanmean(df['Amplitude'].loc[amp & debs])
            mean_TT = np.round(mean_TT / 100, 2)  # Mean in meters
            list_val_TT.append(mean_TT)
            mean_HD = np.nanmean(df['Amplitude HD'].loc[amp & debs])
            mean_HD = np.round(mean_HD / 100, 2)
            list_val_HD.append(mean_HD)
            list_len.append(len(df['Amplitude HD'].loc[amp & debs]))  # number of values corresponding to the conds
    return list_val_HD, list_val_TT, list_len, list_debit


def logarithmic_function(x, a, b):
    return a * np.log(x) + b


# Function to find the closest datetime in df1 for each datetime in df2
def find_closest_datetime(dt, datetime_list):
    closest_dt = min(datetime_list, key=lambda x: abs((x - dt).total_seconds()))
    return closest_dt


def calculate_percentage_old(case, Ebb_data, Flood_data, interpolated_series):
    # objectif : calculate on the interpolated timeseries the percentage of tide corresponding
    if case == 'max': # the first data is a maximum, determine for the whole series df1 and df2
        df1 = time_local_max
        df2 = time_local_min
    elif case == 'min':
        df2 = time_local_max
        df1 = time_local_min
    percentage=[]
    for i in range(len(interpolated_series)):
        time_sta =  interpolated_series.index[i]
        closest_ebb = find_closest_datetime(time_sta, Ebb_data['Datetime'])  # Attention : c'est ok pour spring
        # tide, mais pas pour neap, ou il faudrait trouver les valeurs qui bornent la datetime.
        closest_flood = find_closest_datetime(time_sta, Flood_data['Datetime'])
        if (time_sta <= closest_flood) & (time_sta >= closest_ebb):
            print(' je suis a priori dans une phase de jusant')
            duration_ebb = Ebb_data['Duration'].loc[Ebb_data['Datetime'] == closest_ebb]
            from_start = time_sta - closest_ebb
            percentage = - (100 - (from_start / duration_ebb * 100))
        elif (time_sta <= closest_ebb) & (time_sta >= closest_flood):
            print('Je suis a priori dans une phase de FLOT')
            duration_flood = Flood_data['Duration'].loc[Flood_data['Datetime'] == closest_flood]
            from_start = time_sta - closest_flood
            percentage = from_start / duration_flood * 100
        else:
            print('PB ON EST HORS DU CADRE D2FINI, les closest time ne bornent pas la donnée')
            # check of the closest datetime, in order to determine the other one in a specific place so the datetime is inbetween
            closest_both = find_closest_datetime(time_sta, [closest_ebb, closest_flood])
            print('closest both ', closest_both)
            # Control of the closest
            if (closest_both > time_sta) & (closest_both == closest_flood):
                print('Case 1 : Je recalcule le ebb avant le datetime station \n je suis en ebb')
                closest_ebb = find_closest_datetime(time_sta,
                                                    Ebb_data['Datetime'].loc[Ebb_data['Datetime'] <= time_sta])
                duration_ebb = Ebb_data['Duration'].loc[Ebb_data['Datetime'] == closest_ebb]
                from_start = time_sta - closest_ebb
                percentage = - (100 - (from_start / duration_ebb * 100))
            elif (closest_both > time_sta) & (closest_both == closest_ebb):
                print('Case 2 : Je recalcule le flood avant le datetime station \n je suis en flood')
                closest_flood = find_closest_datetime(time_sta,
                                                      Flood_data['Datetime'].loc[
                                                          Flood_data['Datetime'] <= time_sta])
                duration_flood = Flood_data['Duration'].loc[Flood_data['Datetime'] == closest_flood]
                from_start = time_sta - closest_flood
                percentage = from_start / duration_flood * 100
            elif (closest_both < time_sta) & (closest_both == closest_ebb):
                print(
                    'Case 3 : Je recalcule le flood, je suis en ebb')  # Not useful to calculate the closest flood, but allow to manually cjeck
                closest_flood = find_closest_datetime(time_sta,
                                                      Flood_data['Datetime'].loc[
                                                          Flood_data['Datetime'] >= time_sta])
                duration_ebb = Ebb_data['Duration'].loc[Ebb_data['Datetime'] == closest_ebb]
                from_start = time_sta - closest_ebb
                percentage = - (100 - (from_start / duration_ebb * 100))
            elif (closest_both < time_sta) & (closest_both == closest_flood):
                print(
                    'Case 4 : Je recalcule le Ebb, je suis en flood')  # Not useful to calculate the closest flood, but allow to manually cjeck
                closest_ebb = find_closest_datetime(time_sta,
                                                    Ebb_data['Datetime'].loc[Ebb_data['Datetime'] >= time_sta])
                duration_flood = Flood_data['Duration'].loc[Flood_data['Datetime'] == closest_flood]
                from_start = time_sta - closest_flood
                percentage = from_start / duration_flood * 100
                break
            else:
                print('NO comprendo')
                break
        print(percentage)
        list_percentage.append(percentage.values[0])
        list_tide.append(df_good_time['Tide'].values[0])


def calculate_percentage(time_local_min, time_local_max, interpolated_series):
    # objectif : calculate on the interpolated timeseries the percentage of tide corresponding
    interpolated_series['Percentage of tide'] = np.nan
    for i in range(len(time_local_min)):
        interpolated_series['Percentage of tide'].loc[(interpolated_series.index == time_local_max[i])] = 100
        interpolated_series['Percentage of tide'].loc[(interpolated_series.index == time_local_min[i])] = 0
    interpolated_series['Percentage of tide'] = interpolated_series['Percentage of tide'].interpolate(method='linear')
    mask = interpolated_series['Percentage of tide'] > interpolated_series['Percentage of tide'].shift(-1)
    interpolated_series['Percentage of tide'] = np.where(mask, -interpolated_series['Percentage of tide'], interpolated_series['Percentage of tide'])
    return interpolated_series['Percentage of tide']


def handle_outliers(df, column_name, factor=1.5):
    # Calculate the IQR (Interquartile Range)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Replace the outliers with the median value
    df2 = df.copy()
    df2[column_name] = df2[column_name].apply(lambda x: x if lower_bound <= x <= upper_bound else np.median(x))
    #df2[column_name].median())
    return df2[column_name]


def nearest_future_slack_water(current_time, time_local_SW):
    future_times = [dt for dt in time_local_SW if dt > current_time]
    return min(future_times, key=lambda dt: dt - current_time) if future_times else None

# Function to find the nearest past slack water time
def nearest_past_slack_water(current_time, time_local_SW):
    past_times = [dt for dt in time_local_SW if dt < current_time]
    return min(past_times, key=lambda dt: current_time - dt) if past_times else None


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    # Return colormap object.
    return mpl.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


print('Hello ! I beggin the code ...')



# 1. Je détecte les max et min et calcule les durées à Hon Dau et à Trung Trang
# VARIABLE AND PARAMETER
# FIGURE PARAMETER
fontsize = 28
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 4
s = 25
# LOAD THE DATA
file = '/home/penicaud/Documents/Data/Décharge_waterlevel/Data_2017.xlsx'

# Water level at Trung Trang
df = pd.read_excel(file, sheet_name='Q_trungtrang_vanuc_2017',
                   usecols=['Date', 'Time (hours)', 'Water level (cm)', 'Q (m3/s)' ], skiprows=3) #(m3/s)'
df['Datetime'] = df['Date'] + pd.to_timedelta(df['Time (hours)'], unit='h')
df.drop(['Time (hours)'], axis=1, inplace=True)
df = df.rename(columns={'Water level (cm)': 'Water level TT', 'Q (m3/s)':'Q'})

# Water level at Hon Dau
df2 = pd.read_excel(file, sheet_name='Water_level_HonDau2017', usecols=['Datetime', 'Value'])
df2 = df2.rename(columns={'Value': 'Water level HD'})

water_levels = pd.merge(df, df2, on='Datetime', how='inner')
water_levels = water_levels[['Date', 'Datetime', 'Q', 'Water level TT', 'Water level HD']]

monthly_mean = water_levels.resample('M', on='Datetime').mean()
daily_mean = water_levels.resample('D', on='Datetime').mean()

#################################################################################################
print('Hello again ! I try to find the ebb and flood at TT ...')
# I calculate the min and max of the tides AT TT
# HW and LW AT TT
window_size = 17
local_minima, local_maxima, time_local_min, time_local_max, a, b = \
    find_local_minima(water_levels, 'Water level TT', window_size)
df_LW = pd.DataFrame(time_local_min)
df_HW = pd.DataFrame(time_local_max)
df_LW = df_LW.rename(columns={0: 'Datetime LW'})
df_HW = df_HW.rename(columns={0: 'Datetime HW'})
# 17/01/24 : I add the SW, but it needs to be done on the High frequency, this is why a and b are not exploited arrays

# Amplitude à TT
Ebb = pd.DataFrame(time_local_max)  # the starting datetime is the beginning of the ebb i.e : max water levels at TT
Flood = pd.DataFrame(time_local_min)
Ebb = Ebb.rename(columns={0: 'Datetime'})
Flood = Flood.rename(columns={0: 'Datetime'})
if time_local_max[0] > time_local_min[0]:  # To know which one we need to substract
    print('The first extremum is the minimum data, so it is the flood')
    Flood['Duration'] = time_local_max - time_local_min
    Flood['Amplitude'] = local_maxima - local_minima
    Ebb['Duration'] = np.roll(time_local_min, shift=-1) - time_local_max
    Ebb['Amplitude'] = np.roll(local_minima, shift=-1) - local_maxima
    Ebb.loc[len(Ebb)-1, 'Duration'] = np.nan
    Ebb.loc[len(Ebb)-1, 'Amplitude'] = np.nan
else:
    print('The first extremum is the MAX data, so it is the Ebb')
    Flood['Duration'] = np.roll(time_local_max, shift=-1) - time_local_min
    Flood['Amplitude'] = np.roll(local_maxima, shift=-1) - local_minima
    Flood.loc[len(Flood)-1, 'Duration'] = np.nan
    Flood.loc[len(Flood)-1, 'Amplitude'] = np.nan
    Ebb['Duration'] = time_local_min - time_local_max
    Ebb['Amplitude'] = local_minima - local_maxima


# AT HD : to have the tidal amplitude without damping
col2 = 'Water level HD'
local_minima_HD, local_maxima_HD, time_local_min_HD, time_local_max_HD, a, b = \
    find_local_minima(water_levels, col2, window_size)

df_LW_HD = pd.DataFrame(time_local_min_HD)
df_HW_HD = pd.DataFrame(time_local_max_HD)
df_LW_HD = df_LW_HD.rename(columns={0: 'Datetime LW HD'})
df_HW_HD = df_HW_HD.rename(columns={0: 'Datetime HW HD'})

Ebb_HD = pd.DataFrame(
    time_local_max_HD)  # the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
Flood_HD = pd.DataFrame(time_local_min_HD)
Ebb_HD = Ebb_HD.rename(columns={0: 'Datetime'})
Flood_HD = Flood_HD.rename(columns={0: 'Datetime'})
if time_local_max_HD[0] > time_local_min_HD[0]:  # To know which one we need to substract
    print('The first extremum is the minimum data, so it is the flood_HD')
    Flood_HD['Duration HD'] = time_local_max_HD - time_local_min_HD
    Flood_HD['Amplitude HD'] = local_maxima_HD - local_minima_HD
    Ebb_HD['Duration HD'] = np.roll(time_local_min_HD, shift=-1) - time_local_max_HD
    Ebb_HD['Amplitude HD'] = np.roll(local_minima_HD, shift=-1) - local_maxima_HD
    Ebb_HD.loc[len(Ebb_HD)-1, 'Duration HD'] = np.nan
    Ebb_HD.loc[len(Ebb_HD)-1, 'Amplitude HD'] = np.nan
else:
    print('The first extremum is the MAX data, so it is the Ebb_HD')
    Flood_HD['Duration HD'] = np.roll(time_local_max_HD, shift=-1) - time_local_min_HD
    Flood_HD['Amplitude HD'] = np.roll(local_maxima_HD, shift=-1) - local_minima_HD
    Flood_HD.loc[len(Flood_HD)-1, 'Duration HD'] = np.nan
    Flood_HD.loc[len(Flood_HD)-1, 'Amplitude HD'] = np.nan
    Ebb_HD['Duration HD'] = time_local_min_HD - time_local_max_HD
    Ebb_HD['Amplitude HD'] = local_minima_HD - local_maxima_HD

# Vmin and Vmax
# Min and Max Q i.e. Vmax and Vmin.
local_minima_Q, local_maxima_Q, time_local_min_Q, time_local_max_Q, a, b = \
    find_local_minima(water_levels, 'Q', window_size)
df_min_discharge = pd.DataFrame(time_local_min_Q)
# the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
df_max_discharge = pd.DataFrame(time_local_max_Q)
df_min_discharge = df_min_discharge.rename(columns={0: 'Datetime Vmin'})
df_max_discharge = df_max_discharge.rename(columns={0: 'Datetime Vmax'})


figure = False
if figure :
    fig, axs = plt.subplots(figsize=(18, 10), nrows=3, sharex=True)
    fig.suptitle('2017', fontsize=fontsize)
    ax = axs[0]
    ax.set_title('Trung Trang', fontsize=fontsize - 5)
    ax.grid('both', alpha=0.5)
    # ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels['Q'], color='blue', zorder=0.1)
    ax.plot(daily_mean.index, daily_mean['Q'], label='daily mean', color='violet')
    ax.scatter(time_local_min_Q, local_minima_Q, label='min values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max_Q, local_maxima_Q, label='max values', marker='o', color='red', zorder=1)
    fig.legend()

    ax = axs[1]
    ax.set_title('Trung Trang', fontsize=fontsize - 5)
    ax.grid('both', alpha=0.5)
    # ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Water level (m)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels['Water level TT'] / 100, label='hourly data', color='grey',
            zorder=0.1)
    ax.scatter(time_local_min, local_minima / 100, label='min values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, label='max values', marker='o', color='red', zorder=1)
    # fig.legend()

    ax = axs[2]
    ax.set_title('Hon Dau', fontsize=fontsize - 5)
    ax.grid('both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Water level (m)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels['Water level HD'] / 100, label='hourly water level',
            color='grey', zorder=0.1)
    ax.scatter(time_local_min_HD, local_minima_HD / 100, label='min height values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max_HD, local_maxima_HD / 100, label='max height values', marker='o', color='red', zorder=1)
    date_form = DateFormatter("%d/%m")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xlim(datetime(2017, 1, 1), datetime(2018, 1, 1))
    fig.align_labels()
    fig.savefig('min_max_2017_data.png', format='png')


# SW detection with the resampled serie.
val_interp = 5
resampled_series = water_levels.copy()
resampled_series = resampled_series.set_index('Datetime')
resampled_series = resampled_series.resample(str(val_interp)+'T').asfreq()  # Resample by adding values every 5T
#interpolated_series = resampled_series.interpolate(method='linear')
interpolated_series = resampled_series.copy()
interpolated_series[['Water level TT', 'Water level HD']] =\
    resampled_series[['Water level TT', 'Water level HD']].interpolate(method='linear')


window_size_interp = int(17*60/val_interp)
local_minima_Q_interp, local_maxima_Q_interp, time_local_min_Q_interp, time_local_max_Q_interp, \
local_SW, time_local_SW = \
    find_local_minima(interpolated_series.reset_index(), interpolated_series.reset_index().columns[2],
                      window_size_interp, interp=True)

df_SW = pd.DataFrame(time_local_SW)
df_SW = df_SW.rename(columns={0: 'Datetime SW'})

if df_max_discharge['Datetime Vmax'].iloc[0] < df_min_discharge['Datetime Vmin'].iloc[0] :
    combined_df = pd.concat([df_max_discharge, df_min_discharge], axis=1)
else :
    combined_df = pd.concat([df_min_discharge, df_max_discharge], axis=1)

# Ensure the dataframes are sorted by the datetime columns
df_max_discharge = df_max_discharge.sort_values('Datetime Vmax').reset_index(drop=True)
df_min_discharge = df_min_discharge.sort_values('Datetime Vmin').reset_index(drop=True)
df_SW = df_SW.sort_values('Datetime SW').reset_index(drop=True)

# Step 1 : insert the slack water inbetween the Vmax and Vmin
# Merge zero times with min times to find the zero velocity time just after each min velocity time
SW_after_max = pd.merge_asof(df_max_discharge,  df_SW,
    left_on='Datetime Vmax', right_on='Datetime SW', direction='forward')

# Filter the SW_after_max DataFrame to ensure slack water times are before the next Vmin
SW_after_max['Next Vmin'] = df_min_discharge['Datetime Vmin']#.shift(-1, fill_value=pd.Timestamp.max)
filtered_SW_after_max = SW_after_max[SW_after_max['Datetime SW'] < SW_after_max['Next Vmin']]

# Insert the filtered slack water times into the combined DataFrame
combined_df['LWS'] = filtered_SW_after_max['Datetime SW']
combined_df = combined_df[['Datetime Vmax', 'LWS', 'Datetime Vmin']] #Slack water after Vmax is LWS.
# LWS = Slack Water After Vmax

# Step 2 : insert the slack water inbetween the Vmin and the next Vmax
SW_after_min = pd.merge_asof(df_min_discharge,  df_SW,
    left_on='Datetime Vmin', right_on='Datetime SW', direction='forward')

# Filter the SW_after_max DataFrame to ensure slack water times are before the next Vmin
SW_after_min['Next Vmax'] = df_max_discharge['Datetime Vmax'].shift(-1, fill_value=pd.Timestamp.max)
filtered_SW_after_min = SW_after_min[SW_after_min['Datetime SW'] < SW_after_min['Next Vmax']]

# Insert the filtered slack water times into the combined DataFrame
combined_df['HWS'] = filtered_SW_after_min['Datetime SW']
# HWS = Slack Water After Vmin
df_current_lag = combined_df.copy()

# Step 3 : Insérer les HW et LW.
# HW après Vmin et LW après Vmax
merged_hw = pd.merge_asof(df_min_discharge, df_HW[['Datetime HW']],
                           left_on='Datetime Vmin', right_on='Datetime HW',
                           direction='forward')

merged_hw_HD = pd.merge_asof(df_min_discharge, df_HW_HD[['Datetime HW HD']],
                           left_on='Datetime Vmin', right_on='Datetime HW HD',
                           direction='forward')

merged_lw = pd.merge_asof(df_max_discharge, df_LW[['Datetime LW']],
                           left_on='Datetime Vmax', right_on='Datetime LW',
                           direction='forward')

merged_lw_HD = pd.merge_asof(df_max_discharge, df_LW_HD[['Datetime LW HD']],
                           left_on='Datetime Vmax', right_on='Datetime LW HD',
                           direction='forward')

# Reorder the columns as needed
combined_df['HW'] = merged_hw['Datetime HW']
combined_df['LW'] = merged_lw['Datetime LW']

combined_df = combined_df.rename(columns={'Datetime Vmin': 'Vmin', 'Datetime Vmax': 'Vmax'})
combined_df['Vmax next'] = combined_df['Vmax'].shift(-1)

combined_df = combined_df[['Vmax', 'LW', 'LWS', 'Vmin', 'HW', 'HWS', 'Vmax next']] #Slack water after Vmax is LWS.
# Check, j'ai bien un cycle de LWS-LWS qui dure 25h09, 25h03 pour HWS-HWS, 25h01 pour la médiane des 2 cycles.
lag = 'HW'
median_cycle = (combined_df[lag] - combined_df[lag].shift(1)).median()

# Il ne me manque plus que la daily discharge, et le tidal range.
# Je rajoute une colonne daily Q, qui sera le débit journalier à la date de Vmin
# (arbitrairement mais choisi pour être sûre qu'il y en ait un différent à chaque ligne (i.e. soit Vmin, soit Vmax)
daily_mean['Date'] = daily_mean.index
# Create a dictionary mapping Datetime Vmin to Daily Mean Discharge
daily_mean_map = dict(zip(daily_mean['Date'], daily_mean['Q']))
# Assign the Daily Mean Discharge to corresponding Datetime Vmin
combined_df['Q'] = combined_df['LW'].dt.date.map(daily_mean_map)
# 4/07 : je passe de Vmax à LW car, problème de TR qui correspond à celui du jour précédant...

# Et maintenant le tidal range.
# Je fais le choix pour le TR de 1. faire la moyenne de EBb et Flood. ou
# 2. de ne garder que les valeurs de Ebb, afin d'éviter les nan
# A HON DAU !!!!!!!!
daily_mean_map_TR = dict(zip(Ebb_HD['Datetime'].dt.date, (Flood_HD['Amplitude HD'] + abs(Ebb_HD['Amplitude HD']))/200))
combined_df['TR'] = combined_df['LW'].dt.date.map(daily_mean_map_TR)
# J'interpole donc les valeurs manquantes,
combined_df['TR'] = combined_df['TR'].interpolate()

############################################################################################
# Check de la propagation des LW et HW
merged_hw = pd.merge_asof(df_HW_HD[['Datetime HW HD']], df_HW[['Datetime HW']],
                           left_on='Datetime HW HD', right_on='Datetime HW',
                           direction='forward')

merged_hw['Q'] = merged_hw['Datetime HW HD'].dt.date.map(daily_mean_map)
merged_hw['TR'] = merged_hw['Datetime HW HD'].dt.date.map(daily_mean_map_TR)
merged_hw['TR'] = merged_hw['TR'].interpolate()

merged_hw_filtered = merged_hw[(merged_hw['Datetime HW']-merged_hw['Datetime HW HD'])<pd.Timedelta(hours=5)]
# I want to check the impact of river discharge and TR on the different lags
print('HW lag propagation')
mean_val = (merged_hw_filtered['Datetime HW']-merged_hw_filtered['Datetime HW HD']).mean()
q1, median_val, q4 = (merged_hw_filtered['Datetime HW']-merged_hw_filtered['Datetime HW HD']).quantile([0.25, 0.5, 0.75])
print('mean', mean_val,'\n', 'Q1', q1,'\n', 'med', median_val, '\n','Q4', q4)
time_diffs_hours = (merged_hw_filtered['Datetime HW']-merged_hw_filtered['Datetime HW HD']).dt.total_seconds()/3600
#np.array([(time_local_max[i+1] - time_local_max[i]).total_seconds() / 3600 for i in range(len(time_local_min) - 1)])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_hw_filtered['TR'], time_diffs_hours)
print('correlation with TR', r_value, p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_hw_filtered['Q'], time_diffs_hours)
print('correlation with Q', r_value, p_value)
##########################################

merged_lw = pd.merge_asof(df_LW_HD["Datetime LW HD"], df_LW[['Datetime LW']],
                           left_on='Datetime LW HD', right_on='Datetime LW',
                           direction='forward')

merged_lw['Q'] = merged_lw['Datetime LW HD'].dt.date.map(daily_mean_map)
merged_lw['TR'] = merged_lw['Datetime LW HD'].dt.date.map(daily_mean_map_TR)
merged_lw['TR'] = merged_lw['TR'].interpolate()

merged_lw_filtered = merged_lw[(merged_lw['Datetime LW']-merged_lw['Datetime LW HD'])<pd.Timedelta(hours=7)]
# I want to check the impact of river discharge and TR on the different lags
print('LW lag propagation')
mean_val = (merged_lw_filtered['Datetime LW']-merged_lw_filtered['Datetime LW HD']).mean()
q1, median_val, q4 = (merged_lw_filtered['Datetime LW']-merged_lw_filtered['Datetime LW HD']).quantile([0.25, 0.5, 0.75])
print('mean', mean_val,'\n', 'Q1', q1,'\n', 'med', median_val, '\n','Q4', q4)
time_diffs_hours = (merged_lw_filtered['Datetime LW']-merged_lw_filtered['Datetime LW HD']).dt.total_seconds()/3600
#np.array([(time_local_max[i+1] - time_local_max[i]).total_seconds() / 3600 for i in range(len(time_local_min) - 1)])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_lw_filtered['TR'], time_diffs_hours)
print('correlation with TR', r_value, p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_lw_filtered['Q'], time_diffs_hours)
print('correlation with Q', r_value, p_value)

############################################################################################
############################################################################################
############################################################################################
# Necessity to clean the df : remove nan ==> No need if the TR is interpolated
# But need to remove the NaT of datetime
combined_df_bis = combined_df.drop_duplicates('HW', keep='last').drop_duplicates('LW', keep='last')

lag1 = 'Vmin'
lag2 = 'Vmax next'
shift = False # PLUS NECESSAIRE VU QUE LE SHIFT EST INCLU If I the part to examine is in the following line
# Apply the shift conditionally
lag2_series = combined_df_bis[lag2].shift(-1) if shift else combined_df_bis[lag2]
# Find the indices of NaN values
list_nan1 = combined_df_bis.index[combined_df_bis[lag1].isna()].tolist()
list_nan2 = combined_df_bis.index[lag2_series.isna()].tolist()
list_nan = list_nan1 + list_nan2
combined_df_bis_filter = combined_df_bis.drop(list_nan)
lag2_series = lag2_series.drop(list_nan) if shift else combined_df_bis_filter[lag2]
# I want to check the impact of river discharge and TR on the different lags
print('lag ', lag1, lag2)
mean_val = (lag2_series- combined_df_bis_filter[lag1]).mean()
q1, median_val, q4 = (lag2_series - combined_df_bis_filter[lag1]).quantile([0.25, 0.5, 0.75])
print('mean', mean_val,'\n', 'Q1', q1,'\n', 'med', median_val, '\n','Q4', q4)
time_diffs_hours = (lag2_series - combined_df_bis_filter[lag1]).dt.total_seconds()/3600
#np.array([(time_local_max[i+1] - time_local_max[i]).total_seconds() / 3600 for i in range(len(time_local_min) - 1)])
slope, intercept, r_value, p_value, std_err = stats.linregress(combined_df_bis_filter['TR'], time_diffs_hours)
print('correlation with TR', r_value, p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(combined_df_bis_filter['Q'], time_diffs_hours)
print('correlation with Q', r_value, p_value)
############################################################################################
############################################################################################
############################################################################################
# df recap des valeurs de Q1 Q2 Q3 Q4 pour les débits et TR
df_recap_lag_TR = pd.DataFrame(columns = ['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax',
                                          'LW propagation', 'HW propagation', 'Vmax-Vmin'])
df_recap_lag_TR_std = pd.DataFrame(columns = ['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax',
                                          'LW propagation', 'HW propagation', 'Vmax-Vmin'])

# Catégories avec les débits
df_recap_lag_Q = pd.DataFrame(columns = ['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax',
                                          'LW propagation', 'HW propagation', 'Vmax-Vmin'])
df_recap_lag_Q_std = pd.DataFrame(columns = ['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax',
                                          'LW propagation', 'HW propagation', 'Vmax-Vmin'])
quantile = False
if quantile:
    deb_dis = combined_df_bis.quantile([0.25, 0.5, 0.75])['Q'].values
    deb_TR = combined_df_bis.quantile([0.25, 0.5, 0.75])['TR'].abs().values
else:
    deb_dis = [500, 1000, 1500]
    deb_TR = [1, 2, 3]
print('Quantile = ', str(quantile), 'Discharge value = ' + str(deb_dis))
print('Quantile = ', str(quantile), 'TR value = ' + str(deb_TR))

cond1 = combined_df_bis['Q'] < deb_dis[0]
cond2 = (combined_df_bis['Q'] > deb_dis[0]) & (combined_df_bis['Q'] < deb_dis[1])
cond3 = (combined_df_bis['Q'] > deb_dis[1]) & (combined_df_bis['Q'] < deb_dis[2])
cond4 = combined_df_bis['Q'] > deb_dis[2]
Q1 = combined_df_bis.loc[cond1]
Q2 = combined_df_bis.loc[cond2]
Q3 = combined_df_bis.loc[cond3]
Q4 = combined_df_bis.loc[cond4]
print('')
print('len(Q1) et len(Q4) = ', len(Q1), len(Q4))
# print('duration flood or ebb Q1 / Q4: ', Q1['Duration'].median(), Q4['Duration'].median())

condA = (abs(combined_df_bis['TR'])) < deb_TR[0]
condB = (abs(combined_df_bis['TR'] > deb_TR[0])) & (abs(combined_df_bis['TR']) < deb_TR[1])
condC = (abs(combined_df_bis['TR']) > deb_TR[1]) & (abs(combined_df_bis['TR']) < deb_TR[2])
condD = (abs(combined_df_bis['TR']) > deb_TR[2])
TR1 = combined_df_bis.loc[condA]
TR2 = combined_df_bis.loc[condB]
TR3 = combined_df_bis.loc[condC]
TR4 = combined_df_bis.loc[condD]
print('len(TR1) et len(TR4) = ', len(TR1), len(TR4))
# print('duration flood or ebb TR1 / TR4: ', TR1['Duration'].median(), TR4['Duration'].median())

#cols = ['Diff water level', 'Diff', combined_df_bis.columns[-3]]
cols = combined_df_bis.columns[0:-2]

# ATTENTION : Il faut que je supprime les lignes ou il n'y a pas les LWS et HWS
# ATTENTION : Ca veut dire que je n'ai plus des groupes homogènes : pour HW-HWS, j'ai moins de valeurs dans le Q4,
# car souvent, il n'y a pas de HWS donc je n'ai que 111 val contre 166 dans Q1
for i in range(len(cols)-1):
    print(cols[i], cols[i+1])
    #if cols[i]=='LWS' or cols[i+1]=='HWS' or cols[i]=='HWS' or cols[i+1]=='HWS':
    # Je dois supprimer les nan des sous-df
    Qbis_list = [Q.copy() for Q in [Q1, Q2, Q3, Q4]] #[Q1bis, Q2bis, Q3bis, Q4bis]
    a = 0
    for Qbis in [Q1,Q2,Q3,Q4]:
        list_nan1 = Qbis.index[Qbis[cols[i]].isna()].tolist()
        list_nan2 = Qbis.index[Qbis[cols[i+1]].isna()].tolist()
        list_nan = list_nan1 + list_nan2
        Qbis_list[a] = Qbis.drop(list_nan)
        a = a + 1
    TRbis_list = [TR.copy() for TR in [TR1,TR2, TR3, TR4]] #[Q1bis, Q2bis, Q3bis, Q4bis]
    a = 0
    for TRbis in [TR1,TR2,TR3,TR4]:
        list_nan1 = TRbis.index[TRbis[cols[i]].isna()].tolist()
        list_nan2 = TRbis.index[TRbis[cols[i+1]].isna()].tolist()
        list_nan = list_nan1 + list_nan2
        TRbis_list[a] = TRbis.drop(list_nan)
        a = a + 1

    df_recap_lag_Q[df_recap_lag_Q.columns[i]] = [(Q[cols[i+1]]-Q[cols[i]]).median() for Q in Qbis_list]
    df_recap_lag_Q_std[df_recap_lag_Q_std.columns[i]] = [(Q[cols[i+1]]-Q[cols[i]]).std() for Q in Qbis_list]
    df_recap_lag_TR[df_recap_lag_TR.columns[i]] = [(TR[cols[i+1]]-TR[cols[i]]).median() for TR in TRbis_list]
    df_recap_lag_TR_std[df_recap_lag_TR_std.columns[i]] = [(TR[cols[i+1]]-TR[cols[i]]).std() for TR in TRbis_list]
    #print(str(TR1[col].median()))

# Cas particulier pour HW et LW propagation.
lags_water_propagation = ['HW propagation', 'LW propagation']
a = 0
for df in [merged_hw_filtered, merged_lw_filtered]:
    cond1 = df['Q'] < deb_dis[0]
    cond2 = (df['Q'] > deb_dis[0]) & (df['Q'] < deb_dis[1])
    cond3 = (df['Q'] > deb_dis[1]) & (df['Q'] < deb_dis[2])
    cond4 = df['Q'] > deb_dis[2]
    Q1 = df.loc[cond1]
    Q2 = df.loc[cond2]
    Q3 = df.loc[cond3]
    Q4 = df.loc[cond4]
    print('')
    print('len(Q1) et len(Q4) = ', len(Q1), len(Q4))
    # print('duration flood or ebb Q1 / Q4: ', Q1['Duration'].median(), Q4['Duration'].median())

    condA = (abs(df['TR'])) < deb_TR[0]
    condB = (abs(df['TR'] > deb_TR[0])) & (abs(df['TR']) < deb_TR[1])
    condC = (abs(df['TR']) > deb_TR[1]) & (abs(df['TR']) < deb_TR[2])
    condD = (abs(df['TR']) > deb_TR[2])
    TR1 = df.loc[condA]
    TR2 = df.loc[condB]
    TR3 = df.loc[condC]
    TR4 = df.loc[condD]
    df_recap_lag_Q[lags_water_propagation[a]] = [(Q[Q.columns[1]]-Q[Q.columns[0]]).median() for Q in [Q1,Q2,Q3,Q4]]
    df_recap_lag_Q_std[lags_water_propagation[a]] = [(Q[Q.columns[1]]-Q[Q.columns[0]]).std() for Q in [Q1,Q2,Q3,Q4]]
    df_recap_lag_TR[lags_water_propagation[a]] = [(TR[TR.columns[1]]-TR[TR.columns[0]]).median() for TR in [TR1,TR2,TR3,TR4]]
    df_recap_lag_TR_std[lags_water_propagation[a]] = [(TR[TR.columns[1]]-TR[TR.columns[0]]).std() for TR in [TR1,TR2,TR3,TR4]]
    a = a + 1

#############################################################################"
#############################################################################
# Je calcule la durée moyenne de mes min et max 2 à 2 :
mean_min_TT = np.nanmean([time_local_min[i+1]- time_local_min[i] for i in range(len(time_local_min)-1)])
# Timedelta('1 days 01:04:13.084648493')
mean_max_TT = np.nanmean([time_local_max[i+1]- time_local_max[i] for i in range(len(time_local_max)-1)])
# Timedelta('1 days 01:04:23.414634146')
mean_max_HD = np.nanmean([time_local_max_HD[i+1]- time_local_max_HD[i] for i in range(len(time_local_max_HD)-1)])
#Timedelta('1 days 01:04:25.997130559')
mean_min_HD = np.nanmean([time_local_min_HD[i+1]- time_local_min_HD[i] for i in range(len(time_local_min_HD)-1)])
#Timedelta('1 days 01:04:20.832137733')
mean_entre_SW = np.nanmean([time_local_SW[i+1]- time_local_SW[i] for i in range(len(time_local_SW)-1)])
# Timedelta('0 days 12:25:34.705464868')

##############################################################################
#################################################################################
# Convert timedelta strings to timedeltas
df_recap_lag_TR = df_recap_lag_TR.apply(pd.to_timedelta)
df_recap_lag_TR_std = df_recap_lag_TR_std.apply(pd.to_timedelta)
df_recap_lag_Q = df_recap_lag_Q.apply(pd.to_timedelta)
df_recap_lag_Q_std = df_recap_lag_Q_std.apply(pd.to_timedelta)
# Convert timedeltas to seconds for plotting
df_recap_lag_TR_seconds = df_recap_lag_TR.apply(lambda x: x.dt.total_seconds())
df_recap_lag_Q_seconds = df_recap_lag_Q.apply(lambda x: x.dt.total_seconds())
df_recap_lag_TR_std_seconds = df_recap_lag_TR_std.apply(lambda x: x.dt.total_seconds())
df_recap_lag_Q_std_seconds = df_recap_lag_Q_std.apply(lambda x: x.dt.total_seconds())

y_pos = [0,0,0,1]
fontsize = 10
bar_height = 0.25  # Height of each bar

# Figures avec seulement Q1 et Q4.
fig, ax = plt.subplots(figsize=(4, 4))
colors = ['sandybrown', 'skyblue', 'green', 'red', 'purple', 'brown', 'pink']  # Define colors for each lag
#plt.title('Water level propagation with several tidal range', fontsize = fontsize +2)
for i in [0,3]:
    print('i', i)
    for j, lag in enumerate(df_recap_lag_TR_seconds[['LW propagation', 'HW propagation']]):
        ax.barh(y_pos[i] + j * bar_height, df_recap_lag_TR_seconds[lag][i], bar_height,
                xerr=df_recap_lag_TR_std_seconds[lag][i], capsize = 1, ecolor = colors[j], label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j])
        value = df_recap_lag_TR[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        plt.text(200, y_pos[i] + j * bar_height + bar_height/2, f"{value.components.hours}h{zero}{value.components.minutes}",
                 va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        #if i == 0 :
        #    ax.legend(fontsize = fontsize, loc='upper right')
plt.xticks([])
plt.yticks([0.25, 1.25], ['TR1', 'TR4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'new_LW_HW_propagation_with_Tidal_range_Q1-Q4'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')
#####
# FIGURE 2 : LW and HW propagation depending on Q :
fig, ax = plt.subplots(figsize=(4, 4))
colors = ['sandybrown', 'skyblue', 'green', 'red', 'purple', 'brown', 'pink']  # Define colors for each lag
#ax.set_ylim(0,2)
#plt.title('Water level propagation with several water discharges', fontsize = fontsize +2)
for i in [0,3]:
    print('i', i)
    for j, lag in enumerate(df_recap_lag_Q_seconds[['LW propagation', 'HW propagation']]):
        ax.barh(y_pos[i] + j * bar_height, df_recap_lag_Q_seconds[lag][i], bar_height,
                xerr=df_recap_lag_Q_std_seconds[lag][i], capsize = 1, ecolor = colors[j], label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j])
        value = df_recap_lag_Q[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        plt.text(200, y_pos[i] + j * bar_height + bar_height/2, f"{value.components.hours}h{zero}{value.components.minutes}",
                 va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        if i == 0 and j==1 :
            fig.legend(fontsize = fontsize, loc='upper right')
plt.xticks([])
plt.yticks([0.25, 1.25], ['Q1', 'Q4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'new_LW_HW_propagation_with_discharge_Q1-Q4'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')
#######################################################

# FIGURE 3 : Les autres lags, mis à la queue leu leu pour classes de débit
#colors = ['coral', 'slategray', 'mediumseagreen', 'firebrick', 'purple', 'gold', 'pink']  # Define colors for each lag
#colors = cmap_discretize(mpl.cm.Spectral, 6)
cmap = plt.cm.copper
colors = [cmap(i) for i in range(cmap.N)][::51]
alpha = 0.6
fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor('none')  # Transparent figure background
ax.patch.set_facecolor('none')
#plt.title('Lags with several water discharges', fontsize = fontsize +2)
start =  np.zeros(4)
ax.set_ylim(-0.50,1.4)
for i in [0,3]:
    print('i', i)
    cum_val = 0
    for j, lag in enumerate(df_recap_lag_Q_seconds[['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax']]):
        print('lag = ',lag, 'j=', str(j))
        ax.barh(y_pos[i] - bar_height/2, df_recap_lag_Q_seconds[lag][i], bar_height, label=lag, color=colors[j],
                align = 'edge', alpha=alpha, left = start[i]) # edgecolor=colors[j],
        value = df_recap_lag_Q[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        if value.components.days < 0 :
            value = timedelta(hours=24) - value
            time_lag = f"-{value.components.hours}h{zero}{value.components.minutes}"
        else :
            time_lag = f"{value.components.hours}h{zero}{value.components.minutes}"
        if (lag == 'HW-HWS'):# and (i==3) :
            plt.text(cum_val-2000, y_pos[i]+0.15, time_lag, va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        else :
            plt.text(cum_val+100, y_pos[i], time_lag, va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        cum_val = cum_val + df_recap_lag_Q[lag][i].seconds
        #print(f"{value.components.hours}h{zero}{value.components.minutes}")
        start[i] = start[i] + df_recap_lag_Q_seconds[lag][i]
        print(start)
    #if i == 0 :
    #    fig.legend(fontsize = fontsize, loc='upper right')
plt.xticks([])
plt.yticks([0, 1], ['Q1', 'Q4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'new_Lags_at_TT_with_discharge_Q1-Q4'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')
###############################################################
# Plot just the legend :
fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor('none')  # Transparent figure background
ax.patch.set_facecolor('none')   # Transparent axes background
for j, lag in enumerate(df_recap_lag_Q_seconds[['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax']]):
    print('lag = ', lag, 'j=', str(j))
    bar = ax.plot([], [], label=lag, color=colors[j], linewidth = 10, alpha=alpha)   # edgecolor=colors[j],
ax.legend(fontsize = fontsize, frameon = False)
# Step 3: Hide the axes
ax.axis('off')
fig.savefig('legend.png')
##############################################################
# FIGURE 4 : Les autres lags, mis à la queue leu leu pour classes de TR
fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor('none')  # Transparent figure background
ax.patch.set_facecolor('none')
#plt.title('Lags with several tidal range', fontsize = fontsize +2)
start =  np.zeros(4)
ax.set_ylim(-0.50,1.4)
for i in [0,3]:
    print('i', i)
    cum_val = 0
    for j, lag in enumerate(df_recap_lag_TR_seconds[['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax']]):
        print('lag = ',lag, 'j=', str(j))
        ax.barh(y_pos[i] - bar_height/2, df_recap_lag_TR_seconds[lag][i], bar_height, label=lag, color=colors[j],
                align = 'edge', alpha=alpha,  left = start[i]) #edgecolor=colors[j],
        value = df_recap_lag_TR[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        if value.components.days < 0 :
            value = timedelta(hours=24) - value
            time_lag = f"-{value.components.hours}h{zero}{value.components.minutes}"
        else :
            time_lag = f"{value.components.hours}h{zero}{value.components.minutes}"
        if (lag == 'HW-HWS'):
            plt.text(cum_val - 2000, y_pos[i] + 0.15, time_lag, va='center', ha='left',
                     fontsize=fontsize - 3)  # x = value.seconds/2.5
        else :
            plt.text(cum_val+100, y_pos[i], time_lag, va='center', ha='left', fontsize = fontsize-3) # x = value.seconds/2.5
        cum_val = cum_val + df_recap_lag_TR[lag][i].seconds
        #print(f"{value.components.hours}h{zero}{value.components.minutes}")
        start[i] = start[i] + df_recap_lag_TR_seconds[lag][i]
        print(start)
        #if i == 0 :
        #    ax.legend(fontsize = fontsize, loc='center right')
plt.xticks([])
plt.yticks([0, 1], ['TR1', 'TR4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'new_Lags_at_TT_with_TR_Q1-Q4'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')

###########################################################""
############################################
# Je plot des exemples de courbes pour illustrer Q1 Q4 des Q et TR
# Je plot des exemples de courbes pour illustrer les situations EXTREMES :
# low Q + low TR , low Q + high TR, high Q + low TR, high Q + high TR
crit_Q = 20
crit_TR = 0.15
combined_df_bis['LW next'] = combined_df_bis['LW'].shift(-1)
combined_df_bis['LWS next'] = combined_df_bis['LWS'].shift(-1)
combined_df_bis['TR'] = combined_df_bis['LW next'].dt.date.map(daily_mean_map_TR)
combined_df_bis['Q'] = combined_df_bis['LW next'].dt.date.map(daily_mean_map)
combined_df_bis = combined_df_bis[['Vmax', 'LW', 'LWS', 'Vmin', 'HW', 'HWS', 'Vmax next', 'LW next', 'LWS next', 'Q', 'TR']]

days_closest_median_Q = combined_df_bis.loc[(combined_df_bis['Q'] - combined_df_bis['Q'].median()).abs() < crit_Q]
days_closest_median_TR = combined_df_bis.loc[(combined_df_bis['TR'] - combined_df_bis['TR'].median()).abs() < crit_TR]

low_Q = 250
high_Q = 1200
low_TR = 1
high_TR = 3
days_low_Q_median_TR = days_closest_median_TR[days_closest_median_TR['Q'] < low_Q].dropna()
days_high_Q_median_TR = days_closest_median_TR[days_closest_median_TR['Q'] > high_Q].dropna(subset = ['Vmax', 'LW', 'Vmin',
                                                                                            'HW', 'Vmax next'])
days_median_Q_low_TR = days_closest_median_Q[days_closest_median_Q['TR'] < low_TR].dropna()
days_median_Q_high_TR = days_closest_median_Q[days_closest_median_Q['TR'] > high_TR].dropna()

days_median_Q_median_TR = days_closest_median_TR[(days_closest_median_TR['Q'] - combined_df_bis['Q'].median()).abs() < crit_Q].dropna()

days_closest_high_Q = combined_df_bis.loc[(combined_df_bis['Q'] - high_Q).abs() < crit_Q]
days_closest_high_TR = combined_df_bis.loc[(combined_df_bis['TR'] - high_TR).abs() < crit_TR]
days_closest_low_Q = combined_df_bis.loc[(combined_df_bis['Q'] - low_Q).abs() < crit_Q]
days_closest_low_TR = combined_df_bis.loc[(combined_df_bis['TR'] - low_TR).abs() < crit_TR]

days_low_Q_low_TR = days_closest_low_TR[days_closest_low_TR['Q'] < low_Q].dropna()
days_high_Q_low_TR = days_closest_low_TR[days_closest_low_TR['Q'] > high_Q].dropna(subset=['Vmax', 'LW', 'Vmin',
                                                                                            'HW', 'Vmax next'])
days_low_Q_high_TR = days_closest_high_TR[days_closest_high_TR['Q'] < low_Q].dropna()
days_high_Q_high_TR = days_closest_high_TR[days_closest_high_TR['Q'] > high_Q].dropna(subset=['Vmax', 'LW', 'Vmin',
                                                                                            'HW', 'Vmax next'])

fontsize = 35
alpha = 0.3
lw = 4
s = 10
color_twin = 'royalblue'
colors = ["gray", 'moccasin', "gray", 'moccasin', "gray", 'moccasin', "gray", 'moccasin', "gray", 'moccasin']
list_name = ['low_Q_low_TR', 'low_Q_median_TR', 'low_Q_high_TR', 'high_Q_low_TR', 'high_Q_median_TR', 'high_Q_high_TR']
incr=0
for list_days in [days_low_Q_low_TR, days_low_Q_median_TR, days_low_Q_high_TR,
                  days_high_Q_low_TR, days_high_Q_median_TR, days_high_Q_high_TR]:
    print('next list_days')
    for l in range(len(list_days)):
        print(l)
        # Attention il faut sélectionner un exemple ou il y a LWS et HWS.
        date = list_days.reset_index()['Vmin'].dt.date[l]  # Vmin
        print('date', date)
        # Pour Q1 : je choisis une TR moyen
        fig, ax = plt.subplots(figsize=(20, 10), nrows=1, sharex=True)
        plt.title(date, fontsize=fontsize)
        ax.grid(which='both', alpha=0.5)
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_ylabel('Water elevation at TT (m)', fontsize=fontsize)
        ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT water level',
                color='black', lw=lw, zorder=0.1)
        # ax.scatter(time_local_min, local_minima / 100, marker='x', color='black', s=s, zorder=1)
        # ax.scatter(time_local_max, local_maxima / 100, marker='x', color='black', s=s, zorder=1)
        twin = ax.twinx()
        twin.set_ylabel('Discharge at TT (m$³$/s)', fontsize=fontsize, color=color_twin)
        twin.tick_params(axis='y', colors=color_twin, labelsize=fontsize - 5)  # Set tick label color
        twin.spines['right'].set_color(color_twin)  # Set spine color
        twin.plot(water_levels['Datetime'], water_levels['Q'], label='TT Discharge', ls='--', color=color_twin,
                  lw=lw, zorder=0.1)
        # twin.plot(daily_mean.index, daily_mean['Q'], label='Daily discharge', ls='--', color=color_twin,
        #          zorder=0.1)
        twin.axhline(0, color=color_twin)
        # twin.scatter(time_local_min_Q, local_minima_Q, label='extreme values', marker='x', color='black', s=s, zorder=1)
        # twin.scatter(time_local_max_Q, local_maxima_Q, marker='x', color='black', s=s, zorder=1)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.axvline(x=list_days['LW'].iloc[l], color='k', lw=lw, zorder=0)
        ax.axvline(x=list_days['HW'].iloc[l], color='k', lw=lw, zorder=0)
        ax.axvline(x=list_days['LW next'].iloc[l], color='k', lw=lw, zorder=0)

        ax.axvline(x=list_days['Vmin'].iloc[l], color=color_twin, lw=lw, ls='--', zorder=0)
        ax.axvline(x=list_days['Vmax'].iloc[l], color=color_twin, lw=lw, ls='--', zorder=0)
        ax.axvline(x=list_days['Vmax next'].iloc[l], color=color_twin, lw=lw, ls='--', zorder=0)

        x1 = list_days['LWS'].iloc[l]
        x2 = list_days['HWS'].iloc[l]
        if pd.isna(x1) or pd.isna(x2):
            print('One is NAT')
        else:
            ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor='gray', alpha=alpha)

        x1 = list_days['HWS'].iloc[l]
        x2 = list_days['LWS next'].iloc[l]
        print('x1', x1, 'x2', x2)
        if pd.isna(x1) or pd.isna(x2):
            print('One is NAT')
            if date == datetime(2022, 6, 6).date():
                print('Specific case High Q')
                ax.axvspan(x1, datetime(2022, 6, 9), ymin=-10, ymax=10, facecolor='moccasin', alpha=alpha)
            elif date == datetime(2022, 8, 27).date():
                print('Specific case High Q')
                ax.axvspan(datetime(2022, 8, 25), datetime(2022, 8, 29), ymin=-10, ymax=10, facecolor='moccasin',
                           alpha=alpha)
            elif date == datetime(2022, 6, 20).date():
                print('Specific case High Q')
                ax.axvspan(datetime(2022, 6, 18), x2, ymin=-10, ymax=10, facecolor='moccasin', alpha=alpha)
            elif date == datetime(2022, 5, 27).date():
                print('Specific case High Q low TR')
                ax.axvspan(datetime(2022, 5, 18), datetime(2022, 5, 30), ymin=-10, ymax=10, facecolor='moccasin',
                           alpha=alpha)
            elif date == datetime(2022, 6, 19).date():
                print('Specific case High Q HIGH TR')
                ax.axvspan(datetime(2022, 6, 18), datetime(2022, 8, 29), ymin=-10, ymax=10, facecolor='moccasin',
                           alpha=alpha)
        else:
            ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor='moccasin', alpha=alpha)
        # twin.legend(lines + lines2, labels + labels2, fontsize=fontsize)
        # for i in range(len(list_days.columns[:-2])-1):
        #    for (x1, x2) in zip(list_days[list_days.columns[i]], list_days[list_days.columns[i+1]]):
        #        # print('x1' , x1, 'x2', x2)
        #        if pd.isna(x1) or pd.isna(x2):
        #            print('One is NAT')
        #        else:
        #            ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor=colors[i], alpha=alpha)
        date_form = DateFormatter("%H:%M")  # Define the date format
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(date_form)
        ax.tick_params(axis='y', labelsize=fontsize - 5)
        ax.tick_params(axis='x', labelsize=fontsize - 5)
        date1 = list_days.reset_index()['Vmax'][l] - timedelta(hours=5)
        date2 = list_days.reset_index()['Vmax next'][l] + timedelta(hours=5)
        date1 = list_days.reset_index()['LWS'][l] - timedelta(hours=5)
        date2 = list_days.reset_index()['LWS next'][l] + timedelta(hours=5)
        date1 = list_days.reset_index()['HW'][l] - timedelta(hours=12)
        date2 = list_days.reset_index()['LW next'][l] + timedelta(hours=12)

        # date1 = (date - timedelta(hours=5))
        # date2 = (date + timedelta(hours=5))
        ax.set_xlim(date1, date2)
        outfile = list_name[incr] + '_exemple_tidal_elevations_Q_lag_' +\
                  str(date1.day) + str(date1.month) + str(date1.year) + '.png'
        fig.savefig(outfile, format='png')
    incr = incr+1

######################


fontsize = 25
alpha = 0.5
color_twin = 'royalblue'
cmap = plt.cm.Spectral
colors = [cmap(i) for i in range(cmap.N)][::51]
s=10
lw=5
for list_days in [days_low_Q , days_low_TR, days_high_TR]:
    # Attention il faut sélectionner un exemple ou il y a LWS et HWS.
    date = list_days.reset_index()['Vmin'].dt.date[0]
    print('date', date)
    # Pour Q1 : je choisis une TR moyen
    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, sharex=True)
    plt.title(date, fontsize = fontsize)
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_ylabel('Tidal elevation at TT (m)', fontsize=fontsize)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT water level',
            color='black', lw=lw, zorder=0.1)
    ax.scatter(time_local_min, local_minima / 100, marker='x', color='black', s=s, zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, marker='x', color='black', s=s, zorder=1)
    twin = ax.twinx()
    twin.set_ylabel('Discharge at TT (m$³$/s)', fontsize=fontsize, color = color_twin)
    twin.tick_params(axis='y', colors=color_twin)  # Set tick label color
    twin.spines['right'].set_color(color_twin)    # Set spine color
    twin.plot(water_levels['Datetime'], water_levels['Q'], label='TT Discharge', ls='--', color=color_twin, lw=lw,
              zorder=0.1)
    #twin.plot(daily_mean.index, daily_mean['Q'], label='Daily discharge', ls='--', color=color_twin,
    #          zorder=0.1)
    twin.axhline(0, color=color_twin)
    twin.scatter(time_local_min_Q, local_minima_Q, label='extreme values', marker='x', color='black', s=s, zorder=1)
    twin.scatter(time_local_max_Q, local_maxima_Q, marker='x', color='black', s=s, zorder=1)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    #twin.legend(lines + lines2, labels + labels2, fontsize=fontsize)

    for i in range(len(list_days.columns[:-2])-1):
        for (x1, x2) in zip(list_days[list_days.columns[i]], list_days[list_days.columns[i+1]]):
            print('x1' , x1, 'x2', x2)
            if pd.isna(x1) or pd.isna(x2):
                print('One is NAT')
            else:
                ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor=colors[i], alpha=alpha)

    date_form = DateFormatter("%H:%M")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(date_form)
    date1 = list_days.reset_index()['Vmax'][0] - timedelta(hours=5)
    date2 = list_days.reset_index()['Vmax next'][0] + timedelta(hours=5)
    plt.subplots_adjust(left=0.1, right=0.83, top=0.9, bottom=0.15)
    #date1 = (date - timedelta(hours=5))
    #date2 = (date + timedelta(hours=5))
    ax.set_xlim(date1, date2)
    fig.savefig('exemple_tidal_elevations_Q_lag_' + str(date1.day) + str(date1.month) + str(date1.year) + '.png', format='png')

# Plot particulier pour High Q car il n'y a pas de slack water :
for list_days in [days_high_Q] : #, days_low_Q , days_low_TR, days_high_TR]:
    # Attention il faut sélectionner un exemple ou il y a LWS et HWS.
    date = list_days.reset_index()['Vmax'].dt.date[1]
    print('date', date)
    # Pour Q1 : je choisis une TR moyen
    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, sharex=True)
    plt.title(date, fontsize = fontsize)
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_ylabel('Tidal elevation at TT (m)', fontsize=fontsize)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT water level',
            color='black', lw=2, zorder=0.1)
    ax.scatter(time_local_min, local_minima / 100, marker='x', color='black', s=s, zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, marker='x', color='black', s=s, zorder=1)
    twin = ax.twinx()
    twin.set_ylabel('Discharge at TT (m$³$/s)', fontsize=fontsize, color = color_twin)
    twin.tick_params(axis='y', colors=color_twin)  # Set tick label color
    twin.spines['right'].set_color(color_twin)    # Set spine color
    twin.plot(water_levels['Datetime'], water_levels['Q'], label='TT Discharge', ls='--', color=color_twin,
              zorder=0.1)
    #twin.plot(daily_mean.index, daily_mean['Q'], label='Daily discharge', ls='--', color=color_twin,
    #          zorder=0.1)
    twin.axhline(0, color=color_twin)
    twin.scatter(time_local_min_Q, local_minima_Q, label='extreme values', marker='x', color='black', s=s, zorder=1)
    twin.scatter(time_local_max_Q, local_maxima_Q, marker='x', color='black', s=s, zorder=1)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    # twin.legend(lines + lines2, labels + labels2, fontsize=fontsize)

    list_columns = [0,1,3,4,6]
    for i in range(len(list_columns)-1):
        for (x1, x2) in zip(list_days[list_days.columns[list_columns[i]]], list_days[list_days.columns[list_columns[i+1]]]):
            print('x1' , x1, 'x2', x2)
            if pd.isna(x1) or pd.isna(x2):
                print('One is NAT')
            else:
                ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor=colors[list_columns[i]], alpha=alpha)

    date_form = DateFormatter("%d/%m/%Y")  # Define the date format
    date_form = DateFormatter("%H:%M")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(date_form)
    date1 = list_days.reset_index()['Vmax'][0] - timedelta(hours=5)
    date2 = list_days.reset_index()['Vmax next'][0] + timedelta(hours=5)
    #date1 = (date - timedelta(hours=5))
    #date2 = (date + timedelta(hours=5))
    ax.set_xlim(date1, date2)
    fig.savefig('exemple_tidal_elevations_Q_lag_' + str(date1.day) + str(date1.month) + str(date1.year) + '.png',
                format='png')

#########
###########
#############
# 5/07 : calcul du damping
quantile = False
if quantile:
    deb = Ebb.quantile([0.25, 0.5, 0.75])['Q'].values
else:
    deb = [500, 1000, 1500]

cond1 = Ebb_df['Q'] < deb[0]
cond2 = (Ebb_df['Q'] > deb[0]) & (Ebb_df['Q'] < deb[1])
cond3 = (Ebb_df['Q'] > deb[1]) & (Ebb_df['Q'] < deb[2])
cond4 = Ebb_df['Q'] > deb[2]
# Condition on the amplitude
condA = (abs(Ebb_df['Amplitude HD']) < 100)
condB = (100 < abs(Ebb_df['Amplitude HD'])) & (abs(Ebb_df['Amplitude HD']) <= 200)
condC = (200 < abs(Ebb_df['Amplitude HD'])) & (abs(Ebb_df['Amplitude HD']) <= 300)
condD = (abs(Ebb_df['Amplitude HD']) > 300)

########
######## TESTS
########
combined_df_bis['LW next'] = combined_df_bis['LW'].shift(-1)
combined_df_bis['LWS next'] = combined_df_bis['LWS'].shift(-1)
combined_df_bis = combined_df_bis[['Vmax', 'LW', 'LWS', 'Vmin', 'HW', 'HWS', 'Vmax next', 'LW next', 'LWS next', 'Q', 'TR']]
crit_Q = 20
crit_TR = 0.1
days_closest_median_Q = combined_df_bis.loc[(combined_df_bis['Q'] - combined_df_bis['Q'].median()).abs() < crit_Q]
days_closest_median_TR = combined_df_bis.loc[(combined_df_bis['TR'] - combined_df_bis['TR'].median()).abs() < crit_TR]
low_Q = 300
high_Q = 1200
low_TR = 1.4
high_TR = 2.6
days_low_Q = days_closest_median_TR[days_closest_median_TR['Q'] < low_Q]
days_high_Q = days_closest_median_TR[days_closest_median_TR['Q'] > high_Q]
days_low_TR = days_closest_median_Q[days_closest_median_Q['TR'] < low_TR]
days_high_TR = days_closest_median_Q[days_closest_median_Q['TR'] > high_TR]

alpha = 0.3
lw=4
s=10
colors = ["gray", 'moccasin',"gray", 'moccasin',"gray", 'moccasin',"gray", 'moccasin', "gray", 'moccasin' ]
for list_days in [days_high_Q]:
    print('next list_days')
    for l in range(len(list_days)):
        print(l)
        # Attention il faut sélectionner un exemple ou il y a LWS et HWS.
        date = list_days.reset_index()['Vmin'].dt.date[l]
        print('date', date)
        # Pour Q1 : je choisis une TR moyen
        fig, ax = plt.subplots(figsize=(20, 10), nrows=1, sharex=True)
        plt.title(date, fontsize = fontsize)
        ax.grid(which='both', alpha=0.5)
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_ylabel('Tidal elevation at TT (m)', fontsize=fontsize)
        ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT water level',
                color='black', lw=lw, zorder=0.1)
        #ax.scatter(time_local_min, local_minima / 100, marker='x', color='black', s=s, zorder=1)
        #ax.scatter(time_local_max, local_maxima / 100, marker='x', color='black', s=s, zorder=1)
        twin = ax.twinx()
        twin.set_ylabel('Discharge at TT (m$³$/s)', fontsize=fontsize, color = color_twin)
        twin.tick_params(axis='y', colors=color_twin)  # Set tick label color
        twin.spines['right'].set_color(color_twin)    # Set spine color
        twin.plot(water_levels['Datetime'], water_levels['Q'], label='TT Discharge', ls='--', color=color_twin, lw=lw,
                  zorder=0.1)
        #twin.plot(daily_mean.index, daily_mean['Q'], label='Daily discharge', ls='--', color=color_twin,
        #          zorder=0.1)
        twin.axhline(0, color=color_twin)
        #twin.scatter(time_local_min_Q, local_minima_Q, label='extreme values', marker='x', color='black', s=s, zorder=1)
        #twin.scatter(time_local_max_Q, local_maxima_Q, marker='x', color='black', s=s, zorder=1)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.axvline(x=list_days['LW'].iloc[l], color='k', lw=2, zorder = 0)
        ax.axvline(x=list_days['HW'].iloc[l], color='k', lw=2, zorder = 0)
        ax.axvline(x=list_days['LW next'].iloc[l], color='k', lw=2, zorder=0)

        ax.axvline(x=list_days['Vmin'].iloc[l], color=color_twin, lw=2, zorder = 0)
        ax.axvline(x=list_days['Vmax'].iloc[l], color=color_twin, lw=2, zorder = 0)
        ax.axvline(x=list_days['Vmax next'].iloc[l], color=color_twin, lw=2, zorder=0)

        x1 = list_days['LWS'].iloc[l]
        x2 = list_days['HWS'].iloc[l]
        if pd.isna(x1) or pd.isna(x2):
            print('One is NAT')
        else:
            ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor='gray', alpha=alpha)

        x1 = list_days['HWS'].iloc[l]
        x2 = list_days['LWS next'].iloc[l]
        print('x1', x1, 'x2', x2)
        if pd.isna(x1) or pd.isna(x2):
            print('One is NAT')
            if date == datetime(2022,6,6).date():
                print('Specific case High Q')
                ax.axvspan(x1, datetime(2022,6,9), ymin=-10, ymax=10, facecolor='moccasin', alpha=alpha)
            elif date == datetime(2022,8,27).date():
                print('Specific case High Q')
                ax.axvspan(datetime(2022,8,25), datetime(2022,8, 29), ymin=-10, ymax=10, facecolor='moccasin',
                           alpha=alpha)
            elif date == datetime(2022,6,20).date():
                print('Specific case High Q')
                ax.axvspan(datetime(2022, 6, 18), x2, ymin=-10, ymax=10, facecolor='moccasin',  alpha=alpha)
        else:
            ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor='moccasin', alpha=alpha)
        #twin.legend(lines + lines2, labels + labels2, fontsize=fontsize)
        #for i in range(len(list_days.columns[:-2])-1):
        #    for (x1, x2) in zip(list_days[list_days.columns[i]], list_days[list_days.columns[i+1]]):
        #        # print('x1' , x1, 'x2', x2)
        #        if pd.isna(x1) or pd.isna(x2):
        #            print('One is NAT')
        #        else:
        #            ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor=colors[i], alpha=alpha)
        date_form = DateFormatter("%H:%M")  # Define the date format
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(date_form)
        date1 = list_days.reset_index()['Vmax'][l] - timedelta(hours=5)
        date2 = list_days.reset_index()['Vmax next'][l] + timedelta(hours=5)
        date1 = list_days.reset_index()['LWS'][l] - timedelta(hours=5)
        date2 = list_days.reset_index()['LWS next'][l] + timedelta(hours=5)
        date1 = list_days.reset_index()['HW'][l] - timedelta(hours=12)
        date2 = list_days.reset_index()['LW next'][l] + timedelta(hours=12)

        #date1 = (date - timedelta(hours=5))
        #date2 = (date + timedelta(hours=5))
        ax.set_xlim(date1, date2)
        fig.savefig('exemple_tidal_elevations_Q_lag_' + str(date1.day) + str(date1.month) + str(date1.year) + '.png',
                    format='png')

print(np.sum((combined_df_bis.loc[combined_df_bis['TR'] < 1.6]['HWS'] -
              combined_df_bis.loc[combined_df_bis['TR'] < 1.6]['LWS']).isna()))
# 25/07/24 :
mean_duration_LWS_HWS_high_TR = (combined_df_bis.loc[combined_df_bis['TR'] > 2.83]['HWS'] -
                                 combined_df_bis.loc[combined_df_bis['TR'] > 2.83 ]['LWS']).dropna().mean()
mean_duration_HWS_LWS_high_TR = (combined_df_bis.loc[combined_df_bis['TR'] > 2.83]['HWS'] -
                                 combined_df_bis.loc[combined_df_bis['TR'] > 2.83 ]['LWS']).dropna().mean()


# Je cherhche les min, max Q et min max TR pour voir min Q + min TR etc..
days_min_TR_min_Q = combined_df_bis.loc[(combined_df_bis['TR'] < low_TR) & (combined_df_bis['Q'] < low_Q)]
days_max_TR_max_Q = combined_df_bis.loc[(combined_df_bis['TR'] > high_TR) & (combined_df_bis['Q'] > high_Q)]
days_max_TR_min_Q = combined_df_bis.loc[(combined_df_bis['TR'] > high_TR) & (combined_df_bis['Q'] < low_Q)]
days_min_TR_max_Q = combined_df_bis.loc[(combined_df_bis['TR'] < low_TR) & (combined_df_bis['Q'] > high_Q)]

# Cas particulier pour HWS-Vmax, car Vmax est après
# Il faut que je shift tout le tableau : définition d'un nouveau df
# Attention, légère incohérence avec les données de débits et TR car on décale d'un jour, mais à priori pas énorme
df_Vmax_HWS = combined_df[['Vmax', 'HWS', 'Q', 'TR']].copy()
df_Vmax_HWS['Vmax shift'] = df_Vmax_HWS['Vmax'].shift(-1)
# Je calcule donc maintenant les nouvelles catégories.
cond1 = df_Vmax_HWS['Q'] < deb_dis[0]
cond2 = (df_Vmax_HWS['Q'] > deb_dis[0]) & (df_Vmax_HWS['Q'] < deb_dis[1])
cond3 = (df_Vmax_HWS['Q'] > deb_dis[1]) & (df_Vmax_HWS['Q'] < deb_dis[2])
cond4 = df_Vmax_HWS['Q'] > deb_dis[2]
Q1 = df_Vmax_HWS.loc[cond1]
Q2 = df_Vmax_HWS.loc[cond2]
Q3 = df_Vmax_HWS.loc[cond3]
Q4 = df_Vmax_HWS.loc[cond4]
print('len(Q1) et len(Q4) = ', len(Q1), len(Q4))
# print('duration flood or ebb Q1 / Q4: ', Q1['Duration'].median(), Q4['Duration'].median())
condA = (abs(df_Vmax_HWS['TR'])) < deb_TR[0]
condB = (abs(df_Vmax_HWS['TR'] > deb_TR[0])) & (abs(df_Vmax_HWS['TR']) < deb_TR[1])
condC = (abs(df_Vmax_HWS['TR']) > deb_TR[1]) & (abs(df_Vmax_HWS['TR']) < deb_TR[2])
condD = (abs(df_Vmax_HWS['TR']) > deb_TR[2])
TR1 = df_Vmax_HWS.loc[condA]
TR2 = df_Vmax_HWS.loc[condB]
TR3 = df_Vmax_HWS.loc[condC]
TR4 = df_Vmax_HWS.loc[condD]
print('len(TR1) et len(TR4) = ', len(TR1), len(TR4))

col2 = 'Vmax shift'
col1 = 'HWS'
Qbis_list = [Q.copy() for Q in [Q1, Q2, Q3, Q4]] #[Q1bis, Q2bis, Q3bis, Q4bis]
a = 0
for Qbis in [Q1,Q2,Q3,Q4]:
    list_nan1 = Qbis.index[Qbis[col2].isna()].tolist() #.shift(-1)
    list_nan2 = Qbis.index[Qbis[col1].isna()].tolist()
    list_nan = list_nan1 + list_nan2
    Qbis_list[a] = Qbis.drop(list_nan)
    a = a + 1
TRbis_list = [TR.copy() for TR in [TR1,TR2, TR3, TR4]] #[Q1bis, Q2bis, Q3bis, Q4bis]
a = 0
for TRbis in [TR1,TR2,TR3,TR4]:
    list_nan1 = TRbis.index[TRbis[col2].isna()].tolist() #.shift(-1)
    list_nan2 = TRbis.index[TRbis[col1].isna()].tolist()
    list_nan = list_nan1 + list_nan2
    TRbis_list[a] = TRbis.drop(list_nan)
    a = a + 1

df_recap_lag_Q['HWS-Vmax'] = [(Q[col2]-Q[col1]).median() for Q in Qbis_list]
df_recap_lag_Q_std['HWS-Vmax'] = [(Q[col2]-Q[col1]).std() for Q in Qbis_list]
df_recap_lag_TR['HWS-Vmax'] = [(TR[col2]-TR[col1]).median() for TR in TRbis_list]
df_recap_lag_TR_std['HWS-Vmax'] = [(TR[col2]-TR[col1]).std() for TR in TRbis_list]

for list_days in [days_low_TR]: #[days_low_Q, days_high_Q, days_low_TR, days_high_TR]:
    date = list_days.reset_index()['Vmin'].dt.date[1]
    print(date)
    year = date.year
    m = date.month
    date_indice1 = (date - timedelta(days=5))
    date_indice2 = (date + timedelta(days=5))
    d1 = date_indice1.day
    d2 = date_indice2.day
    # m = date_indice1.month
    indice_max_Q = np.where(
        (pd.to_datetime(time_local_max_Q).month == m) & (pd.to_datetime(time_local_max_Q).year == year) & (
                pd.to_datetime(time_local_max_Q).day >= d1) & (pd.to_datetime(time_local_max_Q).day <= d2))
    indice_min_wl = np.where(
        (pd.to_datetime(time_local_min).month == m) & (pd.to_datetime(time_local_min).year == year) & (
                pd.to_datetime(time_local_min).day >= d1) & (pd.to_datetime(time_local_min).day <= d2))
    indice_min_Q = np.where(
        (pd.to_datetime(time_local_min_Q).month == m) & (pd.to_datetime(time_local_min_Q).year == year) & (
                pd.to_datetime(time_local_min_Q).day >= d1) & (pd.to_datetime(time_local_min_Q).day <= d2))
    indice_max_wl = np.where(
        (pd.to_datetime(time_local_max).month == m) & (pd.to_datetime(time_local_max).year == year) & (
                pd.to_datetime(time_local_max).day >= d1) & (pd.to_datetime(time_local_max).day <= d2))
    indice_min_wl_HD = np.where(
        (pd.to_datetime(time_local_min_HD).month == m) & (pd.to_datetime(time_local_min_HD).year == year) & (
                pd.to_datetime(time_local_min_HD).day >= d1) & (pd.to_datetime(time_local_min_HD).day <= d2))
    indice_max_wl_HD = np.where(
        (pd.to_datetime(time_local_max_HD).month == m) & (pd.to_datetime(time_local_max_HD).year == year) & (
                pd.to_datetime(time_local_max_HD).day >= d1) & (pd.to_datetime(time_local_max_HD).day <= d2))

    # Figures Exemples :
    # Pour Q1 : je choisis une TR moyen
    fontsize = 18
    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, sharex=True)
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_ylabel('Tidal elevation at TT (m)', fontsize=fontsize)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT water level',
            color='black', lw=2, zorder=0.1)
    ax.scatter(time_local_min, local_minima / 100, marker='x', color='black', s=s, zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, marker='x', color='black', s=s, zorder=1)
    twin = ax.twinx()
    twin.set_ylabel('Discharge at TT (m$³$/s)', fontsize=fontsize, color = 'coral')
    twin.tick_params(axis='y', colors='coral')  # Set tick label color
    twin.spines['right'].set_color('coral')    # Set spine color
    twin.plot(water_levels['Datetime'], water_levels['Q'], label='TT Discharge', ls='--', color='coral',
              zorder=0.1)
    #twin.plot(daily_mean.index, daily_mean['Q'], label='Daily discharge', ls='--', color='coral',
    #          zorder=0.1)
    twin.axhline(0, color='coral')
    twin.scatter(time_local_min_Q, local_minima_Q, label='extreme values', marker='x', color='black', s=s, zorder=1)
    twin.scatter(time_local_max_Q, local_maxima_Q, marker='x', color='black', s=s, zorder=1)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    twin.legend(lines + lines2, labels + labels2, fontsize=fontsize)

    for (x1, x2) in zip(time_local_min[indice_min_wl], time_local_max_Q[indice_max_Q]):
        ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor='gray', alpha=0.2)
    for (x3, x4) in zip(time_local_min_Q[indice_min_Q], time_local_max[indice_max_wl]):
        ax.axvspan(x3, x4, ymin=-10, ymax=10, facecolor='teal', alpha=0.2)

    date_form = DateFormatter("%d/%m/%Y")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(date_form)
    date1 = (date - timedelta(days=1))
    date2 = (date + timedelta(days=1))
    ax.set_xlim(date1, date2)
    fig.savefig('exemple_tidal_elevations_Q_lag_' + str(date1.day) + str(date1.month) + str(date1.year) + '.png', format='png')

# 24/06 : no slack water :
no_SW = combined_df_bis.loc[combined_df_bis['HWS'].isna() | combined_df_bis['LWS'].isna()]
print('No slack water with discharge > to ')
print(no_SW.sort_values(by='Q').loc[no_SW['Q'] > 1000])

# Je veux checker la durée de

print('ok')



################################
################################
# 5/08/24
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


Ebb_df = pd.concat([Ebb_HD, Ebb], axis=1)
Ebb_df_abs = Ebb_df.copy()
Ebb_df_abs = Ebb_df_abs.rename(columns={'Amplitude ebb':'Amplitude'})
Ebb_df_abs[['Amplitude', 'Amplitude HD']] = Ebb_df_abs[['Amplitude', 'Amplitude HD']].abs()

Flood_df = pd.concat([Flood_HD, Flood], axis=1)
Flood_df = Flood_df.rename(columns={'Amplitude flood':'Amplitude'})

Ebb_and_flood = pd.concat([Ebb_df_abs, Flood_df], axis=0)

cmap = cmc.cm.batlow
quantile = False
if quantile:
    deb = Ebb_and_flood.quantile([0.25, 0.5, 0.75])['Q'].values
else:
    deb = [500, 1000, 1500]

cond1 = Ebb_and_flood['Q'] < deb[0]
cond2 = (Ebb_and_flood['Q'] > deb[0]) & (Ebb_and_flood['Q'] < deb[1])
cond3 = (Ebb_and_flood['Q'] > deb[1]) & (Ebb_and_flood['Q'] < deb[2])
cond4 = Ebb_and_flood['Q'] > deb[2]
Q1 = Ebb_and_flood.loc[cond1]
Q2 = Ebb_and_flood.loc[cond2]
Q3 = Ebb_and_flood.loc[cond3]
Q4 = Ebb_and_flood.loc[cond4]
label_title = ['<500', '500-1000', '1000-1500', '>1500']
debit = []
for d in deb:
    debit.append(str(np.round(d, 0)))
debit.append('>' + str(np.round(deb[2], 0)))
treshold=0
color_500 = 0.05
color_1000 = 0.3
color_1500 = 0.65
color_2000 = 0.85
list_color = [color_500, color_1000, color_1500, color_2000]


interp = 'polyfit'
fig, ax = plt.subplots(figsize=(16, 10))
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Tidal amplitude at TT (m)', fontsize=fontsize - 2)
ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize - 2)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
filter_condition = Ebb_and_flood['Amplitude HD'] > treshold
p1 = ax.scatter(abs(Ebb_and_flood['Amplitude HD'].loc[filter_condition] / 100),
           abs(Ebb_and_flood['Amplitude'].loc[filter_condition] / 100),
           c=Ebb_and_flood['Q'].loc[filter_condition], cmap=cmap, vmin = 0, vmax = 2200, alpha=0.5)
cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize - 1)
cbar.outline.set_linewidth(0.05)
c = 0
x = np.arange(0, 5, 1)
ax.plot(x,x, c='gray')
# Set the font size for the legend labels
#for label in legend.get_texts():
#    label.set_fontsize(19)  # Set the desired font size
outfile = '2017_Amplitude_TT_vs_HD_Amplitude_'
outfile = outfile + '.png'
fig.savefig(outfile, format='png')
