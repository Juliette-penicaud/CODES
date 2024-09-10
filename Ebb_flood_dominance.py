# 24/10/23 : Je veux calculer le temps de flood et ebb tide a TT et comparer avec HD sur 1 an ou 2 ans pour voir
# influence du débit ou non sur la marée.
# Je veux travailler sur la relation entre % marée (temps) par rapp à % hauteur d'eau (dans estuaire? => Quelles val ?
# avec les données de ADCP Violaine ?) et l'intrusion saline (salinité aux points fixe?) et les vitesses de courants
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
import matplotlib as mpl
from openpyxl import load_workbook
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from scipy import stats
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
mpl.use('Agg')


# COnvention : 100% is HT, 0% is LT, + from L to HT and - from H to LT.
# convention 1 : From HT to LT, we begin by -99% near to HT, -1% near to LT.  CIRCLE
# Trouble convention 1 : representation on graphs ..

# Functions :
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
file = '/home/penicaud/Documents/Data/Décharge_waterlevel/Data_2021-2022.xlsx'

columns_to_load = list(range(25))
# Water level at Trung Trang
df = pd.read_excel(file, sheet_name='water_level_TrungTrang2021-2022', usecols=columns_to_load, skiprows=4, nrows=730)
df = df.rename(columns={'Unnamed: 0': 'Date'})
melted_df = pd.melt(df, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df['Datetime'] = pd.to_datetime(melted_df['Date']) + pd.to_timedelta(melted_df['Hour'], unit='h')
melted_df.sort_values("Datetime", inplace=True)
melted_df = melted_df.rename(columns={'Value': 'Water level Trung Trang'})
melted_df.drop(['Date', 'Hour'], axis=1, inplace=True)

# Water level at Hon Dau
df2 = pd.read_excel(file, sheet_name='sea_level-HonDau_2021-2022', usecols=columns_to_load, skiprows=4, nrows=730)
df2 = df2.rename(columns={'Unnamed: 0': 'Date'})
melted_df2 = pd.melt(df2, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df2['Datetime'] = pd.to_datetime(melted_df2['Date']) + pd.to_timedelta(melted_df2['Hour'], unit='h')
melted_df2.sort_values("Datetime", inplace=True)
melted_df2 = melted_df2.rename(columns={'Value': 'Water level Hon Dau'})
melted_df2.drop(['Date', 'Hour'], axis=1, inplace=True)

df_Q = pd.read_excel(file, sheet_name='Q_TrungTrang_2021-2022',  skiprows=2) # usecols=list(range(2, 5)),
df_Q['Datetime'] = pd.to_datetime(df_Q['Date']) + pd.to_timedelta(df_Q['Hour'], unit='h')
df_Q.drop(['Date', 'Hour'], axis=1, inplace=True)

merged_df = pd.merge(melted_df, melted_df2, on='Datetime', how='inner')
water_levels = pd.merge_asof(merged_df, df_Q, on='Datetime', direction='nearest')

monthly_mean = water_levels.resample('M', on='Datetime').mean()
daily_mean = water_levels.resample('D', on='Datetime').mean()

#################################   EBB AND FLOOD AT TT  #######################################################

print('Hello again ! I try to find the ebb and flood at TT ...')
# I calculate the min and max of the tides AT TT
window_size = 17
local_minima, local_maxima, time_local_min, time_local_max, a, b = \
    find_local_minima(water_levels, water_levels.columns[0], window_size)
# 17/01/24 : I add the SW, but it need to be done on the High frequency, a and b are not exploited arrays

# I create a dataframe of the ebb duration - flood duration.
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

# 22/11 : I add a part to calculate the discharge during the flood and ebb phase to compare it to the datetime
ebb_discharge = []
for i in range(len(Ebb)):
    discharge_phase = np.mean(water_levels['Q'].loc[(water_levels['Datetime'] >= Ebb['Datetime'].loc[i]) &
                                                    (water_levels['Datetime'] < Ebb['Datetime'].loc[i] +
                                                     Ebb['Duration'].loc[i])])
    ebb_discharge.append(discharge_phase)
Ebb['Q'] = ebb_discharge
flood_discharge = []
for i in range(len(Ebb)):
    discharge_phase = np.mean(water_levels['Q'].loc[(water_levels['Datetime'] >= Flood['Datetime'].loc[i]) &
                                                    (water_levels['Datetime'] < Flood['Datetime'].loc[i] +
                                                     Flood['Duration'].loc[i])])
    flood_discharge.append(discharge_phase)
Flood['Q'] = flood_discharge

figure_test_discharge = False
if figure_test_discharge:
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.grid(True, alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
    ax.scatter(daily_mean.index, daily_mean['Q'], label='daily mean', marker='o', color='red')
    ax.scatter(Ebb['Datetime'], Ebb['Q'], label='ebb mean', marker='<', color='black')
    ax.scatter(Flood['Datetime'], Flood['Q'], label='flood mean', marker='>', color='gray')
    ax.scatter(Flood['Datetime'], (Flood['Q'] + Ebb['Q']) / 2, label='mean on flood + ebb', marker='d', color='cyan')
    ax.legend()
    fig.savefig('Dailymean_vs_mean_on_phases_discharge.png', format='png')

######################################################## Same for HD #################################################
col = water_levels.columns[0]
D = 'Datetime'
col2 = water_levels.columns[2]
local_minima_HD, local_maxima_HD, time_local_min_HD, time_local_max_HD, a, b = \
    find_local_minima(water_levels, col2, window_size)

Ebb_HD = pd.DataFrame(
    time_local_max_HD)  # the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
Flood_HD = pd.DataFrame(time_local_min_HD)
Ebb_HD = Ebb_HD.rename(columns={0: 'Datetime'})
Flood_HD = Flood_HD.rename(columns={0: 'Datetime'})
if time_local_max_HD[0] > time_local_min_HD[0]:  # To know which one we need to substract
    print('The first extremum is the minimum data, so it is the flood_HD')
    Flood_HD['Duration'] = time_local_max_HD - time_local_min_HD
    Flood_HD['Amplitude'] = local_maxima_HD - local_minima_HD
    Ebb_HD['Duration'] = np.roll(time_local_min_HD, shift=-1) - time_local_max_HD
    Ebb_HD['Amplitude'] = np.roll(local_minima_HD, shift=-1) - local_maxima_HD
    Ebb_HD.loc[len(Ebb_HD)-1, 'Duration'] = np.nan
    Ebb_HD.loc[len(Ebb_HD)-1, 'Amplitude'] = np.nan
else:
    print('The first extremum is the MAX data, so it is the Ebb_HD')
    Flood_HD['Duration'] = np.roll(time_local_max_HD, shift=-1) - time_local_min_HD
    Flood_HD['Amplitude'] = np.roll(local_maxima_HD, shift=-1) - local_minima_HD
    Flood_HD.loc[len(Flood_HD)-1, 'Duration'] = np.nan
    Flood_HD.loc[len(Flood_HD)-1, 'Amplitude'] = np.nan
    Ebb_HD['Duration'] = time_local_min_HD - time_local_max_HD
    Ebb_HD['Amplitude'] = local_minima_HD - local_maxima_HD

m=8
d=12
print("Duration of Ebb and flood 18 june 2022",
      Ebb_HD[(Ebb_HD['Datetime'].dt.year == 2022) & (Ebb_HD['Datetime'].dt.month == m) &
             (Ebb_HD['Datetime'].dt.day == d)],
      Flood_HD[(Flood_HD['Datetime'].dt.year == 2022) & (Flood_HD['Datetime'].dt.month == m) &
             (Flood_HD['Datetime'].dt.day == d)]
      )
############################### Select the min and max Q values for each day # 28/11/23
local_minima_Q, local_maxima_Q, time_local_min_Q, time_local_max_Q, a, b = \
    find_local_minima(water_levels, water_levels.columns[3], window_size)

df_min_discharge = pd.DataFrame(time_local_min_Q)
# the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
df_max_discharge = pd.DataFrame(time_local_max_Q)
df_min_discharge = df_min_discharge.rename(columns={0: 'Datetime'})
df_max_discharge = df_max_discharge.rename(columns={0: 'Datetime'})
if time_local_max_Q[0] > time_local_min_Q[0]:  # To know which one we need to substract
    print('The first extremum is the minimum data')
    case = 'min'
    df_min_discharge['Duration'] = time_local_max_Q - time_local_min_Q
    df_min_discharge['Value'] = local_minima_Q
    df_max_discharge['Duration'] = np.roll(time_local_min_Q, shift=-1) - time_local_max_Q
    df_max_discharge['Value'] = np.roll(local_maxima_Q, shift=-1)
    df_max_discharge.loc[len(df_max_discharge)-1, 'Duration'] = np.nan
    df_max_discharge.loc[len(df_max_discharge)-1, 'Value'] = np.nan
else:
    print('The first extremum is the MAX data')
    case = 'max'
    df_min_discharge['Duration'] = np.roll(time_local_max_Q, shift=-1) - time_local_min_Q
    df_min_discharge['Value'] = np.roll(local_minima_Q, shift=-1)
    df_min_discharge.loc[len(df_min_discharge)-1, 'Duration'] = np.nan
    df_min_discharge.loc[len(df_min_discharge)-1, 'Value'] = np.nan
    df_max_discharge['Duration'] = time_local_min_Q - time_local_max_Q
    df_max_discharge['Value'] = local_maxima_Q

figure = False
if figure:
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.grid(True, alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
    ax.plot(daily_mean.index, daily_mean['Q'], zorder=1, label='daily mean', color='violet')
    ax.plot(water_levels['Datetime'], water_levels['Q'], label='hourly discharge', zorder=0, color='grey')
    ax.scatter(time_local_min_Q, local_minima_Q, label='min Q values', marker='o', zorder=1, color='black')
    ax.scatter(time_local_max_Q, local_maxima_Q, label='max Q values', marker='o', zorder=1, color='red')
    ax.legend()
    ax.set_xlim(datetime(2021, 1, 1), datetime(2023, 1, 1))
    fig.savefig('Min_and_max_Q_values.png', format='png')

    # PLot of the min max discharges in subplot1 and water levels subplot2 on the 2 years.
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
    ax = axs[0]
    ax.grid('both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
    ax.plot(daily_mean.index, daily_mean['Q'], label='daily mean', color='violet')
    ax.plot(water_levels['Datetime'], water_levels['Q'], label='hourly data', color='grey', zorder=0.1)
    ax.scatter(time_local_min_Q, local_minima_Q, label='min values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max_Q, local_maxima_Q, label='max values', marker='o', color='red', zorder=1)
    ax.set_xlim(datetime(2021, 1, 1), datetime(2023, 1, 1))
    fig.legend()
    ax = axs[1]
    ax.grid('both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Tidal range (m)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='hourly water level',
            color='grey', zorder=0.1)
    ax.plot(daily_mean.index, daily_mean[water_levels.columns[0]] / 100, label='daily mean', color='violet')
    ax.scatter(time_local_min, local_minima / 100, label='min height values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, label='max height values', marker='o', color='red', zorder=1)
    date_form = DateFormatter("%d/%m")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(date_form)
    fig.savefig('Comp_Min_and_max_Qvalues_vs_HeightvaluesTT.png', format='png')

    # 3 subplots for tidal range at HD and TT, and discharge
    fig, axs = plt.subplots(figsize=(18, 15), nrows=3, sharex=True)
    ax = axs[0]
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
    # ax.plot(daily_mean.index, daily_mean['Q'], label='daily mean', color='violet')
    ax.plot(water_levels['Datetime'], water_levels['Q'], label='hourly data', color='grey', zorder=0.1)
    ax.scatter(time_local_min_Q, local_minima_Q, label='min values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max_Q, local_maxima_Q, label='max values', marker='o', color='red', zorder=1)
    fig.legend()
    ax = axs[1]
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Water level at TT (m)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='hourly water level',
            color='grey', zorder=0.1)
    # ax.plot(daily_mean.index, daily_mean[water_levels.columns[0]] / 100, label='daily mean', color='violet')
    ax.scatter(time_local_min, local_minima / 100, label='minimum height values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, label='maximum height values', marker='o', color='red', zorder=1)
    ax = axs[2]
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Water level at HD (m)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[2]] / 100, label='hourly water level',
            color='grey', zorder=0.1)
    # ax.plot(daily_mean.index, daily_mean[water_levels.columns[2]] / 100, label='daily mean', color='violet')
    ax.scatter(time_local_min_HD, local_minima_HD / 100, label='min height values', marker='o', color='black', zorder=1)
    ax.scatter(time_local_max_HD, local_maxima_HD / 100, label='max height values', marker='o', color='red', zorder=1)
    date_form = DateFormatter("%Y/%m/%d")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xlim(datetime(2021, 7, 30), datetime(2021, 8, 3))
    fig.align_labels()
    fig.savefig('Comp_Min_and_max_Qvalues_vs_water_levels_zoom.png', format='png')
    ######################################################################################################
    # 18/01
    # 2 subplots : 1 for tidal range at HD , 2 tidal range TT + discharge
    s = 20
    d1 = 15
    d2 = 25
    m = 6
    year = 2022
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

    fig, axs = plt.subplots(figsize=(20, 12), nrows=2, sharex=True)
    ax = axs[0]
    ax.grid(which='both', alpha=0.5)
    # ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Water elevation (m)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[2]] / 100, label='HD',
            color='grey', zorder=0.1)
    ax.scatter(time_local_min_HD, local_minima_HD / 100, marker='o', color='black', s=s, zorder=1)
    ax.scatter(time_local_max_HD, local_maxima_HD / 100, marker='o', color='black', s=s, zorder=1)

    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT', lw=2,
            color='black', zorder=0.1)
    ax.scatter(time_local_min, local_minima / 100, label='extreme height', marker='o', color='black', s=s, zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, marker='o', color='black', s=s, zorder=1)
    for (x1_HD, x2_HD) in zip(time_local_min_HD[indice_min_wl_HD], time_local_min[indice_min_wl]):
        ax.axvspan(x1_HD, x2_HD, ymin=-10, ymax=10, facecolor='black', alpha=0.2)
    for (x3_HD, x4_HD) in zip(time_local_max_HD[indice_max_wl_HD], time_local_max[indice_max_wl]):
        ax.axvspan(x3_HD, x4_HD, ymin=-10, ymax=10, facecolor='orange', alpha=0.2)
    ax.legend(fontsize=fontsize - 10)

    ax = axs[1]
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Time', fontsize=fontsize - 5)
    ax.set_ylabel('Water elevation at TT (m)', fontsize=fontsize - 5)
    ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT water level',
            color='black', lw=2, zorder=0.1)
    ax.scatter(time_local_min, local_minima / 100, marker='o', color='black', s=s, zorder=1)
    ax.scatter(time_local_max, local_maxima / 100, marker='o', color='black', s=s, zorder=1)

    twin = ax.twinx()
    twin.set_ylabel('Discharge at TT (m$³$/s)', fontsize=fontsize - 5)
    twin.plot(water_levels['Datetime'], water_levels['Q'], label='TT Discharge', ls='--', color='violet',
              zorder=0.1)
    twin.scatter(time_local_min_Q, local_minima_Q, label='extreme values', marker='o', color='black', s=s, zorder=1)
    twin.scatter(time_local_max_Q, local_maxima_Q, marker='o', color='black', s=s, zorder=1)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    twin.legend(lines + lines2, labels + labels2, fontsize=fontsize - 10)

    for (x1, x2) in zip(time_local_min[indice_min_wl], time_local_max_Q[indice_max_Q]):
        ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor='gray', alpha=0.2)
    for (x3, x4) in zip(time_local_min_Q[indice_min_Q], time_local_max[indice_max_wl]):
        ax.axvspan(x3, x4, ymin=-10, ymax=10, facecolor='brown', alpha=0.2)

    date_form = DateFormatter("%d/%m")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xlim(datetime(year, m, d1), datetime(year, m, d2))
    fig.align_labels()
    fig.savefig('Tidal_elevations_Q_lag_' + str(d1) + str(m) + str(year) + '.png', format='png')

#" 28/11 Lag between max of levels and max of currents
lags, corrs, corrs2 = [], [], []
val_interp = 5
year_constraint = True
month_constraint = True
resampled_series = water_levels.copy()
resampled_series = resampled_series.set_index('Datetime')
resampled_series = resampled_series.resample(str(val_interp)+'T').asfreq()  # Resample by adding values every 5T
interpolated_series = resampled_series.interpolate(method='linear')
interpolated_series['Percentage of tide'] = calculate_percentage(time_local_min_HD, time_local_max_HD, interpolated_series)

window_size_interp = int(17*60/val_interp)
local_minima_Q_interp, local_maxima_Q_interp, time_local_min_Q_interp, time_local_max_Q_interp, \
local_SW, time_local_SW = \
    find_local_minima(interpolated_series.reset_index(), interpolated_series.reset_index().columns[3],
                      window_size_interp, interp=True)

year = [2021, 2022]
for y in year:
    months = np.arange(1, 13, 1)
    for month in months:
        a = '0' if month < 10 else ''
        selected_interp = interpolated_series[(interpolated_series.index.month == month) &
                                              (interpolated_series.index.year == y)].copy()
        title2 = a + str(month) + '/' + str(y)
        selected_interp2 = selected_interp.copy()
        lag, corr = calculate_corr2(selected_interp2[["Water level Trung Trang"]], -selected_interp2[["Q"]], year_constraint, y,
                                   month_constraint, month, datetime=False) #CORR2 is needed because we only look into
        # the backward direction, and with only positives values of correlation.
        lags.append(lag)
        corrs.append(corr)

# transformation of lag value to datetime
lags2, lags_datetime = [], []
for l in lags :
    val = abs(l) * val_interp *60
    lags2.append(val)
    lags_datetime.append(timedelta(seconds=val))
lag_discharge = lags2
lags_datetime = pd.TimedeltaIndex(lags_datetime, unit='m')
mean_lags_currents_waterlevel_monthly = pd.DataFrame({'Lags':lag_discharge, 'Lags datetime':lags_datetime, 'Corrs': corrs, 'Q': monthly_mean['Q'].values})
print(mean_lags_currents_waterlevel_monthly.loc[mean_lags_currents_waterlevel_monthly['Lags datetime'] ==
                                                mean_lags_currents_waterlevel_monthly['Lags datetime'].min()])




#30/11 : calcul de la renverse des courants, wui correspond
# I cut because I know the lists do not start and end at the same points
list_time_max_HD = time_local_max_HD[1:]
list_time_min_Q = time_local_min_Q[:-1]

df1 = pd.DataFrame({'Time max HD':list_time_max_HD})#, 'Time min Q': list_time_min_Q})
df2 = pd.DataFrame({'Time min Q': list_time_min_Q})
df1['Nearest datetime min Q'] = df1['Time max HD'].apply(lambda x: min(df2['Time min Q'], key=lambda dt: abs(dt - x)))
df1['Diff'] = df1['Time max HD'] - df1['Nearest datetime min Q']

cond_hours = (df1['Diff'] < timedelta(hours=-4)) & (df1['Diff'] > timedelta(hours=4))
df1 = df1[~cond_hours]

##########################################  Concat the min and max
# 27/10 : Look for a possible impact of the discharge on the amplitudes.
# I want to substract the amplitudes of the closest periods. Perhaps I will need to filter the neap tides.
# Est ce que j'ai besoin de faire sur les 2 ? A priori non. Sauf si je fais l'opération directement sur le tmin et max
# NOT USED
TT_min_max = pd.DataFrame(time_local_min)
TT_min_max = TT_min_max.rename(columns={0: 'Time min'})
TT_min_max['Min'] = local_minima
TT_min_max['Time max'] = time_local_max
TT_min_max['Max'] = local_maxima

HD_min_max = pd.DataFrame(time_local_min_HD)
HD_min_max = HD_min_max.rename(columns={0: 'Time min'})
HD_min_max['Min'] = local_minima_HD
HD_min_max['Time max'] = time_local_max_HD
HD_min_max['Max'] = local_maxima_HD

# filter the neap tides in order to improve the quality of the fit (now the ideal window of 17 is found TODO : redo
filter = True  # if True remove all value < treshold_HD
treshold = 100  # Value in cm
if filter:
    # 26/10 : I want to select the date where the amplitudes are under the tresholds :
    date_neap = Ebb_HD[D].loc[Ebb_HD['Amplitude'] > -treshold].dt.date
    # Je fais sur HD car il y a pas/moins d'influence potentielle du débit
    mask = HD_min_max['Time min'].dt.date.isin(date_neap)
    mask2 = HD_min_max['Time max'].dt.date.isin(date_neap)
    HD_min_max_filtered = HD_min_max[~(mask | mask2)]
    TT_min_max_filtered = TT_min_max[~(mask | mask2)]

# Substraction of the extremas to find the difference of amplitude between HD TT.
result_datetimes_min, result_datetimes_max = [], []
result_values_min, result_values_max = [], []
filter = False
if filter:  # choose if we use the filtered data to do so or not
    df_HD = HD_min_max_filtered
    df_TT = TT_min_max_filtered
else:
    df_HD = HD_min_max
    df_TT = TT_min_max
for index1, row1 in df_HD.iterrows():
    closest_index_max = np.argmin(
        np.abs(df_TT['Time max'] - row1['Time max']))  # Find the closest datetime index
    closest_row2_max = df_TT.iloc[closest_index_max]  # Get the row with the closest datetime from df2
    closest_index_min = np.argmin(
        np.abs(df_TT['Time min'] - row1['Time min']))  # Find the closest datetime index
    closest_row2_min = df_TT.iloc[closest_index_min]  # Get the row with the closest datetime from df2

    result_datetime_max = row1['Time max']
    result_value_max = row1['Max'] - closest_row2_max['Max']  # Subtract the values
    result_datetime_min = row1['Time min']
    result_value_min = row1['Min'] - closest_row2_min['Min']  # Subtract the values

    result_datetimes_max.append(result_datetime_max)
    result_values_max.append(result_value_max)
    result_datetimes_min.append(result_datetime_max)
    result_values_min.append(result_value_max)

# Create a new DataFrame with the results
result_df_min = pd.DataFrame({'Time min': result_datetimes_min, 'Diff min': result_values_min})
result_df_max = pd.DataFrame({'Time max': result_datetimes_max, 'Diff max': result_values_max})
# 1/11 : not used because I prefer using the Tidal range at HD vs at TT

# J'ajoute le débit journalier le plus proche de Ebb et Flood datetime
result_value_Q = []
for index1, row1 in Ebb.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_Q.append(daily_mean['Q'].iloc[closest_index])
Ebb['Q'] = result_value_Q
result_value_Qf = []
for index1, row1 in Flood.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_Qf.append(daily_mean['Q'].iloc[closest_index])
Flood['Q'] = result_value_Qf
result_value_QHD = []
for index1, row1 in Ebb_HD.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_QHD.append(daily_mean['Q'].iloc[closest_index])
Ebb_HD['Q'] = result_value_QHD
result_value_QfHD = []
for index1, row1 in Flood_HD.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_QfHD.append(daily_mean['Q'].iloc[closest_index])
Flood_HD['Q'] = result_value_QfHD

# 30/10 Je cherche à caractériser la différence d'amplitude par rapport à l'amplitude (spring tide ou neap tide en gros)
# et du débit. On peut quantifier la différence d'amplitude par rapport à l'amplitude avec la couleur associée au débit
# Je vais rajouter le rapport moyen pour des amplitudes et débit particuliers.
# 1ere catégorie : les débits de 500m3 en 500m3 (Cat 1. <500, Cat2 500<x<1000, cat3 1000<x<1500, cat4 >1500
# 2e catégorie : les amplitudes à HD. Cat A : moins de 1m, Cat B : moins de 2m cat C :<3m, catD : >3m
# Avec la fenetre de 17, on a normalement toutes les valeurs de ebb et flood qui correspondent, donc on peut faire
# un seul tableau avec HD et TT
# Useful only in order to change the column names of HD
Ebb_HD_copy = Ebb_HD.copy()
Ebb_HD_copy = Ebb_HD_copy.rename(columns={col: col + ' HD' for col in Ebb_HD_copy.columns})
Ebb_df = pd.concat([Ebb_HD_copy, Ebb], axis=1)
# 5/12 : I add a column of the minimum of current the nearest to Ebb at TT datetime, corresponding to the maximum of flooding
# 'Nearest datetime min/max Q' recense la datetime de la décharge min i.e vitesse min pour ensuite calculer ce lag.
Ebb_df['Nearest datetime min Q'] = Ebb_df['Datetime'].apply(lambda x: min(time_local_min_Q, key=lambda dt: abs(dt - x)))
Ebb_df['Diff init'] = Ebb_df['Nearest datetime min Q'] - Ebb_df['Datetime']
Ebb_df['Diff'] = Ebb_df['Datetime'] - Ebb_df['Nearest datetime min Q']
Ebb_df['Diff water level'] = Ebb_df['Datetime'] - Ebb_df['Datetime HD']

Flood_HD_copy = Flood_HD.copy()
Flood_HD_copy = Flood_HD_copy.rename(columns={col: col + ' HD' for col in Flood_HD_copy.columns})
Flood_df = pd.concat([Flood_HD_copy, Flood], axis=1)
# 5/12 : I add a column of the minimum of current the nearest to Ebb at TT datetime, corresponding to the maximum of flooding
Flood_df['Nearest datetime max Q'] = Flood_df['Datetime'].apply(lambda x: min(time_local_max_Q, key=lambda dt: abs(dt - x)))
Flood_df['Diff init'] = Flood_df['Nearest datetime max Q'] - Flood_df['Datetime']
Flood_df['Diff'] = Flood_df['Datetime'] - Flood_df['Nearest datetime max Q']
Flood_df['Diff water level'] = Flood_df['Datetime'] - Flood_df['Datetime HD']

# 17/01 : je rajoute une durée entre la date de HT et le plus proche slack water.
old_slack_water = True
if old_slack_water :
    print('ATTENTION OLD METHOD FOR SLACK WATER')
    Flood_df['Nearest slack water'] = Flood_df['Datetime'].apply(lambda x: min(time_local_SW, key=lambda dt: abs(dt - x)))
    Ebb_df['Nearest slack water'] = Ebb_df['Datetime'].apply(lambda x: min(time_local_SW, key=lambda dt: abs(dt - x)))
    Flood_df['Diff slack water low water'] = Flood_df['Nearest slack water'] - Flood_df['Datetime']
    Ebb_df['Diff slack water high water'] = Ebb_df['Nearest slack water'] - Ebb_df['Datetime']
    Ebb_df['Diff slack water high water filtered'] = handle_outliers(Ebb_df, 'Diff slack water high water')
    Flood_df['Diff slack water low water filtered'] = handle_outliers(Flood_df, 'Diff slack water low water')
else :
    # 24/05 : modification, pour avoir une colonne avec la slack water la plus proche avant et après la valeur de LW
    # Je calcule la SW la plus proche avant et après l'heure de marée haute.
    # 28/05 : Je vais modifier pour avoir les nearest slack water en fonction de Vmin ou Vmax
    # Il faut que je mette les Vmin et Vmax dans le même tableau, comme ca je choisis bien le slack water qui correspond à l'entre deux.
    Flood_df['Nearest slack water before Vmax'] = Flood_df['Nearest datetime max Q'].apply(
        lambda x: min([dt for dt in time_local_SW if x > dt], key=lambda dt: (x - dt), default=None))
    Flood_df['Diff HWS-Vmax'] = Flood_df['Nearest datetime max Q'] - Flood_df['Nearest slack water before Vmax']
    crit_duration_slack = pd.Timedelta(hours=14)
    nombre_valeurs_exclues = Flood_df['Diff HWS-Vmax'].iloc[np.where(Flood_df['Diff HWS-Vmax'] > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood previous slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff low water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Flood_df.loc[Flood_df['Diff HWS-Vmax'] > crit_duration_slack, 'Diff HWS-Vmax'] = pd.NaT

    Flood_df['Nearest slack water after Vmax'] = Flood_df['Nearest datetime max Q'].apply(
        lambda x: min([dt for dt in time_local_SW if dt > x], key=lambda dt: (dt - x), default=None))
    Flood_df['Diff Vmax-LWS'] = Flood_df['Nearest slack water after Vmax'] - Flood_df['Nearest datetime max Q']
    # Il est possible de ne pas avoir de slack water pendant des jours si pas de retournement: neap-tide ou faible Q
    crit_duration_slack = pd.Timedelta(hours=15)
    nombre_valeurs_exclues = Flood_df['Diff Vmax-LWS'].iloc[np.where(Flood_df['Diff Vmax-LWS']  > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood next slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff low water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Flood_df.loc[Flood_df['Diff Vmax-LWS'] > crit_duration_slack, 'Diff Vmax-LWS'] = pd.NaT

    # Je sélectionne la slack water la plus proche entre celle avant et celle après
    Flood_df['Diff low water-nearest slack water'] = Flood_df[['Diff low water-previous slack water', 'Diff low water-next slack water']].min(axis=1)

    # Meme chose mais pour Ebb
    Ebb_df['Nearest slack water before HW'] = Ebb_df['Datetime'].apply(
        lambda x: min([dt for dt in time_local_SW if x > dt], key=lambda dt: (x - dt), default=None))
    Ebb_df['Diff high water-previous slack water'] = Ebb_df['Datetime'] - Ebb_df['Nearest slack water before HW']
    nombre_valeurs_exclues = Ebb_df['Diff high water-previous slack water'].iloc[np.where(Ebb_df['Diff high water-previous slack water']  > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood previous slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff high water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Ebb_df.loc[Ebb_df['Diff high water-previous slack water'] > crit_duration_slack, 'Diff high water-previous slack water'] = pd.NaT

    Ebb_df['Nearest slack water after HW'] = Ebb_df['Datetime'].apply(
        lambda x: min([dt for dt in time_local_SW if dt > x], key=lambda dt: (dt - x), default=None))
    Ebb_df['Diff high water-next slack water'] = Ebb_df['Nearest slack water after HW'] - Ebb_df['Datetime']
    # Il est possible de ne pas avoir de slack water pendant plusieurs jours si pas de retournement: neap-tide ou faible Q
    nombre_valeurs_exclues = Ebb_df['Diff high water-next slack water'].iloc[np.where(Ebb_df['Diff high water-next slack water']  > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood next slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff high water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Ebb_df.loc[Ebb_df['Diff high water-next slack water'] > crit_duration_slack, 'Diff high water-previous slack water'] = pd.NaT

    # Je sélectionne la slack water la plus proche entre celle avant et celle après
    Ebb_df['Diff high water-nearest slack water'] = Ebb_df[['Diff high water-previous slack water', 'Diff high water-next slack water']].min(axis=1)


    """
    Flood_df['Nearest slack water before LW'] = Flood_df['Datetime'].apply(
        lambda x: min([dt for dt in time_local_SW if x > dt], key=lambda dt: (x - dt), default=None))
    Flood_df['Diff low water-previous slack water'] = Flood_df['Datetime'] - Flood_df['Nearest slack water before LW']
    crit_duration_slack = pd.Timedelta(hours=16)
    nombre_valeurs_exclues = Flood_df['Diff low water-previous slack water'].iloc[np.where(Flood_df['Diff low water-previous slack water']  > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood previous slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff low water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Flood_df.loc[Flood_df['Diff low water-previous slack water'] > crit_duration_slack, 'Diff low water-previous slack water'] = pd.NaT

    Flood_df['Nearest slack water after LW'] = Flood_df['Datetime'].apply(
        lambda x: min([dt for dt in time_local_SW if dt > x], key=lambda dt: (dt - x), default=None))
    Flood_df['Diff low water-next slack water'] = Flood_df['Nearest slack water after LW'] - Flood_df['Datetime']
    # Il est possible de ne pas avoir de slack water pendant des jours si pas de retournement: neap-tide ou faible Q
    nombre_valeurs_exclues = Flood_df['Diff low water-next slack water'].iloc[np.where(Flood_df['Diff low water-next slack water']  > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood next slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff low water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Flood_df.loc[Flood_df['Diff low water-next slack water'] > crit_duration_slack, 'Diff low water-previous slack water'] = pd.NaT

    # Je sélectionne la slack water la plus proche entre celle avant et celle après
    Flood_df['Diff low water-nearest slack water'] = Flood_df[['Diff low water-previous slack water', 'Diff low water-next slack water']].min(axis=1)

    # Meme chose mais pour Ebb
    Ebb_df['Nearest slack water before HW'] = Ebb_df['Datetime'].apply(
        lambda x: min([dt for dt in time_local_SW if x > dt], key=lambda dt: (x - dt), default=None))
    Ebb_df['Diff high water-previous slack water'] = Ebb_df['Datetime'] - Ebb_df['Nearest slack water before HW']
    nombre_valeurs_exclues = Ebb_df['Diff high water-previous slack water'].iloc[np.where(Ebb_df['Diff high water-previous slack water']  > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood previous slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff high water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Ebb_df.loc[Ebb_df['Diff high water-previous slack water'] > crit_duration_slack, 'Diff high water-previous slack water'] = pd.NaT

    Ebb_df['Nearest slack water after HW'] = Ebb_df['Datetime'].apply(
        lambda x: min([dt for dt in time_local_SW if dt > x], key=lambda dt: (dt - x), default=None))
    Ebb_df['Diff high water-next slack water'] = Ebb_df['Nearest slack water after HW'] - Ebb_df['Datetime']
    # Il est possible de ne pas avoir de slack water pendant plusieurs jours si pas de retournement: neap-tide ou faible Q
    nombre_valeurs_exclues = Ebb_df['Diff high water-next slack water'].iloc[np.where(Ebb_df['Diff high water-next slack water']  > crit_duration_slack)].shape[0]
    print('Nombre de valeurs exclues des flood next slack water', nombre_valeurs_exclues)
    # Replace values in 'Diff high water-previous slack water' with np.nan if they are greater than crit_duration_slack
    Ebb_df.loc[Ebb_df['Diff high water-next slack water'] > crit_duration_slack, 'Diff high water-previous slack water'] = pd.NaT

    # Je sélectionne la slack water la plus proche entre celle avant et celle après
    Ebb_df['Diff high water-nearest slack water'] = Ebb_df[['Diff high water-previous slack water', 'Diff high water-next slack water']].min(axis=1)
    """

# Si je veux regarder si TR ou Q semble jouer sur qui de la SW précédante ou suivante est la plus proche de l'extrema de niveau d'eau :
# Ebb_df[['Diff high water-previous slack water', 'Diff high water-next slack water']].loc[(Ebb_df['Diff high water-next slack water'] - Ebb_df['Diff high water-previous slack water']) < pd.Timedelta(hours=0)]
# Ebb_df[['Q', 'Amplitude HD']].loc[(Ebb_df['Diff high water-next slack water'] - Ebb_df['Diff high water-previous slack water']) < pd.Timedelta(hours=0)].sort_values(by='Q')

for df in Flood_df, Ebb_df :
    cond1 = (abs(df['Amplitude HD']) < 100)
    cond2 = (100 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 200)
    cond3 = (200 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 300)
    cond4 = (abs(df['Amplitude HD']) > 300)
    Q1 = df.loc[cond1]
    Q2 = df.loc[cond2]
    Q3 = df.loc[cond3]
    Q4 = df.loc[cond4]
    for Q in [Q1, Q2, Q3, Q4]:
        print('ATTENTION AU NOM DES COLONNES §§')
        print(Q[Q.columns[-2]].min(), Q[Q.columns[-2]].max(),
              Q[Q.columns[-2]].mean(),Q[Q.columns[-2]].std())
        print(Q[Q.columns[10]].min(), Q[Q.columns[10]].max(),
              Q[Q.columns[10]].mean(),Q[Q.columns[10]].std())

print(Ebb_df[(Ebb_df['Datetime HD'].dt.year == 2022) & (Ebb_df['Datetime HD'].dt.month == 6) & (Ebb_df['Datetime HD'].dt.day >= 16) & (Ebb_df['Datetime HD'].dt.day <= 18)]['Amplitude HD'])
print(Flood_df[(Flood_df['Datetime HD'].dt.year == 2022) & (Flood_df['Datetime HD'].dt.month == 6) & (Flood_df['Datetime HD'].dt.day >= 16) & (Flood_df['Datetime HD'].dt.day <= 18)]['Amplitude HD'])

figure_max_current_discharge = False
if figure_max_current_discharge :
    # Lag between the difference vs Q and vs tidal range at HD
    cmap = cmc.cm.batlow
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle('Difference between max of current and of discharge ', fontsize=fontsize)
    ax = axs[0]
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
    ax.set_ylabel('Lag (hours)', fontsize=fontsize - 2)
    y_ebb = ((handle_outliers(Ebb_df.dropna(), 'Diff')).dt.total_seconds() / 3600).astype(float)
    y_flood = ((handle_outliers(Flood_df.dropna(), 'Diff')).dt.total_seconds() / 3600).astype(float)
    mean_ebb = (np.round(np.nanmean(y_ebb), 1))
    mean_flood = (np.round(np.nanmean(y_flood), 1))
    # ax.scatter(Ebb_df.dropna()['Q'], y_ebb, marker='o', color='grey',  label='Ebb duration ' + str(mean_ebb) + 'h')
    # ax.scatter(Flood_df.dropna()['Q'], y_flood, marker='<', color='red', label='Flood duration ' + str(mean_flood) + 'h')
    p1 = ax.scatter(Ebb_df.dropna()['Q'], y_ebb, marker='o',  cmap=cmap, c=abs(Ebb_df.dropna()['Amplitude HD']/100),
                    vmin=0 , vmax=4, label='Ebb duration ' + str(mean_ebb) + 'h')
    ax.scatter(Flood_df.dropna()['Q'], y_flood, marker='<',  cmap=cmap, c=Flood_df.dropna()['Amplitude HD']/100,
               vmin=0, vmax = 4, label='Flood duration ' + str(mean_flood) + 'h')
    cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
    cbar.set_label(label='Tidal range HD (m)', fontsize=fontsize - 1)
    cbar.outline.set_linewidth(0.05)
    x=np.arange(50,2200, 10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_df.dropna()['Q'], y_ebb)
    print(p_value)
    ax.plot(x, slope*x+intercept, color='black', label='r='+str(np.round(r_value,3)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df.dropna()['Q'], y_flood)
    print(p_value)
    ax.plot(x, slope*x+intercept, color='blue', label='r='+str(np.round(r_value,3)))
    ax.legend()

    ax = axs[1]
    ax.grid(which='both', alpha=0.5)
    ax.set_ylabel('Lag (hours)', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal range HD (m)', fontsize=fontsize - 2)
    #ax.scatter(-Ebb_df.dropna()['Amplitude HD']/100, y_ebb, marker='o', color='grey')#, label='Ebb duration ' + str(mean_ebb) + 'h')
    #ax.scatter(Flood_df.dropna()['Amplitude HD']/100, y_flood, marker='<', color='red')#,label='Flood duration ' + str(mean_flood) + 'h')
    p1 = ax.scatter(abs(Ebb_df.dropna()['Amplitude HD']/100), y_ebb, marker='o', cmap=cmap, c=Ebb_df.dropna()['Q'],
                    vmin=0 , vmax=2200, label='Ebb duration ' + str(mean_ebb) + 'h')
    ax.scatter(Flood_df.dropna()['Amplitude HD']/100, y_flood, marker='<', cmap=cmap, c=Flood_df.dropna()['Q'],
               vmin=0, vmax = 2200, label='Flood duration ' + str(mean_flood) + 'h')
    cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
    cbar.set_label(label='Q (m$^3$/s)', fontsize=fontsize - 1)
    cbar.outline.set_linewidth(0.05)
    x=np.arange(0,4, 0.10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(abs(Ebb_df.dropna()['Amplitude HD'])/100, y_ebb)
    print(p_value)
    ax.plot(x, slope*x+intercept, color='black', label='r='+str(np.round(r_value,3)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df.dropna()['Amplitude HD']/100, y_flood)
    print(p_value)
    ax.plot(x, slope*x+intercept, color='blue', label='r='+str(np.round(r_value,3)))
    ax.legend()
    fig.savefig('Difference_min_current_min_height.png')

figure_diffrences_waterlevel = False
if figure_diffrences_waterlevel :
    condition = 'Tidal range'
    quantile = False
    if quantile:
        deb = Ebb_df.quantile([0.25, 0.5, 0.75])['Q'].values
    else:
        deb = [500, 1000, 1500]
    color_500 = 0.05
    color_1000 = 0.3
    color_1500 = 0.65
    color_2000 = 0.85
    list_color = [color_500, color_1000, color_1500, color_2000]
    list_marker = { 0: {'name': 'Ebb', 'marker':'o'},
                   1 : {'name': 'Flood', 'marker' : '<'}}
    cmap = cmc.cm.hawaii_r

    fig, axs = plt.subplots(figsize=(18, 10) , nrows=2, sharex=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # fig.suptitle('Difference between water level datetime at TT and HD ', fontsize=fontsize)

    count_phase = 0
    for phase_df in [Ebb_df, Flood_df]:
        ax = axs[count_phase]
        ax.grid(which='both', alpha=0.5)
        ax.set_ylabel('Lag (hours)', fontsize=fontsize - 2)
        if condition == 'Discharge':
            ax.set_xlabel('Tidal range HD (m)', fontsize=fontsize - 2)
            cond1 = phase_df['Q'] < deb[0]
            cond2 = (phase_df['Q'] > deb[0]) & (phase_df['Q'] < deb[1])
            cond3 = (phase_df['Q'] > deb[1]) & (phase_df['Q'] < deb[2])
            cond4 = phase_df['Q'] > deb[2]
        elif condition == 'Tidal range':
            ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
            cond1 = (abs(phase_df['Amplitude HD']) < 100)
            cond2 = (100 < abs(phase_df['Amplitude HD'])) & (abs(phase_df['Amplitude HD']) <= 200)
            cond3 = (200 < abs(phase_df['Amplitude HD'])) & (abs(phase_df['Amplitude HD']) <= 300)
            cond4 = (abs(phase_df['Amplitude HD']) > 300)

        Q1 = phase_df.loc[cond1]
        Q2 = phase_df.loc[cond2]
        Q3 = phase_df.loc[cond3]
        Q4 = phase_df.loc[cond4]
        debit = []
        for d in deb:
            debit.append(str(np.round(d, 0)))
        debit.append('>' + str(np.round(deb[2], 0)))

        c = 0
        filter = False
        for Q in [Q1, Q2, Q3, Q4]:
            if filter :
                QY = (handle_outliers(Q.dropna(), 'Diff water level').dt.total_seconds() / 3600).astype(float)
                if condition == 'Discharge':
                    QX = (handle_outliers(Q.dropna(), 'Amplitude HD')) / 100
                elif condition == 'Tidal range':
                    QX = (handle_outliers(Q.dropna(), 'Q'))

            else :
                QY = (Q.dropna()['Diff water level'].dt.total_seconds()/ 3600).astype(float)
                if condition == 'Discharge':
                    QX = Q.dropna()['Amplitude HD']/100
                elif condition == 'Tidal range':
                    QX = Q.dropna()['Q']
            mean = (np.round(np.nanmean(QY), 2))
            if c == 0 :
                ax.scatter(abs(QX), QY, marker=list_marker[count_phase]['marker'], color = cmap(list_color[c]),
                           label = list_marker[count_phase]['name'])
            else :
                ax.scatter(abs(QX), QY, marker=list_marker[count_phase]['marker'], color=cmap(list_color[c]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(QX, QY)
            print(p_value)
            if condition == 'Discharge':
                label = str(debit[c] + " m$³$/s, " + str(np.round(slope, 2)) + ' x + ' + str(np.round(intercept, 2)) +
                            ', r=' + str(np.round(r_value, 3)) + ' N=' + str(QX.count()))
                x = np.arange(0, 4, 0.10)

            elif condition == 'Tidal range':
                label = str('HD tidal range <'+str(c+1)+'m'+ str(np.round(slope, 2)) + ' x + ' + str(np.round(intercept, 2)) +
                            ', r=' + str(np.round(r_value, 3)) + ' N=' + str(QX.count()))
                x = np.arange(200, 2000, 10)

            ax.plot(x, slope * x + intercept, color=cmap(list_color[c]), label = label)# label='r=' + str(np.round(r_value, 3)))
            c = c+1
            ax.legend()
        count_phase = count_phase+1
    outfile = 'Difference_water_level_datetime_withclasses'
    if filter :
        outfile = outfile +'_filter'
    if quantile :
        outfile = outfile + '_quantile'
    else :
        outfile = outfile + '_500m3'
    outfile = outfile + '.png'
    fig.savefig(outfile, format = 'png')


    # Figure of 2 subplots for correlation of lags between the water levels vs the discharge1. and 2. tidal range
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle('Difference between water level datetime at TT and HD ', fontsize=fontsize)
    ax = axs[0]
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
    ax.set_ylabel('Lag (hours)', fontsize=fontsize - 2)
    y_ebb = (handle_outliers(Ebb_df.dropna(), 'Diff water level').dt.total_seconds() / 3600).astype(float)
    y_flood = (handle_outliers(Flood_df.dropna(), 'Diff water level').dt.total_seconds() / 3600).astype(float)
    mean_ebb = (np.round(np.nanmean(y_ebb), 1))
    mean_flood = (np.round(np.nanmean(y_flood), 1))
    # ax.scatter(Ebb_df.dropna()['Q'], y_ebb, marker='o', color='grey',  label='Ebb duration ' + str(mean_ebb) + 'h')
    # ax.scatter(Flood_df.dropna()['Q'], y_flood, marker='<', color='red', label='Flood duration ' + str(mean_flood) + 'h')
    p1 = ax.scatter(Ebb_df.dropna()['Q'], y_ebb, marker='o', cmap=cmap,
                    c=abs(Ebb_df.dropna()['Amplitude HD'] / 100),
                    vmin=0, vmax=4, label='mean lag between maxima water levels = ' + str(mean_ebb) + 'h')
    ax.scatter(Flood_df.dropna()['Q'], y_flood, marker='<', cmap=cmap, c=Flood_df.dropna()['Amplitude HD'] / 100,
               vmin=0, vmax=4, label='mean lag between minima water levels = ' + str(mean_flood) + 'h')
    cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
    cbar.set_label(label='Tidal range HD (m)', fontsize=fontsize - 1)
    cbar.outline.set_linewidth(0.05)
    x = np.arange(50, 2200, 10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_df.dropna()['Q'], y_ebb)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='black', label='r=' + str(np.round(r_value, 3)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df.dropna()['Q'], y_flood)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='blue', label='r=' + str(np.round(r_value, 3)))
    ax.legend()

    ax = axs[1]
    ax.grid(which='both', alpha=0.5)
    ax.set_ylabel('Lag (hours)', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal range HD (m)', fontsize=fontsize - 2)
    # ax.scatter(-Ebb_df.dropna()['Amplitude HD']/100, y_ebb, marker='o', color='grey')#, label='Ebb duration ' + str(mean_ebb) + 'h')
    # ax.scatter(Flood_df.dropna()['Amplitude HD']/100, y_flood, marker='<', color='red')#,label='Flood duration ' + str(mean_flood) + 'h')
    p1 = ax.scatter(abs(Ebb_df.dropna()['Amplitude HD'] / 100), y_ebb, marker='o', cmap=cmap,
                    c=Ebb_df.dropna()['Q'],
                    vmin=0, vmax=2200)
    ax.scatter(Flood_df.dropna()['Amplitude HD'] / 100, y_flood, marker='<', cmap=cmap, c=Flood_df.dropna()['Q'],
               vmin=0, vmax=2200)
    cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
    cbar.set_label(label='Q (m$^3$/s)', fontsize=fontsize - 1)
    cbar.outline.set_linewidth(0.05)
    x = np.arange(0, 4, 0.10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(abs(Ebb_df.dropna()['Amplitude HD']) / 100,
                                                                   y_ebb)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='black', label='r=' + str(np.round(r_value, 3)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df.dropna()['Amplitude HD'] / 100, y_flood)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='blue', label='r=' + str(np.round(r_value, 3)))
    ax.legend()
    fig.savefig('Difference_water_level_datetime_filteroutliers.png')



    # Figure of 2 subplots for correlation of lags between the duration of flood and ebb vs the discharge1. and 2. tidal range
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle('Evolution of duration of ebb and flood ', fontsize=fontsize)
    ax = axs[0]
    ax.grid(which='both', alpha=0.5)
    ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
    ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
    if filter :
        y_ebb = (handle_outliers(Ebb_df.dropna(), 'Duration').dt.total_seconds() / 3600).astype(float)
        y_flood = (handle_outliers(Flood_df.dropna(), 'Duration').dt.total_seconds() / 3600).astype(float)
        condition_ebb = abs(Ebb_df['Amplitude HD']) > 100
        condition_flood = Flood_df['Amplitude HD'] > 100
        y_ebb = (Ebb_df.dropna()['Duration'].loc[condition_ebb].dt.total_seconds() / 3600).astype(float)
        y_flood = (Flood_df.dropna()['Duration'].loc[condition_flood].dt.total_seconds() / 3600).astype(float)
    else :
        y_ebb = (Ebb_df.dropna()['Duration'].dt.total_seconds() / 3600).astype(float)
        y_flood = (Flood_df.dropna()['Duration'].dt.total_seconds() / 3600).astype(float)
    mean_ebb = (np.round(np.nanmean(y_ebb), 2))
    mean_flood = (np.round(np.nanmean(y_flood), 2))
    # ax.scatter(Ebb_df.dropna()['Q'], y_ebb, marker='o', color='grey',  label='Ebb duration ' + str(mean_ebb) + 'h')
    # ax.scatter(Flood_df.dropna()['Q'], y_flood, marker='<', color='red', label='Flood duration ' + str(mean_flood) + 'h')
    p1 = ax.scatter(Ebb_df.dropna()['Q'].loc[condition_ebb], y_ebb, marker='o', cmap=cmap,
                    c=abs(Ebb_df.dropna()['Amplitude HD'].loc[condition_ebb] / 100),
                    vmin=0, vmax=4, label='mean ebb duration = ' + str(mean_ebb) + 'h')
    ax.scatter(Flood_df.dropna()['Q'].loc[condition_flood], y_flood, marker='<', cmap=cmap, c=Flood_df.dropna()['Amplitude HD'].loc[condition_flood] / 100,
               vmin=0, vmax=4, label='mean flood duration = ' + str(mean_flood) + 'h')
    cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
    cbar.set_label(label='Tidal range HD (m)', fontsize=fontsize - 1)
    cbar.outline.set_linewidth(0.05)
    x = np.arange(50, 2200, 10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_df.dropna()['Q'].loc[condition_ebb], y_ebb)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='black', label='r=' + str(np.round(r_value, 3)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df.dropna()['Q'].loc[condition_flood], y_flood)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='blue', label='r=' + str(np.round(r_value, 3)))
    ax.legend()

    ax = axs[1]
    ax.grid(which='both', alpha=0.5)
    ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal range HD (m)', fontsize=fontsize - 2)
    # ax.scatter(-Ebb_df.dropna()['Amplitude HD']/100, y_ebb, marker='o', color='grey')#, label='Ebb duration ' + str(mean_ebb) + 'h')
    # ax.scatter(Flood_df.dropna()['Amplitude HD']/100, y_flood, marker='<', color='red')#,label='Flood duration ' + str(mean_flood) + 'h')
    p1 = ax.scatter(abs(Ebb_df.dropna()['Amplitude HD'].loc[condition_ebb] / 100), y_ebb, marker='o', cmap=cmap,
                    c=Ebb_df.dropna()['Q'].loc[condition_ebb],
                    vmin=0, vmax=2200)
    ax.scatter(Flood_df.dropna()['Amplitude HD'].loc[condition_flood] / 100, y_flood, marker='<', cmap=cmap, c=Flood_df.dropna()['Q'].loc[condition_flood],
               vmin=0, vmax=2200)
    cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
    cbar.set_label(label='Q (m$^3$/s)', fontsize=fontsize - 1)
    cbar.outline.set_linewidth(0.05)
    x = np.arange(0, 4, 0.10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(abs(Ebb_df.dropna()['Amplitude HD'].loc[condition_ebb]) / 100,
                                                                   y_ebb)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='black', label='r=' + str(np.round(r_value, 3)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df.dropna()['Amplitude HD'].loc[condition_flood] / 100, y_flood)
    print(p_value)
    ax.plot(x, slope * x + intercept, color='blue', label='r=' + str(np.round(r_value, 3)))
    ax.legend()
    outfile = 'Asymmetry_vsDischarge_Tidalrange'
    if filter :
        outfile = outfile +'_filteredamplitude'
    outfile = outfile + '.png'
    fig.savefig(outfile, format = 'png')


list_HD_Ebb, list_TT_Ebb, list_count_Ebb = calculate_categories_mean(Ebb_df)
list_HD_Flood, list_TT_Flood, list_count_Flood = calculate_categories_mean(Flood_df)

list_HD_Ebb_quartile, list_TT_Ebb_quartile, list_count_Ebb_quartile, list_deb_Ebb = calculate_quartiles_mean(Ebb_df)
list_HD_Flood_quartile, list_TT_Flood_quartile, list_count_Flood_quartile, list_deb_Flood = \
    calculate_quartiles_mean(Flood_df)
# With this method, we 1. select the amplitude we want to study, 2. calculate the quartiles of discharge,
# 3. calculate the Tidal range at TT and HD with the condition of amplitude and discharge. ==> Useful to keep the
# same values in all means, in order to statiscally account for any change.
Ebb_df_abs = Ebb_df.copy()
Ebb_df_abs[['Amplitude', 'Amplitude HD']] = Ebb_df[['Amplitude', 'Amplitude HD']].abs()
Ebb_and_flood = pd.concat([Ebb_df_abs, Flood_df], axis=0)
list_HD_both, list_TT_both, list_count_both = calculate_categories_mean(Ebb_and_flood)
list_HD_both_quartile, list_TT_both_quartile, list_count_both_quartile, list_deb_both = \
    calculate_quartiles_mean(Ebb_and_flood)

# Create a df
tableau_quartile_Ebb = pd.DataFrame({'Amplitude HD': list_HD_Ebb_quartile, 'Amplitude TT': list_TT_Ebb_quartile,
                                     'Q': list_deb_Ebb, 'N': list_count_Ebb_quartile})
tableau_quartile_Flood = pd.DataFrame({'Amplitude HD': list_HD_Flood_quartile, 'Amplitude TT': list_TT_Flood_quartile,
                                       'Q': list_deb_Flood, 'N': list_count_Flood_quartile})
tableau_quartile_both = pd.DataFrame({'Amplitude HD': list_HD_both_quartile, 'Amplitude TT': list_TT_both_quartile,
                                      'Q': list_deb_both, 'N': list_count_both_quartile})
tableau_quartile_Ebb['TT/HD'] = np.round(tableau_quartile_Ebb['Amplitude TT'] / tableau_quartile_Ebb['Amplitude HD'], 2)
tableau_quartile_Flood['TT/HD'] = np.round(
    tableau_quartile_Flood['Amplitude TT'] / tableau_quartile_Flood['Amplitude HD'], 2)
tableau_quartile_both['TT/HD'] = np.round(tableau_quartile_both['Amplitude TT'] / tableau_quartile_both['Amplitude HD'],
                                          2)
tableau_discharge_category_both = pd.DataFrame({'Amplitude HD': list_HD_both, 'Amplitude TT': list_TT_both,
                                                'N': list_count_both})
tableau_discharge_category_both['TT/HD'] = np.round(
    tableau_discharge_category_both['Amplitude TT'] / tableau_discharge_category_both['Amplitude HD'], 2)

tableau_all = pd.concat([tableau_quartile_Ebb, tableau_quartile_Flood], axis=1)
tableau_all.to_csv('Tableau_Ebb_Flood_quartiles.csv')

# Stats over tableau_both
for i in [0, 4, 8, 12]:
    for amp in ['Amplitude HD', 'Amplitude TT']:
        val = (abs(tableau_quartile_both[i:i + 4][amp].std())) / abs(tableau_quartile_both[i:i + 4][amp].mean()) * 100
        print(abs(np.round(val, 2)), '%')

# stats of variations :
for df in [tableau_quartile_Ebb, tableau_quartile_Flood]:
    for i in [0, 4, 8, 12]:
        for amp in ['Amplitude HD', 'Amplitude TT']:
            val = (abs(df[i:i + 4][amp].std())) / abs(df[i:i + 4][amp].mean()) * 100
            print(abs(np.round(val, 2)), '%')

tidal_range_Ebb = [0.66, 1.56, 2.53, 3.33]
tidal_range_Flood = [0.63, 1.54, 2.55, 3.34]
mean_ratio_Ebb = [0.88, 0.82, 0.71, 0.64]
mean_ratio_Flood = [0.85, 0.81, 0.71, 0.65]
mean_ratio_both = [0.87, 0.81, 0.71, 0.64]
tidal_range_both = [0.64, 1.54, 2.54, 3.33]
figure = False
if figure:
    # Figure of the correlation TT/HD amplitude vs HD amplitude (i.e tidal range) over the mean per categories
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle('ratio vs tidal range', fontsize=fontsize)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('mean ratio TT/HD amplitude over 4 categories quartiles discharges', fontsize=fontsize - 2)
    ax.set_xlabel('Mean tidal range', fontsize=fontsize - 2)
    ax.scatter(tidal_range_Ebb, mean_ratio_Ebb, color='grey', alpha=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tidal_range_Ebb, mean_ratio_Ebb)
    label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 4))
    # ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
    x = np.arange(0, 10, 1)
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color='grey', label=label)
    ax.set_xlim(0, 5)
    legend = ax.legend()
    # Set the font size for the legend labels
    for label in legend.get_texts():
        label.set_fontsize(15)  # Set the desired font size
    fig.savefig('mean_ratio_vs_tidal_range_Ebb.png', format='png')

    # From the "raw data" :
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle('ratio vs tidal range', fontsize=fontsize)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('TT/HD', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal range at HD', fontsize=fontsize - 2)
    ax.scatter(abs(Ebb_df['Amplitude HD'] / 100), abs(Ebb_df['Amplitude'] / Ebb_df['Amplitude HD']), color='grey',
               alpha=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(abs(Ebb_df['Amplitude HD'] / 100), abs(
        Ebb_df['Amplitude'] / Ebb_df['Amplitude HD']))
    label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 4))
    # ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
    x = np.arange(0, 5, 1)
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color='grey', label=label)
    legend = ax.legend()
    ax.set_xlim(0, 5)
    # Set the font size for the legend labels
    for label in legend.get_texts():
        label.set_fontsize(15)  # Set the desired font size
    fig.savefig('ratio_vs_amplitude_at_HD_Ebb.png', format='png')
    # Sme for flood :
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle('ratio vs tidal range', fontsize=fontsize)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('TT/HD', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal range at HD', fontsize=fontsize - 2)
    ax.scatter(abs(Flood_df['Amplitude HD'] / 100), abs(Flood_df['Amplitude'] / Flood_df['Amplitude HD']), color='grey',
               alpha=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(abs(Flood_df['Amplitude HD'].dropna() / 100), abs(
        Flood_df['Amplitude'].dropna() / Flood_df['Amplitude HD'].dropna()))
    label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 4))
    # ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
    x = np.arange(0, 5, 1)
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color='grey', label=label)
    legend = ax.legend()
    ax.set_xlim(0, 5)
    # Set the font size for the legend labels
    for label in legend.get_texts():
        label.set_fontsize(15)  # Set the desired font size
    fig.savefig('ratio_vs_amplitude_at_HD_Flood.png', format='png')

    # Same for both
    filter_condition = Ebb_and_flood['Amplitude HD'] > treshold
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('TT/HD', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal range at HD', fontsize=fontsize - 2)
    ax.scatter(abs(Ebb_and_flood['Amplitude HD'].loc[filter_condition] / 100),
               abs(Ebb_and_flood['Amplitude'].loc[filter_condition] /
                   Ebb_and_flood['Amplitude HD'].loc[filter_condition]), color='grey', alpha=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        abs(Ebb_and_flood['Amplitude HD'].loc[filter_condition].dropna() / 100),
        abs(Ebb_and_flood['Amplitude'].loc[filter_condition].dropna() /
            Ebb_and_flood['Amplitude HD'].loc[filter_condition].dropna()))
    label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 4))
    # ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
    x = np.arange(0, 5, 1)
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color='grey', label=label)
    legend = ax.legend()
    ax.set_xlim(0, 5)
    # Set the font size for the legend labels
    for label in legend.get_texts():
        label.set_fontsize(15)  # Set the desired font size
    fig.savefig('ratio_vs_amplitude_at_HD_both.png', format='png')

    # Figure of Amplitude of TT f(ampl_HD, Q)
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle('Amplitude of TT vs tidal range', fontsize=fontsize)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('Tidal range at TT', fontsize=fontsize - 2)
    ax.set_xlabel('Tidal range at HD', fontsize=fontsize - 2)
    ax.scatter(tableau_discharge_category_both['Amplitude HD'], abs(tableau_discharge_category_both['Amplitude TT']),
               color='grey', alpha=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tableau_discharge_category_both['Amplitude HD'],
                                                                   abs(tableau_discharge_category_both['Amplitude TT']))
    label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 4))
    # ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
    x = np.arange(0, 5, 1)
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color='grey', label=label)
    legend = ax.legend()
    ax.set_xlim(0, 5)
    # Set the font size for the legend labels
    for label in legend.get_texts():
        label.set_fontsize(15)  # Set the desired font size
    fig.savefig('Amplitude_TT_vs_HD_Amplitude.png', format='png')

# I select the data I want to study
year_constraint = True
year = 2022
all_year = True
figure = False
save = False
filter = False  # if True remove all value < treshold_HD, and take in acount only the filtered values
treshold = 100  # Value in cm
if all_year:
    year = [2021, 2022]
month_constraint = True
months = np.arange(1, 2, 1)
title = 'Water levels '
Wat_HD = water_levels.columns[2]
D = water_levels.columns[1]
Wat_TT = water_levels.columns[0]
lags, corrs, corrs2 = [], [], []
lags_discharge, corrs_discharge, corrs2_discharge = [], [], []
lags_discharge_HD, corrs_discharge_HD, corrs2_discharge_HD = [], [], []

# 26/10 : I resampled the data in order to have a 5 mn value, to have a finer value of the lag and see if
# it is the reason why its quality decreases with discharge increase
val_interp=5
resampled_series = water_levels.copy()
resampled_series = resampled_series.set_index('Datetime')
resampled_series = resampled_series.resample(str(val_interp)+'T').asfreq()  # Resample by adding values every 5T
interpolated_series = resampled_series.interpolate(method='linear')
interpolated_series['Percentage of tide'] = calculate_percentage(time_local_min_HD, time_local_max_HD, interpolated_series)

if year_constraint and not month_constraint:
    selected_data = water_levels[(water_levels['Datetime'].dt.year == year)]
    selected_ebb = Ebb_df[(Ebb_df['Datetime'].dt.year == year)]
    selected_flood = Flood_df[(Flood_df['Datetime'].dt.year == year)]
    title2 = str(year)
elif not year_constraint and not month_constraint:
    print('no constraint on year or month, I take the whole series')
    selected_data = water_levels
    selected_ebb = Ebb_df
    selected_flood = Flood_df
    title2 = '2021-2022'
else:
    lags, corrs, corrs2 = [], [], []
    if all_year:
        for y in year:
            months = np.arange(1, 13, 1)
            for month in months:
                a = '0' if month < 10 else ''
                selected_interp = interpolated_series[(interpolated_series.index.month == month) &
                                                      (interpolated_series.index.year == y)].copy()
                selected_data = water_levels[(water_levels['Datetime'].dt.year == y) &
                                             (water_levels['Datetime'].dt.month == month)].copy()
                selected_ebb = Ebb_df[(Ebb_df['Datetime'].dt.year == y) & (Ebb_df['Datetime'].dt.month == month)].copy()
                selected_flood = Flood_df[
                    (Flood_df['Datetime'].dt.year == y) & (Flood_df['Datetime'].dt.month == month)].copy()
                title2 = a + str(month) + '/' + str(y)
                if filter:
                    selected_ebb2 = selected_ebb[selected_ebb['Amplitude HD'] < -treshold].copy()
                    selected_flood2 = selected_flood[selected_flood['Amplitude HD'] > treshold].copy()
                    # 26/10 : I want to select the date where the amplitudes are under the tresholds :
                    date_neap = selected_ebb[D].loc[selected_ebb['Amplitude HD'] > -treshold].dt.date
                    # We only take Ebb_df, Flood_df is supposed to follow the same trend ,
                    # selected_flood[D].loc[selected_flood['Amplitude'] < treshold]
                    mask = selected_data[D].dt.date.isin(date_neap)
                    selected_data2 = selected_data[~mask]  # Apply the mask to filter out rows from
                    selected_interp['date_only'] = selected_interp.index.date
                    selected_interp2 = selected_interp[~selected_interp['date_only'].isin(date_neap.values)]
                else:
                    selected_ebb2 = selected_ebb.copy()
                    selected_flood2 = selected_flood.copy()
                    selected_data2 = selected_data.copy()
                    selected_interp2 = selected_interp.copy()

                lag, corr = calculate_corr(selected_interp2[[Wat_TT]], selected_interp2[[Wat_HD]], year_constraint, y,
                                           month_constraint, month, datetime=False) # lag between water levels
                lag_discharge, corr_discharge = calculate_corr2(selected_interp2[["Water level Trung Trang"]],
                                                                 -selected_interp2[["Q"]], year_constraint, y,
                                                                 month_constraint, month, datetime=False)
                lag_dischargeHD, corr_dischargeHD = calculate_corr2(selected_interp2[["Water level Hon Dau"]],
                                                                 -selected_interp2[["Q"]], year_constraint, y,
                                                                 month_constraint, month, datetime=False)
                # CORR2 is needed because we only look into the backward direction,
                # and with only positives values of correlation.
                lags.append(lag)
                corrs.append(corr)
                lags_discharge.append(lag_discharge)
                corrs_discharge.append(corr_discharge)
                lags_discharge_HD.append(lag_dischargeHD)
                corrs_discharge_HD.append(corr_dischargeHD)
    else:
        for month in months:
            a = '0' if month < 10 else ''
            if year_constraint and month_constraint:
                selected_data = water_levels[(water_levels['Datetime'].dt.year == year) &
                                             (water_levels['Datetime'].dt.month == month)]
                selected_ebb = Ebb_df[
                    (Ebb_df['Datetime HD'].dt.year == year) & (Ebb_df['Datetime HD'].dt.month == month)]
                selected_flood = Flood_df[
                    (Flood_df['Datetime HD'].dt.year == year) & (Flood_df['Datetime HD'].dt.month == month)]
                title2 = a + str(month) + '/' + str(year)
            elif month_constraint and not year_constraint:
                selected_data = water_levels[water_levels['Datetime'].dt.month == month]
                selected_ebb = Ebb_df[(Ebb_df['Datetime'].dt.month == month)]
                selected_flood = Flood_df[(Flood_df['Datetime'].dt.month == month)]
                title2 = a + str(month)

        # Plot of the Ebb and Flood duration only with selected data.
        if filter:
            selected_ebb2 = selected_ebb[selected_ebb['Amplitude'] < -treshold]
            selected_flood2 = selected_flood[selected_flood['Amplitude'] > treshold]
            # 26/10 : I want to select the date where the amplitudes are under the tresholds :
            date_neap = selected_ebb[D].loc[selected_ebb['Amplitude'] > -treshold].dt.date
            # We only take Ebb, Flood is supposed to follow the same trend ,
            # selected_flood[D].loc[selected_flood['Amplitude'] < treshold]
            mask = selected_data[D].dt.date.isin(date_neap)
            # Apply the mask to filter out rows from
            selected_data2 = selected_data[~mask]
        else:
            selected_ebb2 = selected_ebb.copy()
            selected_flood2 = selected_flood.copy()
            selected_data2 = selected_data.copy()
        lag, corr = calculate_corr(selected_data2[[Wat_TT, D]], selected_data2[[Wat_HD, D]], year_constraint, year,
                                   month_constraint, month)
        lags.append(lag)
        corrs.append(corr)

    for c in corrs:
        corrs2.append(np.round(c, 4))
    if figure:
        # Figure month by month, only relevant if year_constraint == True
        fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig.suptitle('Ebb-Flood duration at TT ' + title2, fontsize=fontsize)
        ax = axs[0]
        ax.grid(which='both', alpha=0.5)
        twin = ax.twinx()
        twin.set_ylabel('Ebb/Flood', fontsize=fontsize - 2)
        twin.set_ylim(0.25, 2.5)
        ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
        ax.set_ylim(5, 20)
        y_ebb = (selected_ebb2['Duration'].dt.total_seconds() / 3600).astype(float)
        y_flood = (selected_flood2['Duration'].dt.total_seconds() / 3600).astype(float)
        rap = y_ebb.reset_index()['Duration'] / y_flood.reset_index()['Duration']
        # Pour le moment je ne m'occupe pas des dates des EBb et Flood que je divise
        mean_ebb = (np.round(np.nanmean(y_ebb), 1))
        mean_flood = (np.round(np.nanmean(y_flood), 1))
        mean_rap = (np.round(np.nanmean(rap), 2))
        ax.plot(selected_ebb2['Datetime'], y_ebb, marker='o', color='grey', label='Ebb duration ' + str(mean_ebb) + 'h',
                lw=2)
        ax.plot(selected_flood2['Datetime'], y_flood, marker='<', color='red',
                label='Flood duration ' + str(mean_flood) + 'h', lw=2)
        longest_df = max([selected_flood2, selected_ebb2], key=len)
        twin.plot(longest_df['Datetime'], rap, marker='^', color='k', label='Ebb/Flood ' + str(mean_rap), lw=2)
        # Ebb_and_flood = pd.merge_asof(selected_flood2, selected_ebb2, on='Datetime')
        # rap = Ebb_and_flood['Duration_y'] / Ebb_and_flood['Duration_x']
        # twin.plot(Ebb_and_flood['Datetime'] + timedelta(days=1), rap, marker='^', color='k', label='Ebb/Flood '+str(mean_rap), lw=2)
        legend1 = ax.legend(loc='upper left')
        legend2 = twin.legend(loc='upper right')
        # twin.add_artist(legend2)

        ax = axs[1]
        ax.grid(which='both', alpha=0.5)
        ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
        ax.set_ylim(-1, 2.5)
        date_form = DateFormatter("%d/%m")  # Define the date format
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.xaxis.set_major_formatter(date_form)
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.plot(selected_data['Datetime'], selected_data[col] / 100, color='grey', alpha=0.5, lw=2)
        # if filter :
        #    ax.plot(selected_ebb2['Datetime'], -selected_ebb2['Amplitude'] / 100, marker='o', color='grey',
        #            label='Ebb duration ' + str(mean_ebb) + 'h', lw=2)
        #    ax.plot(selected_flood2['Datetime'], selected_flood2['Amplitude'] / 100, marker='<', color='red',
        #            label='Flood duration ' + str(mean_flood) + 'h', lw=2)
        if month == 12:
            ax.set_xlim(datetime(y, month, 1), datetime(y + 1, 1, 2))
        else:
            ax.set_xlim(datetime(y, month, 1), datetime(y, month + 1, 2))
        if save:
            outfile = 'Ebb_Flood_duration_' + str(month) + str(y)
            if filter:
                outfile = outfile + '_filtered_at_' + str(treshold)
            outfile = outfile + '.png'
            fig.savefig(outfile, format='png')

if all_year:
    lags_hours = []
    for l in lags:
        delta_minutes = timedelta(minutes=5)
        delta_seconds = delta_minutes * l
        lags_hours.append(delta_seconds)

lags2, lags_datetime = [], []
for l in lags_discharge :
    val = abs(l) * val_interp *60
    lags2.append(val)
    lags_datetime.append(timedelta(seconds=val))
lags_datetime = pd.TimedeltaIndex(lags_datetime, unit='m')

mean_lags_currents_waterlevel_monthly_filtered = pd.DataFrame({'Lags at TT': lags2, 'Lags datetime at TT':lags_datetime,
                                                      'Corrs at TT': corrs_discharge, 'Q': monthly_mean['Q'].values})
lags2, lags_datetime = [], []
for l in lags_discharge_HD :
    val = abs(l) * val_interp *60
    lags2.append(val)
    lags_datetime.append(timedelta(seconds=val))
lags_datetime = pd.TimedeltaIndex(lags_datetime, unit='m')
mean_lags_currents_waterlevel_monthly_filtered['Lags at HD'] = lags2
mean_lags_currents_waterlevel_monthly_filtered['Lags datetime at HD'] = lags_datetime
mean_lags_currents_waterlevel_monthly_filtered['Corrs at HD'] = corrs_discharge_HD

figure = False
if figure :
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.grid(True, alpha=0.5)
    ax.set_ylabel('Lag (hours)', fontsize=fontsize - 5)
    ax.set_xlabel('Discharge (m$³$/s)', fontsize=fontsize - 5)
    y_lag = (mean_lags_currents_waterlevel_monthly_filtered['Lags datetime at TT'].dt.total_seconds() /3600).astype(float)
    ax.scatter(mean_lags_currents_waterlevel_monthly_filtered['Q'], y_lag , marker='o', color='black')
    slope, intercept, r_value, p_value, std_err = stats.linregress(mean_lags_currents_waterlevel_monthly_filtered['Q'], y_lag)
    label = str(str(np.round(slope,2))+' x + ' +str(np.round(intercept, 0)) + ', r='+ str(np.round(r_value,3)))
    x = np.arange(200,1250,10)
    ax.plot(x, slope*x + intercept, label=label)
    #ax.set_xlim(datetime(2021, 1, 1), datetime(2023, 1, 1))
    ax.legend()
    fig.savefig('Lag_reverse_currents_vs_discharge_monthly.png', format='png')



###############################

# Create a df of Q, lags and corrs
data_phase_Q = monthly_mean[['Q']].copy()
data_phase_Q["lag"] = lags_hours
data_phase_Q["corr"] = corrs

# normally the corresponding date is just the filtered selected interp whole time series
treshold = 100
date_neap = Ebb_df[D + ' HD'].loc[abs(Ebb_df['Amplitude HD']) < treshold].dt.date
date_neap2 = Flood_df[D + ' HD'].loc[Flood_df['Amplitude HD'] < treshold].dt.date
# We only take Ebb, Flood is supposed to follow the same trend ,
# selected_flood[D].loc[selected_flood['Amplitude'] < treshold]
mask = water_levels[D].dt.date.isin(date_neap)
mask2 = water_levels[D].dt.date.isin(date_neap2)
water_levels_filtered = water_levels[~(mask | mask2)]  # Apply the mask to filter out rows from
interpolated_series['date_only'] = interpolated_series.index.date
filtered_interpolated_series = interpolated_series[~(interpolated_series['date_only'].isin(date_neap.values) |
                                                     interpolated_series['date_only'].isin(date_neap2.values))]
mask = Ebb_df[D + ' HD'].dt.date.isin(date_neap)
mask2 = Ebb_df[D + ' HD'].dt.date.isin(date_neap2)
Ebb_df_filtered = Ebb_df[~(mask | mask2)]
mask = Flood_df[D + ' HD'].dt.date.isin(date_neap)
mask2 = Flood_df[D + ' HD'].dt.date.isin(date_neap2)
Flood_df_filtered = Flood_df[~(mask | mask2)]

################################################ TEST TO FIND THE BEST WINDOW
# 30/10 : check the best lag window to detect min and max : 17 is ok !
window_check = False
if window_check:
    w = np.arange(14, 30)
    for window_size in w:
        local_minima, local_maxima, time_local_min, time_local_max, a, b = \
            find_local_minima(water_levels, water_levels.columns[0], window_size)

        # I create a dataframe of the ebb duration - flood duration.
        Ebb = pd.DataFrame(
            time_local_max)  # the starting datetime is the beginning of the ebb i.e : max water levels at TT
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
        local_minima_HD, local_maxima_HD, time_local_min_HD, time_local_max_HD, a, b = \
            find_local_minima(water_levels, col2, window_size)

        Ebb_HD = pd.DataFrame(
            time_local_max_HD)  # the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
        Flood_HD = pd.DataFrame(time_local_min_HD)
        Ebb_HD = Ebb_HD.rename(columns={0: 'Datetime'})
        Flood_HD = Flood_HD.rename(columns={0: 'Datetime'})
        if time_local_max_HD[0] > time_local_min_HD[0]:  # To know which one we need to substract
            print('The first extremum is the minimum data, so it is the flood_HD')
            Flood_HD['Duration'] = time_local_max_HD - time_local_min_HD
            Flood_HD['Amplitude'] = local_maxima_HD - local_minima_HD
            Ebb_HD['Duration'] = np.roll(time_local_min_HD, shift=-1) - time_local_max_HD
            Ebb_HD['Amplitude'] = np.roll(local_minima_HD, shift=-1) - local_maxima_HD
            Ebb_HD.loc[len(Ebb_HD)-1, 'Duration'] = np.nan
            Ebb_HD.loc[len(Ebb_HD) - 1, 'Amplitude'] = np.nan
    else:
            print('The first extremum is the MAX data, so it is the Ebb_HD')
            Flood_HD['Duration'] = np.roll(time_local_max_HD, shift=-1) - time_local_min_HD
            Flood_HD['Amplitude'] = np.roll(local_maxima_HD, shift=-1) - local_minima_HD
            Flood_HD.loc[len(Flood_HD)-1, 'Duration'] = np.nan
            Flood_HD.loc[len(Flood_HD) - 1, 'Amplitude'] = np.nan
            Ebb_HD['Duration'] = time_local_min_HD - time_local_max_HD
            Ebb_HD['Amplitude'] = local_minima_HD - local_maxima_HD
    if len(Ebb) == len(Ebb_HD):
        print('window size ', window_size)

####################################" FIGURES   ###################################################################
# 1.
# 24/10 : I check if I well detected the min and max with the function
# TT only
color_500 = 0.05
color_1000 = 0.3
color_1500 = 0.65
color_2000 = 0.85
list_color = [color_500, color_1000, color_1500, color_2000]

# cmap = sns.color_palette("Spectral_r", as_cmap=True)
cmap = cmc.cm.hawaii_r
cmap = cmc.cm.batlow

# 13/11 : Je veux enregistrer les niveaux d'eau les plus haut et les plus bas
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Water level at TT', fontsize=fontsize)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (cm)', fontsize=fontsize - 2)
date_form = DateFormatter("%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
# plt.gcf().autofmt_xdate()
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 2)
ax.plot(df['Datetime'], df[col], color='grey', alpha=0.5, lw=0.5)
ax.scatter(time_local_min, local_minima, color='red', s=s)
ax.scatter(time_local_max, local_maxima, color='brown', s=s)  # ax.set_xlim(datetime(2022, 5, 1), datetime(2022, 9, 1))
# ax.plot(monthly_mean.index, monthly_mean['Q'], color='black', label='Monthly mean at TT')
fig.savefig('water_levels_min_max_TT_zoom.png', format='png')

# CHeck if min and max are well detected AT HD
# HD ONLY
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Water level at HD', fontsize=fontsize)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (cm)', fontsize=fontsize - 2)
date_form = DateFormatter("%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
# plt.gcf().autofmt_xdate()
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 2)
ax.plot(df['Datetime'], df[col2], color='grey', alpha=0.5, lw=0.5)
ax.scatter(time_local_min_HD, local_minima_HD, color='red', s=s)
ax.scatter(time_local_max_HD, local_maxima_HD, color='brown', s=s)
# CHeck if min and max are well detected AT BOTH STATIONS
# Both station, month by month, year by year
for year in [2021, 2022]:
    for month in np.arange(1, 13, 1):
        a = '0' if month < 10 else ''
        fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
        fig.suptitle('Water level comparison ' + a + str(month) + '/' + str(year), fontsize=fontsize)
        ax = axs[0]
        ax.grid(True, alpha=0.5)
        ax.set_ylabel('Water level (cm)', fontsize=fontsize - 2)
        date_form = DateFormatter("%d/%m")  # Define the date format
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(date_form)
        ax.set_xlabel('Time', fontsize=fontsize - 2)
        # Hon Dau
        ax.plot(water_levels['Datetime'], water_levels[col2], color='grey', alpha=0.5, lw=0.5)
        ax.scatter(time_local_min_HD, local_minima_HD, color='red', s=s)
        ax.scatter(time_local_max_HD, local_maxima_HD, color='brown', s=s)  # Trung Trang
        ax = axs[1]
        ax.grid(True, alpha=0.5)
        ax.plot(water_levels['Datetime'], water_levels[col], color='grey', alpha=0.5, lw=0.5)
        ax.scatter(time_local_min, local_minima, color='red', s=s)
        ax.scatter(time_local_max, local_maxima, color='brown', s=s)
        if month == 12:
            ax.set_xlim(datetime(year, month, 1), datetime(year + 1, 1, 1))
        else:
            ax.set_xlim(datetime(year, month, 1), datetime(year, month + 1, 1))
        fig.savefig('Min_and_max_detection_v2_at_both_stations_' + a + str(month) + str(year) + '.png', format='png')

# Plot the amplification (=ratio TT/HD)
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
date_form = DateFormatter("%d/%m")  # Define the date format
date_form = DateFormatter("%m/%Y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=5))
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(date_form)
ax = axs[0]
ax.xaxis.set_major_formatter(date_form)
ax.grid(True, alpha=0.5)
ax.set_ylabel('TT/HD', fontsize=fontsize - 2, labelpad=20)
ax.plot(Ebb_and_flood.sort_values(by='Datetime')['Datetime'],
        Ebb_and_flood.sort_values(by='Datetime')['Amplitude'] / Ebb_and_flood.sort_values(by='Datetime')[
            'Amplitude HD'], color='grey', lw=1)
ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level at HD (m)', fontsize=fontsize - 2)
ax.set_xlabel('Time', fontsize=fontsize - 2)
ax.plot(water_levels['Datetime'], water_levels[col2] / 100, color='grey', lw=1)
ax.set_xlim(datetime(2022, 1, 1), datetime(2022, 12, 31))
fig.savefig('ratio_vs_datetime.png', format='png')

# Plot the amplification (=ratio TT/HD) vs datetime, vs tidal range and vs Discharge
fig, axs = plt.subplots(figsize=(12, 17), nrows=2, sharey=True)
ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Amplification', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range HD (m)', fontsize=fontsize - 2)
slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_and_flood['Amplitude HD'].dropna()/100,
        Ebb_and_flood.dropna()['Amplitude'] / Ebb_and_flood.dropna()['Amplitude HD'])
x=np.arange(0,4,0.1)
label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ',  r=' + str(np.round(r_value, 2))
ax.plot(x,slope*x+intercept, color='k', zorder=10, label = label)
ax.scatter(Ebb_and_flood['Amplitude HD']/100,
        Ebb_and_flood['Amplitude'] / Ebb_and_flood[
            'Amplitude HD'], color='grey', lw=1)
ax.legend()
ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Amplification', fontsize=fontsize - 2)
ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
ax.scatter(Ebb_and_flood['Q'], Ebb_and_flood['Amplitude'] / Ebb_and_flood['Amplitude HD'] , color='grey', lw=1)
slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_and_flood.dropna()['Q'], Ebb_and_flood.dropna()['Amplitude'] / Ebb_and_flood.dropna()['Amplitude HD'])
x=np.arange(0,2300,10)
label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ',  r=' + str(np.round(r_value, 2))
ax.plot(x,slope*x+intercept, color='k', zorder=10, label = label)
ax.legend()
fig.align_labels()
fig.tight_layout()
fig.savefig('amplification_vs_discharge_and_tidal_range.png', format='png')

#################################
# 2.
# Plot de la corrélation fonction du débit.
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_ylabel('correlation r', fontsize=fontsize)
ax.set_xlabel('Q (m$^3$/s)', fontsize=fontsize)
ax.set_xlim(200, 1300)
ax.set_ylim(0.82, 1)
ax.scatter(data_phase_Q['Q'], data_phase_Q['corr'], color='black', s=s)
ax.grid(True, alpha=0.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(data_phase_Q['Q'], data_phase_Q['corr'])
label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 2))
# ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
x = np.arange(100, 1500, 1)
ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color='grey', label=label)
ax.legend()
fig.savefig('Debit_correlation_avecvalinterp_andneapfiltered_relation_2years.png', format='png')

# Plot du lag fonction du débit
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_ylabel('phase lag (s)', fontsize=fontsize)
ax.set_xlabel('Q (m$^3$/s)', fontsize=fontsize)
ax.set_xlim(200, 1300)
# ax.set_ylim(0.82,1)
y_second = [td.total_seconds() for td in data_phase_Q['lag']]
ax.scatter(data_phase_Q['Q'], y_second, color='black', s=s)
ax.grid(True, alpha=0.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(data_phase_Q['Q'], y_second)
label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 2))
# ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
x = np.arange(100, 1500, 1)
ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color='grey', label=label)
ax.legend()
fig.savefig('Debit_lag_avecvalinterp_andneapfiltered_relation_2years.png', format='png')
# Puis il faudra aussi la différence d'amplitude ou indicateur d'amplitude fonction du débit


########################
# 3.
# Flood and Ebb duration
if not month_constraint:
    # Plot of the ebb and flood duration, with also the Ebb/flood and under, the water level at TT
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle('Ebb-Flood duration at TT ' + str(y), fontsize=fontsize)
    ax = axs[0]
    ax.grid(which='both', alpha=0.5)
    twin = ax.twinx()
    twin.set_ylabel('Ebb/Flood', fontsize=fontsize - 2)
    twin.set_ylim(0.25, 2.5)
    ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
    ax.set_ylim(5, 20)
    y_ebb = (selected_ebb2['Duration'].dt.total_seconds() / 3600).astype(float)
    y_flood = (selected_flood2['Duration'].dt.total_seconds() / 3600).astype(float)
    rap = y_ebb.reset_index()['Duration'] / y_flood.reset_index()[
        'Duration']  # Pour le moment je ne m'occupe pas des dates des EBb et Flood que je divise
    mean_ebb = (np.round(np.nanmean(y_ebb), 1))
    mean_flood = (np.round(np.nanmean(y_flood), 1))
    mean_rap = (np.round(np.nanmean(rap), 2))
    ax.plot(selected_ebb2['Datetime'], y_ebb, marker='o', color='grey', label='Ebb duration ' + str(mean_ebb) + 'h',
            lw=2)
    ax.plot(selected_flood2['Datetime'], y_flood, marker='<', color='red',
            label='Flood duration ' + str(mean_flood) + 'h', lw=2)
    longest_df = max([selected_flood2, selected_ebb2], key=len)
    twin.plot(longest_df['Datetime'], rap, marker='^', color='k', label='Ebb/Flood ' + str(mean_rap), lw=2)
    # Ebb_and_flood = pd.merge_asof(selected_flood2, selected_ebb2, on='Datetime')
    # rap = Ebb_and_flood['Duration_y'] / Ebb_and_flood['Duration_x']
    # twin.plot(Ebb_and_flood['Datetime'] + timedelta(days=1), rap, marker='^', color='k', label='Ebb/Flood '+str(mean_rap), lw=2)
    legend1 = ax.legend(loc='upper left')
    legend2 = twin.legend(loc='upper right')
    # twin.add_artist(legend2)

    ax = axs[1]
    ax.grid(which='both', alpha=0.5)
    ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
    ax.set_ylim(-1, 2.5)
    date_form = DateFormatter("%d/%m")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xlabel('Time', fontsize=fontsize)
    ax.plot(selected_data['Datetime'], selected_data[col] / 100, color='grey', alpha=0.5, lw=2)
    # ax.set_xlim(datetime(y, 1, 1), datetime(y + 1, 1, 1))
    # if filter :
    #    ax.plot(selected_ebb2['Datetime'], -selected_ebb2['Amplitude'] / 100, marker='o', color='grey',
    #            label='Ebb duration ' + str(mean_ebb) + 'h', lw=2)
    #    ax.plot(selected_flood2['Datetime'], selected_flood2['Amplitude'] / 100, marker='<', color='red',
    #            label='Flood duration ' + str(mean_flood) + 'h', lw=2)
    if save:
        outfile = 'Ebb_Flood_duration_all_year' + str(y)
        if filter:
            outfile = outfile + '_filtered_at_' + str(treshold)
        outfile = outfile + '.png'
        fig.savefig(outfile, format='png')

# 24/10 : I plot the duration of the Ebb and Flood tides AND the water amplitudes.
fig, axs = plt.subplots(figsize=(18, 18), nrows=4, sharex=True)
y = 2022
selected_ebb = Ebb_df[(Ebb_df['Datetime'].dt.year == y)]
selected_flood = Flood_df[(Flood_df['Datetime'].dt.year == y)]
selected_ebb2 = selected_ebb[selected_ebb['Amplitude HD'] < -treshold].copy()
selected_flood2 = selected_flood[selected_flood['Amplitude HD'] > treshold].copy()
# AT HD
y_ebb = (selected_ebb2['Duration HD'].dt.total_seconds() / 3600).astype(float)
y_flood = (selected_flood2['Duration HD'].dt.total_seconds() / 3600).astype(float)
mean_ebb = int(np.round(np.nanmean(y_ebb), 2))
mean_flood = int(np.round(np.nanmean(y_flood), 2))
ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Duration at HD (hours)', fontsize=20)
ax.scatter(selected_ebb2['Datetime'], y_ebb, marker='o', color='grey', label='Ebb duration ' + str(mean_ebb) + 'h')
ax.scatter(selected_flood2['Datetime'], y_flood, marker='<', color='red',
           label='Flood duration ' + str(mean_flood) + 'h')
ax.legend(fontsize=12)
ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level at HD (m)', fontsize=20)
ax.plot(water_levels['Datetime'], water_levels[col2] / 100, color='grey', alpha=0.5, lw=0.5)
# AT TT
y_ebb = (selected_ebb2['Duration'].dt.total_seconds() / 3600).astype(float)
y_flood = (selected_flood2['Duration'].dt.total_seconds() / 3600).astype(float)
mean_ebb = (np.round(np.nanmean(y_ebb), 2))
mean_flood = (np.round(np.nanmean(y_flood), 2))
ax = axs[2]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Duration at TT (hours)', fontsize=20)
ax.scatter(selected_ebb2['Datetime'], y_ebb, marker='o', color='grey', label='Ebb duration ' + str(mean_ebb) + 'h')
ax.scatter(selected_flood2['Datetime'], y_flood, marker='<', color='red',
           label='Flood duration ' + str(mean_flood) + 'h')
ax.legend(fontsize=12)
ax = axs[3]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level at TT (m)', fontsize=20)
ax.plot(water_levels['Datetime'], water_levels[col] / 100, color='grey', alpha=0.5, lw=0.5)

outfile = 'Ebb_Flood_duration_both_station_filtered_' + str(y)
zoom = False
if zoom:
    month_space = 1
if zoom:
    ax.set_xlim(datetime(2022, 5, 1), datetime(2022, 10, 1))
    outfile = outfile + '_zoom'
else:
    ax.set_xlim(datetime(y, 1, 1), datetime(y, 12, 31))
    month_space = 3
date_form = DateFormatter("%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=20)

outfile = outfile + '.png'
fig.savefig(outfile, format='png')

#### 1 plot for 2 years
# Plot of the 1. Amplitudes difference 2. the water level
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
fig.suptitle('Difference of amplitudes HD-TT', fontsize=fontsize)
ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Amplitude difference (m)', fontsize=fontsize - 2)
twin = ax.twinx()
twin.set_ylabel('Discharge (m$^3$/s)', fontsize=fontsize - 2)
y_min = result_df_min['Diff min'] / 100
y_max = result_df_max['Diff max'] / 100
mean_min = np.round(np.nanmean(y_min), 2)
mean_max = np.round(np.nanmean(y_max), 2)
ax.scatter(result_df_min['Time min'], y_min, marker='o', color='grey',
           label='Amplitude difference at min ' + str(mean_min) + 'm')
# twin.scatter(selected_daily_wat.index, selected_daily_wat['Q'], marker='d', color='blue',
#           label='Daily discharge at TT' + str(mean_dis) + 'm$^3$/s')
# ax.scatter(result_df_max['Time max'].loc[condition_max], y_max, marker='<', color='red',
#           label='Amplitude difference at max ' + str(mean_max) + 'm')
ax.legend()
ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
date_form = DateFormatter("%d/%m")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 2)
ax.plot(water_levels['Datetime'], water_levels[col] / 100 -
        np.average(water_levels[col] / 100), color='grey', alpha=0.5, zorder=3, lw=0.5, label='Trung Trang')
# Vraie question : est ce que je soustrais la valeur moyenne de la période ou bien de la série tempo totale ?
ax.plot(water_levels['Datetime'], water_levels[col2] / 100 -
        np.average(water_levels[col2] / 100), color='brown', alpha=0.5, zorder=1, lw=0.5, label='Hon Dau')
ax.legend()
outfile = 'Difference_amplitude_'
ax.set_xlim(datetime(2021, 1, 1), datetime(2023, 1, 1))
outfile = outfile + 'allyear'
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

######## conditions to plot
years = [2021, 2022]
months = np.arange(1, 13, 1)
for year in years:
    for month in months:
        condition_min = (result_df_min['Time min'].dt.year == year) & (result_df_min['Time min'].dt.month == month)
        condition_max = (result_df_max['Time max'].dt.year == year) & (result_df_min['Time min'].dt.month == month)
        condition_wat = (water_levels[D].dt.year == year) & (water_levels[D].dt.month == month)
        all_year = False
        selected_daily_wat = daily_mean[(daily_mean.index.year == year) & (daily_mean.index.month == month)]
        mean_dis = np.round(np.nanmean(selected_daily_wat['Q']), 2)

        # Plot of the 1. Amplitudes difference 2. the water level
        fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
        fig.suptitle('Difference of amplitudes HD-TT', fontsize=fontsize)
        ax = axs[0]
        ax.grid(True, alpha=0.5)
        ax.set_ylim(1, 4)
        twin.set_ylim(200, 2200)
        ax.set_ylabel('Amplitude difference (m)', fontsize=fontsize - 2)
        twin = ax.twinx()
        twin.set_ylabel('Discharge (m$^3$/s)', fontsize=fontsize - 2)
        y_min = result_df_min['Diff min'].loc[condition_min] / 100
        y_max = result_df_max['Diff max'].loc[condition_max] / 100
        mean_min = np.round(np.nanmean(y_min), 2)
        mean_max = np.round(np.nanmean(y_max), 2)
        ax.scatter(result_df_min['Time min'].loc[condition_min], y_min, marker='o', color='grey',
                   label='Amplitude difference at min ' + str(mean_min) + 'm')
        twin.scatter(selected_daily_wat.index, selected_daily_wat['Q'], marker='d', color='blue',
                     label='Daily discharge at TT' + str(mean_dis) + 'm$^3$/s')
        # ax.scatter(result_df_max['Time max'].loc[condition_max], y_max, marker='<', color='red',
        #           label='Amplitude difference at max ' + str(mean_max) + 'm')
        ax.legend()
        ax = axs[1]
        ax.grid(True, alpha=0.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
        date_form = DateFormatter("%d/%m")  # Define the date format
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(date_form)
        ax.set_xlabel('Time', fontsize=fontsize - 2)
        ax.plot(water_levels['Datetime'].loc[condition_wat], water_levels[col].loc[condition_wat] / 100 -
                np.average(water_levels[col] / 100), color='grey', alpha=0.5, zorder=3, lw=0.5, label='Trung Trang')
        # Vraie question : est ce que je soustrais la valeur moyenne de la période ou bien de la série tempo totale ?
        ax.plot(water_levels['Datetime'].loc[condition_wat], water_levels[col2].loc[condition_wat] / 100 -
                np.average(water_levels[col2] / 100), color='brown', alpha=0.5, zorder=1, lw=0.5, label='Hon Dau')
        ax.legend()
        outfile = 'Difference_amplitude_'
        if month == 12:
            ax.set_xlim(datetime(year, month, 1), datetime(year + 1, 1, 1))
        else:
            ax.set_xlim(datetime(year, month, 1), datetime(year, month + 1, 1))
        outfile = outfile + str(year) + str(month)
        outfile = outfile + '.png'
        fig.savefig(outfile, format='png')

# Plot of the 1. Amplitudes difference 2. the correlation and the lag, 3. the water level
fig, axs = plt.subplots(figsize=(18, 10), nrows=3, sharex=True)
fig.suptitle('', fontsize=fontsize)
ax = axs[0]
twin = ax.twinx()
ax.grid(True, alpha=0.5)
twin.set_ylabel('phase lag (mn)', fontsize=fontsize - 2)
ax.set_ylabel('Amplitude difference (m)', fontsize=fontsize - 2)
y_min = result_df_min['Diff min'] / 100
y_max = result_df_max['Diff max'] / 100
mean_min = (np.round(np.nanmean(y_min), 2))
mean_max = (np.round(np.nanmean(y_max), 2))
ax.scatter(result_df_min['Time min'], y_min, marker='o', color='grey', s=s, label='mean ' + str(mean_min) + 'm')
twin.scatter(data_phase_Q.index, y_second, marker='x', s=s, color="brown", label='phase lag')
# ax.scatter(result_df_max['Time max'], y_max, marker='<', color='red',
#           label='Amplitude difference at max ' + str(mean_max) + 'm')
legend1 = ax.legend(loc='upper left')
legend2 = twin.legend(loc='upper right')
# ax.legend()
l = twin.add_artist(legend2)
l.get_frame().set_alpha(0.5)
ax = axs[1]
ax.grid(True, alpha=0.5)
twin = ax.twinx()
twin.set_ylabel('correlation', fontsize=fontsize - 2)
ax.set_ylabel('Discharge (m$^3$/s)', fontsize=fontsize - 2)
# ax.set_ylim(0.82,1)
y_second = [td.total_seconds() / 60 for td in data_phase_Q['lag']]  # To have in mn : /60
ax.scatter(data_phase_Q.index, data_phase_Q['Q'], marker='d', color='blue', s=s, label='Discharge at TT')
twin.scatter(data_phase_Q.index, data_phase_Q['corr'], marker='<', s=s, color='orange', label='correlation fit HD-TT')
legend1 = ax.legend(loc='upper left')
legend2 = twin.legend(loc='upper right')
# ax.legend()
l = twin.add_artist(legend2)
l.get_frame().set_alpha(0.5)  # Set alpha to 0 for full transparency
ax = axs[2]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
date_form = DateFormatter("%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 2)
ax.plot(water_levels['Datetime'], water_levels[col] / 100 - np.average(water_levels[col] / 100), color='grey',
        alpha=0.5,
        zorder=3, lw=0.5, label='Trung Trang')
ax.plot(water_levels['Datetime'], water_levels[col2] / 100 - np.average(water_levels[col2] / 100), color='brown',
        alpha=0.5,
        zorder=1, lw=0.5, label='Hon Dau')
ax.legend()
if all_year:
    ax.set_xlim(datetime(2021, 1, 1), datetime(2023, 1, 1))
outfile = 'Difference_phaseandcorr_amplitude_allyear'
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

# 27/10 : I try to find the lag, corr, and discharge at the day scale
# ==> NOT RELEVANT, DO NOT work because time scale too short, perhaps on a week - or 14days it would be better.
treshold = 100  # Value in cm
year = [2021, 2022]
lags, corrs = [], []
list_mean_ebb_HD, list_mean_flood_HD, list_mean_ebb_TT, list_mean_flood_TT = [], [], [], []
for y in year:
    months = np.arange(1, 13, 1)
    for month in months:
        a = '0' if month < 10 else ''
        selected_data = water_levels[(water_levels['Datetime'].dt.year == y) &
                                     (water_levels['Datetime'].dt.month == month)].copy()
        selected_interp = interpolated_series[(interpolated_series.index.month == month) &
                                              (interpolated_series.index.year == y)].copy()
        selected_ebb = Ebb_df[(Ebb_df['Datetime HD'].dt.year == y) & (Ebb_df['Datetime HD'].dt.month == month)].copy()
        selected_flood = Flood_df[
            (Flood_df['Datetime HD'].dt.year == y) & (Flood_df['Datetime HD'].dt.month == month)].copy()
        selected_ebb2 = selected_ebb[selected_ebb['Amplitude HD'] < -treshold].copy()
        selected_flood2 = selected_flood[selected_flood['Amplitude HD'] > treshold].copy()
        # 26/10 : I want to select the date where the amplitudes are under the tresholds :
        date_neap = selected_ebb[D].loc[selected_ebb['Amplitude HD'] > -treshold].dt.date
        date_neap2 = selected_flood[D].loc[selected_flood['Amplitude HD'] < treshold].dt.date
        # We only take Ebb_df, Flood_df is supposed to follow the same trend ,
        # selected_flood[D].loc[selected_flood['Amplitude'] < treshold]
        mask = selected_data[D].dt.date.isin(date_neap)
        mask2 = selected_data[D].dt.date.isin(date_neap2)
        selected_data2 = selected_data[~(mask | mask2)]  # Apply the mask to filter out rows from
        selected_interp['date_only'] = selected_interp.index.date
        selected_interp2 = selected_interp[~(selected_interp['date_only'].isin(date_neap.values) |
                                             selected_interp['date_only'].isin(date_neap2.values))]

        for d in selected_interp2.index.day.unique():
            day_interp = selected_interp2[(selected_interp2.index.month == month) &
                                          (selected_interp2.index.year == y) &
                                          (selected_interp2.index.day == d)].copy()
            lag, corr = calculate_corr(day_interp[[Wat_TT]], day_interp[[Wat_HD]], year_constraint, y,
                                       month_constraint, month, datetime=False)
            lags.append(lag)
            corrs.append(corr)

        mean_ebb_HD = (selected_ebb2['Duration HD'].dt.total_seconds() / 3600).astype(float)
        mean_flood_HD = (selected_flood2['Duration HD'].dt.total_seconds() / 3600).astype(float)
        mean_ebb = (selected_ebb2['Duration'].dt.total_seconds() / 3600).astype(float)
        mean_flood = (selected_flood2['Duration'].dt.total_seconds() / 3600).astype(float)
        list_mean_ebb_HD.append(mean_ebb_HD.mean())
        list_mean_flood_HD.append(mean_flood_HD.mean())
        list_mean_ebb_TT.append(mean_ebb.mean())
        list_mean_flood_TT.append(mean_flood.mean())

# 7/11 : I add the monthly means of all values
monthly_mean['Ebb duration TT'] = np.round(list_mean_ebb_TT, 2)
monthly_mean['Ebb duration HD'] = np.round(list_mean_ebb_HD, 2)
monthly_mean['Flood duration TT'] = np.round(list_mean_flood_TT, 2)
monthly_mean['Flood duration HD'] = np.round(list_mean_flood_HD, 2)
# Plot of the monthly means vs discharge
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.suptitle('Ebb-Flood duration at TT', fontsize=fontsize)
ax = axs[0]
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
ax.scatter(monthly_mean['Q'], monthly_mean['Ebb duration TT'], marker='o', color='grey', s=20)
slope, intercept, r_value, p_value, std_err = stats.linregress(monthly_mean['Q'], monthly_mean['Ebb duration TT'])
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
x = np.arange(0, 1200, 1)
ax.plot(x, slope * x + intercept, color='grey', label=label)
ax.legend()
ax = axs[1]
ax.grid(which='both', alpha=0.5)
ax.set_xlabel('Discharge (m$^3$/s)', fontsize=fontsize - 2)
ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
ax.scatter(monthly_mean['Q'], monthly_mean['Flood duration TT'], marker='<', color='red', s=20)
slope, intercept, r_value, p_value, std_err = stats.linregress(monthly_mean['Q'], monthly_mean['Flood duration TT'])
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
ax.plot(x, slope * x + intercept, color='red', label=label)
ax.legend()
outfile = 'Ebb_Flood_duration_TT_vs_discharge'
if filter:
    outfile = outfile + '_filtered_at_' + str(treshold)
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

# 7/11 : I want to calculate the ratio ONLY if the datetimes correspond to the Flood first, and Ebb after with less
# than 24h between
# I need to first, merge on the datetimes of the whole series, then to filter on datetime conditions, hello second to
Ebb_and_flood_for_duration = Ebb_df.merge(Flood_df, on='Datetime HD', suffixes=('_Ebb', '_Flood'))
Ebb_and_flood_filter_for_duration = Ebb_df_filtered.merge(Flood_df_filtered, on='Datetime HD', suffixes=('_Ebb', '_Flood')) #12/05/24 je change Datetime à Datetime HD
Ebb_and_flood_filter_for_duration['time_diff_hours'] = \
    (Ebb_and_flood_filter_for_duration['Datetime_Ebb'] -
     Ebb_and_flood_filter_for_duration['Datetime_Flood']).dt.total_seconds() / 3600
Ebb_and_flood_filter_for_duration_24hfiltered = \
    Ebb_and_flood_filter_for_duration[Ebb_and_flood_filter_for_duration['time_diff_hours'] < 24]
Ebb_and_flood_filter_for_duration_24hfiltered['Duration_ratio'] = \
    Ebb_and_flood_filter_for_duration_24hfiltered['Duration_Ebb'] / \
    Ebb_and_flood_filter_for_duration_24hfiltered['Duration_Flood']
# Plot the ebb and flood duration
for y in [2021, 2022]:
    condition_ebb = (Ebb_df_filtered['Datetime'].dt.year == y)
    condition_flood = (Flood_df_filtered['Datetime'].dt.year == y)
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.suptitle('Ebb-Flood duration at HD', fontsize=fontsize)
    ax = axs[0]
    ax.grid(which='both', alpha=0.5)
    twin = ax.twinx()
    twin.set_ylabel('Ebb/Flood')
    twin.set_ylim(0.25, 2.5)
    ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
    # ax.set_ylim(5, 20)
    y_ebb = (Ebb_df_filtered['Duration HD'].loc[condition_ebb].dt.total_seconds() / 3600).astype(float)
    y_flood = (Flood_df_filtered['Duration HD'].loc[condition_flood].dt.total_seconds() / 3600).astype(
        float)  # ATTENTION : il est possible d'aoir 2
    # tailles différentes de df à cause du treshold, et donc d'avoir un soucis avec la compatibilité si l'on fait
    # la division simple avec le reset_index.
    # Ebb_and_flood_filtered = pd.merge_asof(Flood_df_filtered, Ebb_df_filtered, on='Datetime')
    # rap = Ebb_and_flood_filtered['Duration_y'] / Ebb_and_flood_filtered['Duration_x']
    # rap = y_ebb.reset_index()['Duration'].loc[condition_ebb] / y_flood.reset_index()['Duration'].loc[condition_flood]
    # Pour le moment je ne m'occupe pas des dates des EBb et Flood que je divise
    mean_ebb = (np.round(np.nanmean(y_ebb), 1))
    mean_flood = (np.round(np.nanmean(y_flood), 1))
    # mean_rap = (np.round(np.nanmean(rap), 2))
    ax.plot(Ebb_df_filtered['Datetime HD'].loc[condition_ebb], y_ebb, marker='o', color='grey',
            label='Ebb duration ' + str(mean_ebb) + 'h', lw=2)
    ax.plot(Flood_df_filtered['Datetime HD'].loc[condition_flood], y_flood, marker='<', color='red',
            label='Flood duration ' + str(mean_flood) + 'h', lw=2)
    # twin.plot(Ebb_and_flood['Datetime HD'].loc[Ebb_and_flood['Datetime HD'].dt.year == y] + timedelta(days=1),
    # rap, marker='^', color='k', label='Ebb/Flood ' + str(mean_rap), lw=2)
    legend1 = ax.legend(loc='upper left')
    legend2 = twin.legend(loc='upper right')
    # twin.add_artist(legend2)

    ax = axs[1]
    ax.grid(which='both', alpha=0.5)
    ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
    # ax.set_ylim(-1, 2.5)
    date_form = DateFormatter("%d/%m")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=15))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xlabel('Time', fontsize=fontsize)
    # ax.set_xlim(datetime(year, 1, 1), datetime(year, 12, 31))
    ax.plot(water_levels['Datetime'].loc[water_levels['Datetime'].dt.year == y],
            water_levels[col2].loc[water_levels['Datetime'].dt.year == y] / 100, color='grey', alpha=0.5, lw=2)
    # ax.set_xlim(datetime(2022, 5, 1), datetime(2022, 10, 1))
    ax.plot(Ebb_df_filtered['Datetime HD'].loc[condition_ebb],
            -Ebb_df_filtered['Amplitude HD'].loc[condition_ebb] / 100, marker='o', color='grey',
            label='Ebb duration ' + str(mean_ebb) + 'h', lw=2)
    ax.plot(Flood_df_filtered['Datetime HD'].loc[condition_flood],
            Flood_df_filtered['Amplitude HD'].loc[condition_flood] / 100, marker='<', color='red',
            label='Flood duration ' + str(mean_flood) + 'h', lw=2)
    outfile = 'Ebb_Flood_duration_HD'
    if filter:
        outfile = outfile + '_filtered_at_' + str(treshold)
    outfile = outfile + '.png'
    fig.savefig(outfile, format='png')

# Plot of the duration of flood and Ebb at TT depending on the tidal range at HD ALL DATA:
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# fig.suptitle('Ebb-Flood duration at TT vs tidal range', fontsize=fontsize)
ax = axs[0]
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
y_ebb = (Ebb_df['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(abs(Ebb_df['Amplitude HD'] / 100), y_ebb, marker='o', color='grey', s=20)
slope, intercept, r_value, p_value1, std_err = stats.linregress(abs(Ebb_df['Amplitude HD'] / 100), y_ebb)
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
x = np.arange(0, 5, 0.1)
ax.plot(x, slope * x + intercept, color='grey', label=label)
ax.legend()
ax = axs[1]
ax.grid(which='both', alpha=0.5)
ax.set_xlabel('Tidal range at HD (m)', fontsize=fontsize - 2)
ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
y_flood = (Flood_df['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(Flood_df['Amplitude HD'] / 100, y_flood, marker='<', color='red', s=20)
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df['Amplitude HD'].dropna() / 100,
                                                               y_flood.dropna())
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
ax.plot(x, slope * x + intercept, color='red', label=label)
ax.legend()
outfile = 'Ebb_Flood_duration_TT_vs_tidal_range'
filter = False
if filter:
    outfile = outfile + '_filtered_at_' + str(treshold)
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

# Test of duration vs amplitude HD with quartile on discharge
quantile = False
if quantile:
    deb = Ebb_df.quantile([0.25, 0.5, 0.75])['Q'].values
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
Q1 = Ebb_df.loc[condA]
Q2 = Ebb_df.loc[condB]
Q3 = Ebb_df.loc[condC]
Q4 = Ebb_df.loc[condD]
debit = []
for d in deb:
    debit.append(str(np.round(d, 0)))
debit.append('>' + str(np.round(deb[2], 0)))

fig, ax = plt.subplots(figsize=(18, 8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# fig.suptitle('Ebb-Flood duration at TT', fontsize=fontsize)
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range HD (m)', fontsize=fontsize - 2)
x = np.arange(0.1, 4, 0.1)
c = 0
for Q in [Q1, Q2, Q3, Q4]:
    QX = abs(Q['Amplitude HD'].dropna()) / 100
    QY = (Q['Duration'].dropna().dt.total_seconds() / 3600).astype(float)
    ax.scatter(QX, QY, color=cmap(list_color[c]), alpha=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(QX, QY)
    label = str(debit[c] + " m$³$/s, " + str(np.round(slope, 2)) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(
        np.round(r_value, 4)) + ' N=' + str(QX.count()))
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=3, color=cmap(list_color[c]), label=label)
    legend = ax.legend()
    c = c + 1
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(15)  # Set the desired font size
outfile = 'Ebb_duration_TT_vs_tidalrange_alldata_quartile'
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

# Plot of the duration of flood and Ebb at TT depending on the DISCHARGE at HD ALL DATA:
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# fig.suptitle('Ebb-Flood duration at TT vs tidal range', fontsize=fontsize)
ax = axs[0]
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
y_ebb = (Ebb_df_filtered['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(abs(Ebb_df_filtered['Q'].dropna()), y_ebb.dropna(), marker='o', color='grey', s=20)
slope, intercept, r_value, p_value1, std_err = stats.linregress(abs(Ebb_df_filtered['Q'].dropna()), y_ebb.dropna())
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
x = np.arange(50, 2200, 1)
ax.plot(x, slope * x + intercept, color='grey', label=label)
ax.legend()
ax = axs[1]
ax.grid(which='both', alpha=0.5)
ax.set_xlabel('Discharge (m$³$/s)', fontsize=fontsize - 2)
ax.set_ylabel('Duration (hours)', fontsize=fontsize - 2)
y_flood = (Flood_df_filtered['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(Flood_df_filtered['Q'][0:-1], y_flood.dropna(), marker='<', color='red', s=20)
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df_filtered['Q'][0:-1], y_flood.dropna())
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
ax.plot(x, slope * x + intercept, color='red', label=label)
ax.legend()
outfile = 'Ebb_Flood_duration_TT_vs_discharge_alldata'
filter = True
if filter:
    outfile = outfile + '_filtered_at_' + str(treshold)
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

# Water level at HD and TT
save = True
year_constraint = True
year = 2021
month_constraint = True
months = np.arange(1, 13, 1)
title = 'Water levels '
Wat_HD = merged_df.columns[2]
D = merged_df.columns[1]
Wat_TT = merged_df.columns[0]
if year_constraint and not month_constraint:
    selected_data = merged_df[(merged_df['Datetime'].dt.year == year)]
    title = title + str(year)
elif not year_constraint and not month_constraint:
    print('no constraint on year or month, I take the whole series')
    selected_data = merged_df
    title = title + '2021-2022'
else:
    lags, corrs, corrs2 = [], [], []
    for month in months:
        a = '0' if month < 10 else ''
        title = 'Water levels '
        if year_constraint and month_constraint:
            selected_data = merged_df[(merged_df['Datetime'].dt.year == year) &
                                      (merged_df['Datetime'].dt.month == month)]
            title = title + a + str(month) + '/' + str(year)
            lag, corr = calculate_corr(selected_data[[Wat_TT, D]], selected_data[[Wat_HD, D]], year_constraint, year,
                                       month_constraint, month)
            lags.append(lag)
            corrs.append(corr)
        elif month_constraint and not year_constraint:
            selected_data = merged_df[merged_df['Datetime'].dt.month == month]
            title = title + a + str(month)
            lag, corr = calculate_corr(selected_data[[Wat_TT, D]], selected_data[[Wat_HD, D]], year_constraint, year,
                                       month_constraint, month)
            lags.append(lag)
            corrs.append(corr)
    for c in corrs:
        corrs2.append(np.round(c, 4))

fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle(title)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize)
# ax.set_ylim(-50, 450)
if year_constraint and not month_constraint:
    date_form = DateFormatter("%m/%Y")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
elif not year_constraint and month_constraint:
    date_form = DateFormatter("%d/%m")  # Define the date format
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize)

ax.plot(selected_data[D], selected_data[Wat_HD] / 100 - np.average(selected_data[Wat_HD] / 100), color='black',
        label=Wat_HD)
lag, corr = calculate_corr(selected_data[[Wat_TT, D]], selected_data[[Wat_HD, D]], year_constraint, year,
                           month_constraint, month)
label = Wat_TT + '\n lag = ' + str(lag) + 'h, r=' + str(np.round(corr, 4))
ax.plot(selected_data[D], selected_data[Wat_TT] / 100 - np.average(selected_data[Wat_TT] / 100), color='grey',
        label=label)
plt.legend(fontsize=fontsize)
if save:
    outfile = 'Waterlevel_HD_TT_'
    if year_constraint and not month_constraint:
        ax.set_xlim(datetime(year, 1, 1), datetime(year + 1, 1, 2))
        outfile = outfile + 'allyear' + str(year)
    elif not year_constraint and not month_constraint:
        ax.set_xlim(datetime(2021, 1, 1), datetime(20223, 1, 2))
        outfile = outfile + 'bothyears2021-2022'
    elif not year_constraint and month_constraint:
        outfile = outfile + '2years_month_' + str(month)
    elif year_constraint and month_constraint:
        outfile = outfile + a + str(month) + str(year)
        if month == 12:
            ax.set_xlim(datetime(year, month, 1), datetime(year + 1, 1, 2))
        else:
            ax.set_xlim(datetime(year, month, 1), datetime(year, month + 1, 2))

    outfile = outfile + '.png'
    fig.savefig(outfile, format='png')

color_500 = 0.05
color_1000 = 0.3
color_1500 = 0.65
color_2000 = 0.85
list_color = [color_500, color_1000, color_1500, color_2000]

ratio = True  # set to true if we do not want the mean values of TT and HD but the ratio of TT/HD
fig, ax = plt.subplots(figsize=(18, 10), nrows=1, sharex=True)
# fig.suptitle('Difference of amplitude HD-TT', fontsize=fontsize)
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Tidal range at HD (m)', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range at TT (m)', fontsize=fontsize - 2)
ax.set_xlim(-3, 3)
ax.set_ylim(-4, 4)
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))  # Set minor tick every 0.2 units on the x-axis
p1 = ax.scatter(Ebb['Amplitude'] / 100, Ebb_HD['Amplitude'] / 100, marker='o', c=Ebb['Q'], cmap=cmap,
                vmin=200, vmax=2000, label='Ebb amplitude', s=s)
ax.scatter(Flood['Amplitude'] / 100, Flood_HD['Amplitude'] / 100, marker='<', c=Flood['Q'], cmap=cmap,
           vmin=200, vmax=2000, label='Flood amplitude', s=s)
ax.plot(np.arange(-5, 5, 1), np.arange(-5, 5, 1), alpha=0.5, color='grey')
cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
cbar.set_label(label='Discharge (m$^3$/s)', fontsize=fontsize - 1)
cbar.outline.set_linewidth(0.05)
ax.legend()
# Legend of the Ebb part
y_coord = -0.5
for i in range(4):  # the number are in the order discharge 500, 1m,2,3,4, discharge 1000, 1m,2,3,4 ...
    x_coord = 0
    c = 0
    for dx in [0, 4, 8, 12]:
        if ratio:
            rap = np.round(list_TT_Ebb[i + dx] / list_HD_Ebb[i + dx], 2)
            label = str(rap) + '\nN=' + str(list_count_Ebb[i + dx])
        else:
            label = str(list_HD_Ebb[i + dx]) + '\n' + str(list_TT_Ebb[i + dx]) + '\nN=' + str(list_count_Ebb[i + dx])
        ax.text(x_coord, y_coord, label, color=cmap(list_color[c]), fontsize=fontsize - 4)
        x_coord = x_coord + 0.5
        c = c + 1
    y_coord = y_coord - 1
# Legend of the Flood part
y_coord = 0.5
for i in range(4):  # the number are in the order discharge 500, 1m,2,3,4, discharge 1000, 1m,2,3,4 ...
    x_coord = -2
    c = 0
    for dx in [0, 4, 8, 12]:
        if ratio:
            rap = np.round(list_TT_Flood[i + dx] / list_HD_Flood[i + dx], 2)
            label = str(rap) + '\nN=' + str(list_count_Flood[i + dx])
        else:
            label = str(list_HD_Flood[i + dx]) + '\n' + str(list_TT_Flood[i + dx]) + '\nN=' + str(
                list_count_Flood[i + dx])
        ax.text(x_coord, y_coord, label, color=cmap(list_color[c]), fontsize=fontsize - 4)
        x_coord = x_coord + 0.5
        c = c + 1
    y_coord = y_coord + 1
outfile = 'Amplitudes_TT-HD_Q_'
if ratio:
    outfile = outfile + 'ratiovalues'
else:
    outfile = outfile + 'withmeanvalues'
outfile = outfile + '.png'
fig.savefig(outfile)

# 6/11 Figure of Ebb and flood mixed with HD in the xaxis
fig, ax = plt.subplots(figsize=(18, 10), nrows=1, sharex=True)
# fig.suptitle('Difference of amplitude HD-TT', fontsize=fontsize)
ax.grid(which='both', alpha=0.5)
ax.set_xlabel('Tidal range at HD (m)', fontsize=fontsize - 2)
ax.set_ylabel('Tidal range at TT (m)', fontsize=fontsize - 2)
ax.set_ylim(0, 3)
ax.set_xlim(0, 4)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(1))  # Set minor tick every 0.2 units on the x-axis
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
p1 = ax.scatter(Ebb_and_flood['Amplitude HD'] / 100, Ebb_and_flood['Amplitude'] / 100, marker='o', c=Ebb_and_flood['Q'],
                cmap=cmap,
                vmin=200, vmax=2000, s=s)
ax.plot(np.arange(-5, 5, 1), np.arange(-5, 5, 1), alpha=0.5, color='grey')
cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
cbar.set_label(label='Discharge (m$^3$/s)', fontsize=fontsize - 1)
cbar.outline.set_linewidth(0.05)
# ax.legend()
outfile = 'Amplitudes_TT-HD_Q'
outfile = outfile + '.png'
fig.savefig(outfile)

fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
fig.suptitle('Ebb-Flood Amplitudes at TT', fontsize=fontsize)
ax = axs[0]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Amplitude (m)', fontsize=fontsize - 2)
ax.scatter(Ebb['Datetime'], abs(Ebb['Amplitude'] / 100), marker='o', color='grey', label='Ebb amplitude')
ax.scatter(Flood['Datetime'], Flood['Amplitude'] / 100, marker='<', color='red', label='Flood amplitude')
ax.legend()
ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
date_form = DateFormatter("%m/%y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Time', fontsize=fontsize - 2)
ax.plot(water_levels['Datetime'], water_levels[col] / 100, color='grey', alpha=0.5, lw=0.5)
ax.set_xlim(datetime(2022, 5, 1), datetime(2022, 10, 1))

# 3/11
# plot of the Tidal range at TT vs Tidal range at HD with 4 classes of discharge.
# TODO : checker pourquoi il y a des biais, faire la même courbe sans les moyennes
lab_dis = ['500 m$³$/s', '1000 m$³$/s', '1500 m$³$/s', '>1500 m$³$/s']
fig, ax = plt.subplots(figsize=(18, 10))
# fig.suptitle('Amplitude of TT vs tidal range', fontsize=fontsize)
ax.grid(which='both', alpha=0.5)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.set_ylabel('Tidal range at TT (m)', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range at HD (m)', fontsize=fontsize - 2)
c = 0
for i in [0, 4, 8, 12]:
    ax.scatter(tableau_discharge_category_both['Amplitude HD'].loc[i:i + 4],
               tableau_discharge_category_both['Amplitude TT'].loc[i:i + 4],
               color=cmap(list_color[c]), alpha=0.5)
    slope, intercept, r_value, p_value, std_err = \
        stats.linregress(tableau_discharge_category_both['Amplitude HD'].loc[i:i + 4],
                         tableau_discharge_category_both['Amplitude TT'].loc[i:i + 4])
    label = lab_dis[c] + " {:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(
        np.round(r_value, 4))
    # ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
    x = np.arange(0, 5, 1)
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    legend = ax.legend()
    for j in range(i, i + 4):
        if j == 4 or j == 12:
            ax.text(tableau_discharge_category_both['Amplitude HD'].loc[j] + 0.09,
                    tableau_discharge_category_both['Amplitude TT'].loc[j] - 0.05,
                    'N=' + str(tableau_discharge_category_both['N'].loc[j]), color=cmap(list_color[c]),
                    fontsize=fontsize - 10)
        else:
            ax.text(tableau_discharge_category_both['Amplitude HD'].loc[j] + 0.09,
                    tableau_discharge_category_both['Amplitude TT'].loc[j] + 0.05,
                    'N=' + str(tableau_discharge_category_both['N'].loc[j]), color=cmap(list_color[c]),
                    fontsize=fontsize - 10)
    c = c + 1
ax.set_xlim(0, 4)
ax.set_ylim(0, 2.5)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(15)  # Set the desired font size
fig.savefig('Amplitude_TT_vs_HD_Amplitude.png', format='png')

########### 4 curves depending on the class of discharge
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Amplitude of TT vs tidal range', fontsize=fontsize)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Tidal range at TT', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range at HD', fontsize=fontsize - 2)
c = 0
for i in [0, 4, 8, 12]:
    X = tableau_discharge_category_both['Amplitude HD'].loc[i:i + 3]
    Y = tableau_discharge_category_both['Amplitude TT'].loc[i:i + 3]
    print('X=', X)
    ax.scatter(X, Y, color=cmap(list_color[c]), alpha=0.5)
    ## METHOD 1 :
    # model = LinearRegression(fit_intercept=False)
    # sample_weight = tableau_discharge_category_both['N'].loc[i:i + 3].values
    # model.fit(X.values.reshape(-1, 1), Y.values)
    # r_squared = model.score(X.values.reshape(-1,1),Y.values)
    # r_value = np.sqrt(r_squared)
    ## METHOD 2 :
    """
    slope = np.dot(X, Y) / np.dot(X, X)
    Y_pred = slope * X
    # Calculate the R-squared value
    ssr = np.sum((Y - Y_pred) ** 2)
    sst = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ssr / sst)
    # The 'slope' variable contains the coefficient (slope) of the linear fit
    # The 'r_squared' variable contains the R-squared value
    # print("Coefficient (Slope):", slope)
    # print("R-squared (r_value):", r_squared)
    """

    label = lab_dis[c] + " {:.1e}".format(slope) + ' x, r=' + str(np.round(r_value, 4))
    x = np.arange(0, 5, 1)
    ax.plot(x, slope * x, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    legend = ax.legend()
    for j in range(i, i + 3):
        ax.text(tableau_discharge_category_both['Amplitude HD'].loc[j] + 0.09,
                tableau_discharge_category_both['Amplitude TT'].loc[j] + 0.05,
                'N=' + str(tableau_discharge_category_both['N'].loc[j]), color=cmap(list_color[c]),
                fontsize=fontsize - 4)
    c = c + 1
ax.set_xlim(0, 4)
ax.set_ylim(0, 2.5)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(15)  # Set the desired font size
fig.savefig('Amplitude_TT_vs_HD_Amplitude_with0_vsklearn.png', format='png')

########### 4 curves depending on the class of discharge BUT WITH QUARTILE METHOD :
# 6/11 : Ce que j'aimerais faire, c'est faire un plot non pas avec que les moyennes, mais plutot avec toutes les
# 4 courbes avec les valeurs de quantiles sur l'ensmeble des données
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

###############################################
###############################################
interp = 'polyfit'
fig, ax = plt.subplots(figsize=(16, 10))
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Tidal amplitude at TT (m)', fontsize=fontsize - 2)
ax.set_xlabel('Tidal amplitude at HD (m)', fontsize=fontsize - 2)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
filter_condition = Ebb_and_flood['Amplitude HD'] > treshold
p1 = ax.scatter(abs(Ebb_and_flood['Amplitude HD'].loc[filter_condition] / 100),
           abs(Ebb_and_flood['Amplitude'].loc[filter_condition] / 100),
           c=Ebb_and_flood['Q'].loc[filter_condition], cmap=cmap, alpha=0.5)
cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize - 1)
cbar.outline.set_linewidth(0.05)
c = 0
for Q in [Q1, Q2, Q3, Q4]:
    QX = Q['Amplitude HD'].dropna() / 100
    QY = Q['Amplitude'].dropna() / 100
    #ax.scatter(QX, QY, color=cmap(list_color[c]), alpha=0.5)
    x = np.arange(0, 5, 1)
    if interp == 'linear':
        slope, intercept, r_value, p_value, std_err = stats.linregress(QX, QY)
        label = str(
            debit[c] + " m$³$/s, " + str(np.round(slope, 2)) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(
                np.round(r_value, 4)))
        ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    elif interp == 'log':
        params, covariance = curve_fit(logarithmic_function, QX, QY)
        a, b = params
        x_fit = np.linspace(0.1, 4, 100)  # np.linspace(min(QX), max(QY), 100)  # Generate x values for the curve
        y_fit = logarithmic_function(x_fit, a, b)  # Calculate y values for the curve
        y_predicted = logarithmic_function(QX, a, b)
        r_squared = r2_score(QY, y_predicted)
        r_value = np.sqrt(r_squared)  # ATTENTION : ne donne pas le signe de la correlation
        label = str(debit[c] + " m$³$/s, " + str(np.round(a, 2)) + ' log(x) + ' + str(np.round(b, 2)) + ', r=' + str(
            np.round(r_value, 4)))
        ax.plot(x_fit, y_fit, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    elif interp == 'polyfit':
        coefficients = np.polyfit(QX, QY, 2)
        y_pred = np.polyval(coefficients, QX.sort_values())
        r_value, _ = pearsonr(y_pred, QY.sort_values())
        label1 = str(debit[c] + " m$³$/s, " + str(np.round(coefficients[0], 2)) + ' x${²}$ + ' +
                     str(np.round(coefficients[1], 2)) + ' x + ' + str(np.round(coefficients[2], 2)) + ' , r=' +
                     "{:.3f}".format(r_value) + ' N=' + str(QX.count())) #str(np.round(r_value, 3))
        label = str(label_title[c] + ' m$³$/s, r=' + "{:.3f}".format(r_value) + ' N=' + str(QX.count()))
        ax.plot(QX.sort_values(), y_pred, alpha=0.5, lw=2, color=cmap(list_color[c]), label=label)
    legend = ax.legend()
    c = c + 1
ax.plot(np.arange(0,5,1), np.arange(0,5,1), c='gray')
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(19)  # Set the desired font size
outfile = 'Amplitude_TT_vs_HD_Amplitude_'
if quantile:
    outfile = outfile + 'withquartile_v2'
else:
    outfile = outfile + 'withdischargecategories'
outfile = outfile + '_' + interp + '_allvalues.png'
fig.savefig(outfile, format='png')


################## FIGURE FOR ARTICLE
interp = 'polyfit'
treshold=0
fig, axs = plt.subplots(nrows=3, figsize=(30, 10))
ax=axs[0]
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Tidal range at TT (m)', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range at HD (m)', fontsize=fontsize - 2)
ax.set_xlim(0, 4)
ax.set_ylim(0, 3)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
filter_condition = Ebb_and_flood['Amplitude HD'] > treshold
p1 = ax.scatter(abs(Ebb_and_flood['Amplitude HD'].loc[filter_condition] / 100),
           abs(Ebb_and_flood['Amplitude'].loc[filter_condition] / 100),
           c=Ebb_and_flood['Q'].loc[filter_condition], cmap=cmap, alpha=0.5)
cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
cbar.set_label(label='Discharge (m$^{3}$/s)', fontsize=fontsize - 1)
cbar.outline.set_linewidth(0.05)
c = 0
for Q in [Q1, Q2, Q3, Q4]:
    QX = Q['Amplitude HD'].dropna() / 100
    QY = Q['Amplitude'].dropna() / 100
    #ax.scatter(QX, QY, color=cmap(list_color[c]), alpha=0.5)
    x = np.arange(0, 5, 1)
    if interp == 'linear':
        slope, intercept, r_value, p_value, std_err = stats.linregress(QX, QY)
        label = str(
            debit[c] + " m$³$/s, " + str(np.round(slope, 2)) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(
                np.round(r_value, 4)))
        ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    elif interp == 'log':
        params, covariance = curve_fit(logarithmic_function, QX, QY)
        a, b = params
        x_fit = np.linspace(0.1, 4, 100)  # np.linspace(min(QX), max(QY), 100)  # Generate x values for the curve
        y_fit = logarithmic_function(x_fit, a, b)  # Calculate y values for the curve
        y_predicted = logarithmic_function(QX, a, b)
        r_squared = r2_score(QY, y_predicted)
        r_value = np.sqrt(r_squared)  # ATTENTION : ne donne pas le signe de la correlation
        label = str(debit[c] + " m$³$/s, " + str(np.round(a, 2)) + ' log(x) + ' + str(np.round(b, 2)) + ', r=' + str(
            np.round(r_value, 4)))
        ax.plot(x_fit, y_fit, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    elif interp == 'polyfit':
        coefficients = np.polyfit(QX, QY, 2)
        y_pred = np.polyval(coefficients, QX.sort_values())
        r_value, _ = pearsonr(y_pred, QY.sort_values())
        label1 = str(debit[c] + " m$³$/s, " + str(np.round(coefficients[0], 2)) + ' x${²}$ + ' +
                     str(np.round(coefficients[1], 2)) + ' x + ' + str(np.round(coefficients[2], 2)) + ' , r=' +
                     "{:.3f}".format(r_value) + ' N=' + str(QX.count())) #str(np.round(r_value, 3))
        label = str(label_title[c] + ' m$³$/s, r=' + "{:.3f}".format(r_value) + ' N=' + str(QX.count()))
        ax.plot(QX.sort_values(), y_pred, alpha=0.5, lw=2, color=cmap(list_color[c]), label=label)
    legend = ax.legend()
    c = c + 1
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(15)  # Set the desired font size

#fig, axs = plt.subplots(figsize=(12, 17), nrows=2, sharey=True)
ax = axs[2]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Amplification', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range HD (m)', fontsize=fontsize - 2)
slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_and_flood['Amplitude HD'].dropna()/100,
        Ebb_and_flood.dropna()['Amplitude'] / Ebb_and_flood.dropna()['Amplitude HD'])
x=np.arange(0,4,0.1)
label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ',  r=' + str(np.round(r_value, 2))
ax.plot(x,slope*x+intercept, color='k', zorder=10, label = label)
ax.scatter(Ebb_and_flood['Amplitude HD']/100,
        Ebb_and_flood['Amplitude'] / Ebb_and_flood[
            'Amplitude HD'], color='grey', lw=1)
ax.legend()
ax = axs[1]
ax.grid(True, alpha=0.5)
ax.set_ylabel('Amplification', fontsize=fontsize - 2)
ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
ax.scatter(Ebb_and_flood['Q'], Ebb_and_flood['Amplitude'] / Ebb_and_flood['Amplitude HD'] , color='grey', lw=1)
slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_and_flood.dropna()['Q'], Ebb_and_flood.dropna()['Amplitude'] / Ebb_and_flood.dropna()['Amplitude HD'])
x=np.arange(0,2300,10)
label = "{:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ',  r=' + str(np.round(r_value, 2))
ax.plot(x,slope*x+intercept, color='k', zorder=10, label = label)
ax.legend()
fig.align_labels()
fig.tight_layout()
fig.savefig('3in1_figure_TTvsHD_and_amplification.png', format='png')

"""
for i in range(4):
    X = tableau_quartile_both['Amplitude HD'].loc[i::4]
    Y = tableau_quartile_both['Amplitude TT'].loc[i::4]
    print('X=', X)
    ax.scatter(X,Y, color=cmap(list_color[c]), alpha=0.5)
    ## METHOD 1 :
    # model = LinearRegression(fit_intercept=False)
    # sample_weight = tableau_discharge_category_both['N'].loc[i:i + 3].values
    # model.fit(X.values.reshape(-1, 1), Y.values, sample_weight= sample_weight)
    # r_squared = model.score(X.values.reshape(-1,1),Y.values)
    # r_value = np.sqrt(r_squared)
    ## METHOD 2 :
    # slope = np.dot(X, Y) / np.dot(X, X)
    # Y_pred = slope * X
    # Calculate the R-squared value
    # ssr = np.sum((Y - Y_pred) ** 2)
    # sst = np.sum((Y - np.mean(Y)) ** 2)
    # r_squared = 1 - (ssr / sst)
    # The 'slope' variable contains the coefficient (slope) of the linear fit
    # The 'r_squared' variable contains the R-squared value
    # print("Coefficient (Slope):", slope)
    # print("R-squared (r_value):", r_squared)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    label = str(tableau_quartile_both['Q'].loc[i])+" {:.1e}".format(slope) + ' x, r=' + str(np.round(r_value, 4))
    x = np.arange(0, 5, 1)
    ax.plot(x, slope*x, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    legend = ax.legend()
    for j in [i,i+4,i+8,i+12]:
        ax.text(tableau_quartile_both['Amplitude HD'].loc[j]+0.09,
                tableau_quartile_both['Amplitude TT'].loc[j]+0.05,
                'N='+str(tableau_quartile_both['N'].loc[j]), color = cmap(list_color[c]), fontsize=fontsize-4)
    c = c+1
ax.set_xlim(0, 4)
ax.set_ylim(0,2.5)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(15)  # Set the desired font size
fig.savefig('Amplitude_TT_vs_HD_Amplitude_withquartile.png', format='png')
"""

################### NOT OK AS METHOD  ######################
"""
# 6/11 : same as previous figure, with 4 classes differences but I add a 0, 0 to avoid any bias
          
## METHOD 2
new_list_HD_both = list_HD_both.copy()
new_list_TT_both = list_TT_both.copy()
new_list_N_both = list_count_both.copy()
# Positions to insert 0 values (1st and 6th positions)
positions_to_insert = [0, 5, 10,15]
# Add 0 values at the specified positions
for pos in positions_to_insert:
    new_list_HD_both = new_list_HD_both[:pos] + [0] + new_list_HD_both[pos:]
    new_list_TT_both = new_list_TT_both[:pos] + [0] + new_list_TT_both[pos:]
    new_list_N_both = new_list_N_both[:pos] + [0] + new_list_N_both[pos:]

lab_dis = ['500 m$³$/s', '1000 m$³$/s', '1500 m$³$/s', '>1500 m$³$/s']
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Amplitude of TT vs tidal range', fontsize=fontsize)
ax.grid(True, alpha=0.5)
ax.set_ylabel('Tidal range at TT', fontsize=fontsize - 2)
ax.set_xlabel('Tidal range at HD', fontsize=fontsize - 2)
c = 0
for i in [0,5,10,15]:
    ax.scatter(new_list_HD_both[i:i+5], new_list_TT_both[i:i+5],
               color=cmap(list_color[c]), alpha=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_list_HD_both[i:i+5], new_list_TT_both[i:i+5])
    label = lab_dis[c]+" {:.1e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ' r=' + str(np.round(r_value, 4))
    # ax.plot(np.arange(0.83,1,0.01), slope*np.arange(0.83,1,0.01)+intercept, alpha=0.5, lw = 1, color='grey', label=label)
    x = np.arange(0, 5, 1)
    ax.plot(x, slope * x + intercept, alpha=0.5, lw=1, color=cmap(list_color[c]), label=label)
    legend = ax.legend()
    for j in range(i,i+5):
        ax.text(new_list_HD_both[j]+0.09,
                new_list_TT_both[j]+0.05,
                'N='+str(new_list_N_both[j]), color = cmap(list_color[c]), fontsize=fontsize-4)
    c = c+1
ax.set_xlim(0, 4)
ax.set_ylim(0,2.5)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(15)  # Set the desired font size
fig.savefig('Amplitude_TT_vs_HD_Amplitude_with0.png', format='png')

"""

# 25/01 : Corrélations des lags avec le tidal range et le débit.
factor = 2
Flood_copy= Flood_df.copy()
Ebb_copy= Ebb_df.copy()
print('ATTENTION AU NOM DES COLONNES §§§§')
col_ebb = Ebb_df.columns[-2]
col_flood = Flood_df.columns[-2]
Flood_copy[col_flood + ' filter'] = handle_outliers(Flood_copy, col_flood, factor)
Ebb_copy[col_ebb + ' filter'] = handle_outliers(Ebb_copy, col_ebb, factor)
for col in ['Diff water level', 'Diff' ] :
    Flood_copy[col + ' filter'] = handle_outliers(Flood_copy, col, factor)
    Ebb_copy[col + ' filter'] = handle_outliers(Ebb_copy, col, factor)

# 12/05/24 : Je refais pour vraiment prendre en compte une filtration :
filtration = False
factor = 2
print('Filtration = ' + str(filtration))
for col in ['Diff water level', 'Diff' ] :
    if filtration :
        col = col + ' filter'
        #y_ebb = ((handle_outliers(Ebb_df, col, factor)).dt.total_seconds() / 3600).astype(float)
        #y_flood = ((handle_outliers(Flood_df, col, factor)).dt.total_seconds() / 3600).astype(float)
    print(col)
    y_ebb = (Ebb_copy[col].dt.total_seconds() / 3600).astype(float)
    y_flood = (Flood_copy[col].dt.total_seconds() / 3600).astype(float)
    list_nan_ebb = y_ebb.index[y_ebb.isna()].tolist()
    list_nan_flood = y_ebb.index[y_ebb.isna()].tolist()
    list_nan_flood.append(Flood_copy.index[Flood_copy['Amplitude HD'].isna()][0])
    #else :
    #list_nan_ebb = y_ebb.index[y_ebb.isna()].tolist()
    #list_nan_flood = y_flood.index[y_flood.isna()].tolist()
    # list_nan_ebb.append(Ebb_df.index[Ebb_df['Amplitude HD'].isna()].tolist())
    #list_nan_flood.append(Flood_df.index[Flood_df['Amplitude HD'].isna()][0])
    print(len(list_nan_flood), len(list_nan_ebb))
    # list_nan_ebb = [item for sublist in list_nan_ebb for item in sublist]
    # list_nan_flood = [item for sublist in list_nan_flood for item in sublist]
    Ebb_nan_filter = Ebb_copy.drop(list_nan_ebb)
    Flood_nan_filter = Flood_copy.drop(list_nan_flood)

    print("Correlation with Q ")
    print('Ebb')
    slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_nan_filter['Q'], y_ebb.drop(list_nan_ebb))
    print('r = ' + str(np.round(r_value, 3))+' p_value = '+str(p_value))
    print('Flood')
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Q'], y_flood.drop(list_nan_flood))
    print('r = ' + str(np.round(r_value, 3))+' p_value = '+str(p_value))
    print("Correlation with Tidal range ")
    print('Ebb')
    slope, intercept, r_value, p_value, std_err = stats.linregress(-Ebb_nan_filter['Amplitude HD'], y_ebb.drop(list_nan_ebb))
    print('r = ' + str(np.round(r_value, 3))+' p_value = '+str(p_value))
    print('Flood')
    slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Amplitude HD'], y_flood.drop(list_nan_flood))
    print('r = ' + str(np.round(r_value, 3))+' p_value = '+str(p_value))

# Specific case for the slack water because of the column name :
print('ATTENTION AU NOM DES COLONNES !!!!§')
col_ebb = Ebb_df.columns[-2]
col_flood = Flood_df.columns[-2]
print('filtration = '+str(filtration))
if filtration:
    print('indeed, I do the filtration')
    y_ebb = ((handle_outliers(Ebb_df, col_ebb, factor)).dt.total_seconds() / 3600).astype(float)
    y_flood = ((handle_outliers(Flood_df, col_flood, factor)).dt.total_seconds() / 3600).astype(float)
else:
    print('no filtration')
    y_ebb = (Ebb_df[col_ebb].dt.total_seconds() / 3600).astype(float)
    y_flood = (Flood_df[col_flood].dt.total_seconds() / 3600).astype(float)
list_nan_ebb = y_ebb.index[y_ebb.isna()].tolist()
list_nan_flood = y_flood.index[y_flood.isna()].tolist()
#list_nan_ebb.append(Ebb_df.index[Ebb_df['Amplitude HD'].isna()].tolist())
list_nan_flood.append(Flood_df.index[Flood_df['Amplitude HD'].isna()][0])
#list_nan_ebb = [item for sublist in list_nan_ebb for item in sublist]
#list_nan_flood = [item for sublist in list_nan_flood for item in sublist]
print(len(list_nan_flood), len(list_nan_ebb))
Ebb_nan_filter = Ebb_copy.drop(list_nan_ebb)
Flood_nan_filter = Flood_copy.drop(list_nan_flood)

print("Correlation with Q ")
print('Ebb')
slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_nan_filter['Q'], y_ebb.drop(list_nan_ebb))
print('r = ' + str(np.round(r_value, 3)) + ' p_value = ' + str(p_value))
print('Flood')
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Q'], y_flood.drop(list_nan_flood))
print('r = ' + str(np.round(r_value, 3)) + ' p_value = ' + str(p_value))
print("Correlation with Tidal range ")
print('Ebb')
slope, intercept, r_value, p_value, std_err = stats.linregress(-Ebb_nan_filter['Amplitude HD'],
                                                               y_ebb.drop(list_nan_ebb))
print('r = ' + str(np.round(r_value, 3)) + ' p_value = ' + str(p_value))
print('Flood')
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Amplitude HD'],
                                                               y_flood.drop(list_nan_flood))
print('r = ' + str(np.round(r_value, 3)) + ' p_value = ' + str(p_value))


filtration = True
factor = 2
Flood_copy= Flood_df.copy()
Ebb_copy= Ebb_df.copy()
print('ATTENTION AU NOM DES COLONNES !!!!§')
col_ebb = Ebb_df.columns[-2]
col_flood = Flood_df.columns[-2]
Flood_copy[col_flood + ' filter'] = handle_outliers(Flood_copy, col_flood, factor)
Ebb_copy[col_ebb + ' filter'] = handle_outliers(Ebb_copy, col_ebb, factor)
for col in ['Diff water level', 'Diff' ] :
    Flood_copy[col + ' filter'] = handle_outliers(Flood_copy, col, factor)
    Ebb_copy[col + ' filter'] = handle_outliers(Ebb_copy, col, factor)

for df in Flood_copy, Ebb_copy :
    cond1 = (abs(df['Amplitude HD']) < 100)
    cond2 = (100 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 200)
    cond3 = (200 < abs(df['Amplitude HD'])) & (abs(df['Amplitude HD']) <= 300)
    cond4 = (abs(df['Amplitude HD']) > 300)
    Q1 = df.loc[cond1]
    Q2 = df.loc[cond2]
    Q3 = df.loc[cond3]
    Q4 = df.loc[cond4]
    col = 'Diff water level filter'
    for Q in [Q1, Q2, Q3, Q4]:
        print(Q[col].min(), Q[col].max(),
              Q[col].mean(),Q[col].std())
# Catégories avec les débits
for df in Flood_copy, Ebb_copy :
    cond1 = (abs(df['Q']) < 500)
    cond2 = (500 < abs(df['Q'])) & (abs(df['Q']) <= 1000)
    cond3 = (200 < abs(df['Q'])) & (abs(df['Q']) <= 1500)
    cond4 = (abs(df['Q']) > 1500)
    Q1 = df.loc[cond1]
    Q2 = df.loc[cond2]
    Q3 = df.loc[cond3]
    Q4 = df.loc[cond4]
    col = 'Diff'
    for Q in [Q1, Q2, Q3, Q4]:
        print(Q[col].min(), Q[col].max(),
              Q[col].mean(),Q[col].std())



# 26/03/24
# Calcul de toutes les r et p value,
Ebb_and_flood_clean = Ebb_and_flood.dropna(subset=['Amplitude', 'Q', 'Amplitude HD'])
# Calculate correlation coefficient and p-value
r_value, p_value = pearsonr(Ebb_and_flood_clean['Amplitude'], Ebb_and_flood_clean['Q'])
print(r_value, p_value)


# 26/04 : plot de water level vs Q pour voir le lag
cmap = cmc.cm.hawaii_r
fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle('Water level vs discharge', fontsize=fontsize)
ax.grid(True, alpha=0.5)
ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
ax.set_ylabel('Water level (m)', fontsize=fontsize - 2)
ax.scatter(water_levels[3], water_levels[0], marker='o', cmap=cmap,  vmin=0, vmax=4, label=' h')
cbar = plt.colorbar(p1, ax=ax)  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=fontsize - 4, width=0.5)
cbar.set_label(label='Tidal range HD (m)', fontsize=fontsize - 1)
cbar.outline.set_linewidth(0.05)
fig.savefig('test_water_level_vs_discharge.png')



# 12/05 : Je veux créer un tableau qui répertorie tous les lags calculés, selon toutes les catégories (Q1-Q4 pour TR et Q) pour faire des boxplots.
############################ DEBUT DE CONSTRUCTION DU TABLEAU   ################################"
list_nan_ebb = Ebb_df['Diff'].index[Ebb_df['Diff'].isna()].tolist()
list_nan_flood = Flood_df['Diff'].index[Flood_df['Diff'].isna()].tolist()
list_nan_flood.append(Flood_df.index[Flood_df['Amplitude HD'].isna()][0])
Ebb_nan_filter = Ebb_df.drop(list_nan_ebb)
Flood_nan_filter = Flood_df.drop(list_nan_flood)

# 12/05 : je rajoute un maillon des lags pour qu'ils se suivent tous !
Flood_nan_filter['lag Vmin-Vmax'] = Flood_df['Nearest datetime max Q'] - Ebb_df['Nearest datetime min Q']
Ebb_nan_filter['lag Vmin-Vmax'] = Flood_df['Nearest datetime max Q'] - Ebb_df['Nearest datetime min Q'] # Je sais,
# il y a une ebb = flood, en fait, cette colonne est la même peu importe ebb ou flood, c'est juste pour avoir le
# meme nombre de colonnes dans les 2 df
#Flood_nan_filter['lag LWS-Vmin'] = Ebb_df['Nearest datetime min Q'] - Flood_df['Diff slack water low water']


# DF avec TR categories
df_recap_lag_TR = pd.DataFrame(columns = ['LW propagation', 'Vmax-LW', 'LW-LWS', 'HW propagation', 'Vmin-HW', 'HW-HWS',
                                          'Vmax-Vmin'])
df_recap_lag_TR_std = pd.DataFrame(columns = ['LW propagation', 'Vmax-LW', 'LW-LWS', 'HW propagation', 'Vmin-HW', 'HW-HWS',
                                          'Vmax-Vmin'])

# Catégories avec les débits
df_recap_lag_Q = pd.DataFrame(columns = ['LW propagation', 'Vmax-LW', 'LW-LWS', 'HW propagation', 'Vmin-HW', 'HW-HWS',
                                          'Vmax-Vmin'])
df_recap_lag_Q_std = pd.DataFrame(columns = ['LW propagation', 'Vmax-LW', 'LW-LWS', 'HW propagation', 'Vmin-HW', 'HW-HWS',
                                          'Vmax-Vmin'])
quantile = True
a = 0
Ebb_nan_filter['Amplitude HD'] = Ebb_nan_filter['Amplitude HD'].abs()
for df in Flood_nan_filter, Ebb_nan_filter:
    if quantile:
        deb_dis = df.quantile([0.25, 0.5, 0.75])['Q'].values
        deb_TR = df.quantile([0.25, 0.5, 0.75])['Amplitude HD'].abs().values
    else:
        deb_dis = [500, 1000, 1500]
        deb_TR = [100, 200, 300]
    print('Quantile = ', str(quantile), 'Discharge value = ' + str(deb_dis))
    print('Quantile = ', str(quantile), 'TR value = ' + str(deb_TR))

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
    print('duration flood or ebb Q1 / Q4: ', Q1['Duration'].mean(), Q4['Duration'].mean())

    condA = (abs(df['Amplitude HD'])) < deb_TR[0]
    condB = (abs(df['Amplitude HD'] > deb_TR[0])) & (abs(df['Amplitude HD']) < deb_TR[1])
    condC = (abs(df['Amplitude HD']) > deb_TR[1]) & (abs(df['Amplitude HD']) < deb_TR[2])
    condD = (abs(df['Amplitude HD']) > deb_TR[2])
    TR1 = df.loc[condA]
    TR2 = df.loc[condB]
    TR3 = df.loc[condC]
    TR4 = df.loc[condD]
    print('len(TR1) et len(TR4) = ', len(TR1), len(TR4))
    print('duration flood or ebb TR1 / TR4: ', TR1['Duration'].mean(), TR4['Duration'].mean())

    cols = ['Diff water level', 'Diff', df.columns[-3]]
    print('ATTENTION AU NOM DES COLONNES !!!!§')

    for col in cols:
        print(col, 'a', str(a))
        df_recap_lag_Q[df_recap_lag_Q.columns[a]] = [Q[col].mean() for Q in [Q1,Q2,Q3,Q4]]
        df_recap_lag_Q_std[df_recap_lag_Q_std.columns[a]] = [Q[col].std() for Q in [Q1,Q2,Q3,Q4]]
        df_recap_lag_TR[df_recap_lag_TR.columns[a]] = [TR[col].mean() for TR in [TR1,TR2,TR3,TR4]]
        df_recap_lag_TR_std[df_recap_lag_TR_std.columns[a]] = [TR[col].std() for TR in [TR1,TR2,TR3,TR4]]
        print(str(TR1[col].mean()))
        a = a +1

col = 'lag Vmin-Vmax'
print(col, 'a', str(a))
df_recap_lag_Q[df_recap_lag_Q.columns[a]] = [Q[col].mean() for Q in [Q1, Q2, Q3, Q4]]
df_recap_lag_Q_std[df_recap_lag_Q_std.columns[a]] = [Q[col].std() for Q in [Q1, Q2, Q3, Q4]]

df_recap_lag_TR[df_recap_lag_TR.columns[a]] = [TR[col].mean() for TR in [TR1, TR2, TR3, TR4]]
df_recap_lag_TR_std[df_recap_lag_TR_std.columns[a]] = [TR[col].std() for TR in [TR1, TR2, TR3, TR4]]

# Je teste de déduire le lag LWS-Vmin à partir du lag Vmin Vmax - (LW-LWS + Vmas LW)
df_recap_lag_Q['LWS-Vmin'] = df_recap_lag_Q['Vmax-Vmin']- (df_recap_lag_Q['LW-LWS']+df_recap_lag_Q['Vmax-LW'])
df_recap_lag_TR['LWS-Vmin'] = df_recap_lag_TR['Vmax-Vmin']- (df_recap_lag_TR['LW-LWS']+df_recap_lag_TR['Vmax-LW'])
# Pour boucler la boucle : je rajoute le lag de HWS à Vmax
df_recap_lag_Q['HWS-Vmax'] = Flood_df['Nearest datetime max Q'] - Ebb_df['Nearest slack water']
df_recap_lag_TR['HWS-Vmax'] = Flood_df['Nearest datetime max Q'] - Ebb_df['Nearest slack water']

############################ FIN DE CONSTRUCTION DU TABLEAU   ################################"
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

# Figures temporaires
# Plot of low water lag propagation.
# PARAM COMMUNS ENTRE LES FIGURES
fontsize = 12
bar_height = 0.25  # Height of each bar
y_pos = np.arange(len(df_recap_lag_TR_seconds.index))

# FIGURE 1 : LW and HW propagation depending on TR :
fig, ax = plt.subplots(figsize=(6, 10))
plt.title('Water level propagation with several tidal range', fontsize = fontsize +2)
colors = ['sandybrown', 'skyblue', 'green', 'red', 'purple', 'brown', 'pink']  # Define colors for each lag
for i in range(len(df_recap_lag_TR_seconds)):
    print('i', i)
    for j, lag in enumerate(df_recap_lag_TR_seconds[['LW propagation', 'HW propagation']]):
        ax.barh(y_pos[i] + j * bar_height, df_recap_lag_TR_seconds[lag][i], bar_height,
                xerr=df_recap_lag_TR_std_seconds[lag][i], capsize = 1, ecolor = colors[j], label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j])
        value = df_recap_lag_TR[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        plt.text(200, y_pos[i] + j * bar_height + bar_height/2, f"{value.components.hours}h{zero}{value.components.minutes}",
                 va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        if i == 0 :
            ax.legend(fontsize = fontsize, loc='center right')
plt.xticks([])
plt.yticks([0.25, 1.25, 2.25, 3.25], ['TR1', 'TR2', 'TR3', 'TR4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'LW_HW_propagation_with_Tidal_range_categories'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')
#####
# FIGURE 2 : LW and HW propagation depending on Q :
fig, ax = plt.subplots(figsize=(6, 10))
plt.title('Water level propagation with several water discharges', fontsize = fontsize +2)
colors = ['sandybrown', 'skyblue', 'green', 'red', 'purple', 'brown', 'pink']  # Define colors for each lag
for i in range(len(df_recap_lag_Q_seconds)):
    print('i', i)
    for j, lag in enumerate(df_recap_lag_Q_seconds[['LW propagation', 'HW propagation']]):
        ax.barh(y_pos[i] + j * bar_height, df_recap_lag_Q_seconds[lag][i], bar_height,
                xerr=df_recap_lag_Q_std_seconds[lag][i], capsize = 1, ecolor = colors[j], label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j])
        value = df_recap_lag_Q[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        plt.text(200, y_pos[i] + j * bar_height + bar_height/2, f"{value.components.hours}h{zero}{value.components.minutes}",
                 va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        if i == 0 :
            ax.legend(fontsize = fontsize, loc='center right')
plt.xticks([])
plt.yticks([0.25, 1.25, 2.25, 3.25], ['Q1', 'Q2', 'Q3', 'Q4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'LW_HW_propagation_with_discharge_categories'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')
#######################################################

# FIGURE 3 : Les autres lags, mis à la queue leu leu pour classes de débit
fig, ax = plt.subplots(figsize=(6, 10))
plt.title('Lags with several water discharges', fontsize = fontsize +2)
colors = ['sandybrown', 'skyblue', 'green', 'red', 'purple', 'brown', 'pink']  # Define colors for each lag
start =  np.zeros(4)
for i in range(len(df_recap_lag_Q_seconds)):
    print('i', i)
    cum_val = 0
    for j, lag in enumerate(df_recap_lag_Q_seconds[['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS']]):
        print('lag = ',lag, 'j=', str(j))
        ax.barh(y_pos[i] - bar_height/2, df_recap_lag_Q_seconds[lag][i], bar_height, label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j], left = start[i])
        value = df_recap_lag_Q[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        if value.components.days < 0 :
            value = timedelta(hours=24) - value
            time_lag = f"-{value.components.hours}h{zero}{value.components.minutes}"
        else :
            time_lag = f"{value.components.hours}h{zero}{value.components.minutes}"
        plt.text(cum_val+10, y_pos[i], time_lag, va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        cum_val = cum_val + df_recap_lag_Q[lag][i].seconds
        #print(f"{value.components.hours}h{zero}{value.components.minutes}")
        start[i] = start[i] + df_recap_lag_Q_seconds[lag][i]
        print(start)
        if i == 0 :
            ax.legend(fontsize = fontsize, loc='center right')

plt.xticks([])
plt.yticks([0, 1, 2, 3], ['Q1', 'Q2', 'Q3', 'Q4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'Lags_at_TT_with_discharge_categories'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')

##############################################################
# FIGURE 4 : Les autres lags, mis à la queue leu leu pour classes de TR
fig, ax = plt.subplots(figsize=(6, 10))
plt.title('Lags with several tidal range', fontsize = fontsize +2)
colors = ['sandybrown', 'skyblue', 'green', 'red', 'purple', 'brown', 'pink']  # Define colors for each lag
start =  np.zeros(4)
for i in range(len(df_recap_lag_TR_seconds)):
    print('i', i)
    cum_val = 0
    for j, lag in enumerate(df_recap_lag_TR_seconds[['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS']]):
        print('lag = ',lag, 'j=', str(j))
        ax.barh(y_pos[i] - bar_height/2, df_recap_lag_TR_seconds[lag][i], bar_height, label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j], left = start[i])
        value = df_recap_lag_TR[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        if value.components.days < 0 :
            value = timedelta(hours=24) - value
            time_lag = f"-{value.components.hours}h{zero}{value.components.minutes}"
        else :
            time_lag = f"{value.components.hours}h{zero}{value.components.minutes}"
        plt.text(cum_val+10, y_pos[i], time_lag, va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        cum_val = cum_val + df_recap_lag_TR[lag][i].seconds
        #print(f"{value.components.hours}h{zero}{value.components.minutes}")
        start[i] = start[i] + df_recap_lag_TR_seconds[lag][i]
        print(start)
        if i == 0 :
            ax.legend(fontsize = fontsize, loc='center right')

plt.xticks([])
plt.yticks([0, 1, 2, 3], ['TR1', 'TR2', 'TR3', 'TR4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'Lags_at_TT_with_TR_categories'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')

#######################################
#######################################
#######################################
# Mêmes figures que les 4 écédantes, mais avec seulement Q1 et Q4.
y_pos = [0,0,0,1]
fontsize = 10
bar_height = 0.25  # Height of each bar

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
outfile = 'LW_HW_propagation_with_Tidal_range_Q1-Q4'
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
outfile = 'LW_HW_propagation_with_discharge_Q1-Q4'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')
#######################################################

# FIGURE 3 : Les autres lags, mis à la queue leu leu pour classes de débit
colors = ['coral', 'slategray', 'mediumseagreen', 'firebrick', 'purple', 'gold', 'pink']  # Define colors for each lag
fig, ax = plt.subplots(figsize=(4, 4))
#plt.title('Lags with several water discharges', fontsize = fontsize +2)
start =  np.zeros(4)
ax.set_ylim(-0.50,1.8)
for i in [0,3]:
    print('i', i)
    cum_val = 0
    for j, lag in enumerate(df_recap_lag_Q_seconds[['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax']]):
        print('lag = ',lag, 'j=', str(j))
        ax.barh(y_pos[i] - bar_height/2, df_recap_lag_Q_seconds[lag][i], bar_height, label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j], left = start[i])
        value = df_recap_lag_Q[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        if value.components.days < 0 :
            value = timedelta(hours=24) - value
            time_lag = f"-{value.components.hours}h{zero}{value.components.minutes}"
        else :
            time_lag = f"{value.components.hours}h{zero}{value.components.minutes}"
        if (lag == 'LWS-Vmin') and (i==3) :
            plt.text(cum_val-2000, y_pos[i]+0.15, time_lag, va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        else :
            plt.text(cum_val+100, y_pos[i], time_lag, va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
        cum_val = cum_val + df_recap_lag_Q[lag][i].seconds
        #print(f"{value.components.hours}h{zero}{value.components.minutes}")
        start[i] = start[i] + df_recap_lag_Q_seconds[lag][i]
        print(start)
        if i == 0 :
            fig.legend(fontsize = fontsize, loc='upper right')
plt.xticks([])
plt.yticks([0, 1], ['Q1', 'Q4'], fontsize = fontsize)  # Set custom y-axis tick labelsplt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
outfile = 'Lags_at_TT_with_discharge_Q1-Q4'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')

##############################################################
# FIGURE 4 : Les autres lags, mis à la queue leu leu pour classes de TR
fig, ax = plt.subplots(figsize=(4, 4))
#plt.title('Lags with several tidal range', fontsize = fontsize +2)
start =  np.zeros(4)
ax.set_ylim(-0.50,1.4)
for i in [0,3]:
    print('i', i)
    cum_val = 0
    for j, lag in enumerate(df_recap_lag_TR_seconds[['Vmax-LW', 'LW-LWS', 'LWS-Vmin', 'Vmin-HW', 'HW-HWS', 'HWS-Vmax']]):
        print('lag = ',lag, 'j=', str(j))
        ax.barh(y_pos[i] - bar_height/2, df_recap_lag_TR_seconds[lag][i], bar_height, label=lag, color=colors[j],
                align = 'edge', alpha=0.35, edgecolor=colors[j], left = start[i])
        value = df_recap_lag_TR[lag][i]
        zero = 0 if value.components.minutes < 10 else ''
        if value.components.days < 0 :
            value = timedelta(hours=24) - value
            time_lag = f"-{value.components.hours}h{zero}{value.components.minutes}"
        else :
            time_lag = f"{value.components.hours}h{zero}{value.components.minutes}"
        plt.text(cum_val+100, y_pos[i], time_lag, va='center', ha='left', fontsize = fontsize) # x = value.seconds/2.5
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
outfile = 'Lags_at_TT_with_TR_Q1-Q4'
if quantile :
    outfile = outfile + '_with_quantile'
outfile = outfile+'.png'
plt.savefig(outfile, format = 'png')



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

# Ici je calcule s'il y a une influence du TR ou Q sur la durée du cycle Vmin-Vmin
time_diffs_hours = np.array([(time_local_max[i+1] - time_local_max[i]).total_seconds() / 3600 for i in range(len(time_local_min) - 1)])
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Amplitude HD'], time_diffs_hours)
print(r_value, p_value)

# Calcul de Vlax -Vmin et Vmin Vmax et de l'influence de Q et TR
hours_series = (Ebb_df['Nearest datetime min Q'].shift(-1) - Flood_df['Nearest datetime max Q']).apply(lambda x: x.total_seconds() / 3600)
hours_series = hours_series[:-1]
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Amplitude HD'], hours_series)
print(r_value, p_value)

# Calcul de LWS-HWS et influence de Q et TR
hours_series = (Ebb_df['Nearest slack water'].shift(-1) - Flood_df['Nearest slack water']).apply(lambda x: x.total_seconds() / 3600)
hours_series = hours_series[:-1]
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Q'], hours_series)
print(r_value, p_value)

# Calcul de HWS-LWS et influence de Q et TR
hours_series = (Flood_df['Nearest slack water'] - Ebb_df['Nearest slack water']).apply(lambda x: x.total_seconds() / 3600)
hours_series = hours_series[:-1]
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Q'], hours_series)
print(r_value, p_value)

# Calcul de Vmax-LWS et influence de Q et TR
hours_series = (Flood_df['Nearest datetime max Q'] - Ebb_df['Nearest slack water']).apply(lambda x: x.total_seconds() / 3600)
hours_series = hours_series[:-1]
print(hours_series.mean())
# Calcul de LWS-Vmin et influence de Q et TR
hours_series = (Ebb_df['Nearest slack water'] - Ebb_df['Nearest datetime min Q']).apply(lambda x: x.total_seconds() / 3600)
hours_series = hours_series[:-1]
print(hours_series.mean())
# Calcul de Vmin-HWS et influence de Q et TR
hours_series = (Ebb_df['Nearest datetime min Q'] - Flood_df['Nearest slack water'].shift(1)).apply(lambda x: x.total_seconds() / 3600)
hours_series = hours_series[1:]
print(hours_series.mean())
# Calcul de HWS-Vmax et influence de Q et TR
hours_series = (Flood_df['Nearest slack water']-Flood_df['Nearest datetime max Q']).apply(lambda x: x.total_seconds() / 3600)
hours_series = hours_series[1:]
print(hours_series.mean())

slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_nan_filter['Q'], hours_series)
print(r_value, p_value)

######
#17/05 je rajoute dans df_recap_Q les valeurs de ces nouveaux lags.

Flood_nan_filter['Vmax-LWS'] = Flood_df['Nearest datetime max Q'] - Ebb_df['Nearest slack water']
Flood_nan_filter['LWS-Vmin'] = Ebb_df['Nearest slack water'] - Ebb_df['Nearest datetime min Q']
Flood_nan_filter['Vmin-HWS'] = (Ebb_df['Nearest datetime min Q'] - Flood_df['Nearest slack water'].shift(1))
Flood_nan_filter['HWS-Vmax'] = Flood_df['Nearest slack water']-Flood_df['Nearest datetime max Q']

quantile = True
a = 0
for df in Flood_nan_filter:
    if quantile:
        deb_dis = df.quantile([0.25, 0.5, 0.75])['Q'].values
    else:
        deb_dis = [500, 1000, 1500]
    print('Quantile = ', str(quantile), 'Discharge value = ' + str(deb_dis))
    cond1 = df['Q'] < deb_dis[0]
    cond2 = (df['Q'] > deb_dis[0]) & (df['Q'] < deb_dis[1])
    cond3 = (df['Q'] > deb_dis[1]) & (df['Q'] < deb_dis[2])
    cond4 = df['Q'] > deb_dis[2]
    Q1 = df.loc[cond1]
    Q2 = df.loc[cond2]
    Q3 = df.loc[cond3]
    Q4 = df.loc[cond4]
    print('len(Q1) et len(Q4) = ', len(Q1), len(Q4))
    print('duration flood or ebb Q1 / Q4: ', Q1['Duration'].mean(), Q4['Duration'].mean())

    cols = ['Vmax-LWS', 'LWS-Vmin', 'Vmin-HWS', 'HWS-Vmax']
    for col in cols:
        print(col, 'a', str(a))
        df_recap_lag_Q[df_recap_lag_Q.columns[a]] = [Q[col].mean() for Q in [Q1,Q2,Q3,Q4]]
        df_recap_lag_Q_std[df_recap_lag_Q_std.columns[a]] = [Q[col].std() for Q in [Q1,Q2,Q3,Q4]]
        a = a +1


# Je calcule la somme pour voir si l'on arrive à une valeur d'une période entière
df_recap_lag_TR[['Vmax-LW', 'LW-LWS', 'Vmin-HW','HW-HWS', 'LWS-Vmin', 'HWS-Vmax']].sum(axis=1)

# 24/05/24 : I want to pick an exemple in order to illustrate all the lags between current and water level
# First, I find a list of days close to median discharge
crit_Q = 20
crit_TR = 0.1
days_closest_median_discharge = daily_mean.loc[(daily_mean['Q'] - daily_mean['Q'].median()).abs() < crit_Q]
days_closest_median_TR = Ebb_and_flood.loc[(Ebb_and_flood['Amplitude HD'] - Ebb_and_flood['Amplitude HD'].median()).abs() < crit_TR]
year = 2021
m = 8
d1 = 1
d2 = 10
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

fig, ax = plt.subplots(figsize=(20, 10), nrows=1, sharex=True)
ax.grid(which='both', alpha=0.5)
ax.set_xlabel('Time', fontsize=fontsize - 5)
ax.set_ylabel('Tidal elevation at TT (m)', fontsize=fontsize - 5)
ax.plot(water_levels['Datetime'], water_levels[water_levels.columns[0]] / 100, label='TT water level',
        color='black', lw=2, zorder=0.1)
ax.scatter(time_local_min, local_minima / 100, marker='x', color='black', s=s, zorder=1)
ax.scatter(time_local_max, local_maxima / 100, marker='x', color='black', s=s, zorder=1)

twin = ax.twinx()
twin.set_ylabel('Discharge at TT (m$³$/s)', fontsize=fontsize - 5, color = 'coral')
twin.tick_params(axis='y', colors='coral')  # Set tick label color
twin.spines['right'].set_color('coral')    # Set spine color
twin.plot(water_levels['Datetime'], water_levels['Q'], label='TT Discharge', ls='--', color='coral',
          zorder=0.1)
twin.plot(daily_mean.index, daily_mean['Q'], label='Daily discharge', ls='--', color='coral',
          zorder=0.1)
twin.axhline(0, color='coral')
twin.scatter(time_local_min_Q, local_minima_Q, label='extreme values', marker='x', color='black', s=s, zorder=1)
twin.scatter(time_local_max_Q, local_maxima_Q, marker='x', color='black', s=s, zorder=1)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = twin.get_legend_handles_labels()
twin.legend(lines + lines2, labels + labels2, fontsize=fontsize - 10)

for (x1, x2) in zip(time_local_min[indice_min_wl], time_local_max_Q[indice_max_Q]):
    ax.axvspan(x1, x2, ymin=-10, ymax=10, facecolor='gray', alpha=0.2)
for (x3, x4) in zip(time_local_min_Q[indice_min_Q], time_local_max[indice_max_wl]):
    ax.axvspan(x3, x4, ymin=-10, ymax=10, facecolor='teal', alpha=0.2)

date_form = DateFormatter("%d/%m/%Y")  # Define the date format
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=12))
ax.xaxis.set_major_formatter(date_form)
ax.set_xlim(datetime(year, m, d1), datetime(year, m, d2))
fig.savefig('exemple_tidal_elevations_Q_lag_' + str(d1) + str(m) + str(year) + '.png', format='png')




###############
##################
#### JNGCGC
# Il faut d'abord ajouter le débit journalier à la période :
# J'ajoute le débit journalier le plus proche de Ebb et Flood datetime
result_value_Q = []
for index1, row1 in Ebb.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_Q.append(daily_mean['Q'].iloc[closest_index])
Ebb['Q'] = result_value_Q
result_value_Qf = []
for index1, row1 in Flood.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_Qf.append(daily_mean['Q'].iloc[closest_index])
Flood['Q'] = result_value_Qf
result_value_QHD = []
for index1, row1 in Ebb_HD.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_QHD.append(daily_mean['Q'].iloc[closest_index])
Ebb_HD['Q'] = result_value_QHD
result_value_QfHD = []
for index1, row1 in Flood_HD.iterrows():
    datetime_Q = daily_mean.index + timedelta(hours=12)  # because the mean is done from midnight to 23h
    closest_index = np.argmin(np.abs(datetime_Q - row1[D]))  # Find the closest datetime index
    result_value_QfHD.append(daily_mean['Q'].iloc[closest_index])
Flood_HD['Q'] = result_value_QfHD
# Puis créer les df.


Ebb_HD_copy = Ebb_HD.copy()
Ebb_HD_copy = Ebb_HD_copy.rename(columns={col: col + ' HD' for col in Ebb_HD_copy.columns})
Ebb_df = pd.concat([Ebb_HD_copy, Ebb], axis=1)
Ebb_df['Amplitude HD'] = -Ebb_df['Amplitude HD']
Ebb_df['Amplitude'] = -Ebb_df['Amplitude']
Flood_HD_copy = Flood_HD.copy()
Flood_HD_copy = Flood_HD_copy.rename(columns={col: col + ' HD' for col in Flood_HD_copy.columns})
Flood_df = pd.concat([Flood_HD_copy, Flood], axis=1)


x = np.arange(0, 4, 0.1)
color0 = 'teal' #darkgreen
color1 = 'coral' #hotpink
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# fig.suptitle('Ebb-Flood duration at TT vs tidal range', fontsize=fontsize)
ax = axs[0]
ax.set_title('Ebb', fontsize = fontsize)
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Duration (h)', fontsize=fontsize - 2)
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(5))
y_ebb = (Ebb_df['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(abs(Ebb_df['Amplitude HD'] / 100), y_ebb, marker='o', color=color0, s=20)
slope, intercept, r_value, p_value1, std_err = stats.linregress(abs(Ebb_df['Amplitude HD'] / 100), y_ebb)
#label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
label = 'r = ' + str(np.round(r_value, 2)) + ' , p = ' + " {:.2e}".format(p_value1)
ax.plot(x, slope * x + intercept, color='k', lw=4, label=label)
ax.legend()

ax = axs[1]
ax.set_title('Flood', fontsize = fontsize)
ax.grid(which='both', alpha=0.5)
ax.set_xlabel('Tidal range at HD (m)', fontsize=fontsize - 2)
ax.set_ylabel('Duration (h)', fontsize=fontsize - 2)
#ax.set_yticks(np.arange(5,22.5,2.5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(5))
y_flood = (Flood_df['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(Flood_df['Amplitude HD'] / 100, y_flood, marker='<', color=color1, s=20)
slope, intercept, r_value, p_value2, std_err = stats.linregress(Flood_df['Amplitude HD'].dropna() / 100,
                                                               y_flood.dropna())
#label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
label = 'r = ' + str(np.round(r_value, 2)) + ' , p = ' + " {:.2e}".format(p_value2)
ax.plot(x, slope * x + intercept, color='k', lw=4, label=label)
ax.legend()
outfile = 'test_Ebb_Flood_duration_TT_vs_tidal_range'
filter = False
if filter:
    outfile = outfile + '_filtered_at_' + str(treshold)
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

x = np.arange(50, 2200, 1)
fig, axs = plt.subplots(figsize=(18, 10), nrows=2, sharex=True)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# fig.suptitle('Ebb-Flood duration at TT vs tidal range', fontsize=fontsize)
ax = axs[0]
ax.set_title('Ebb', fontsize = fontsize)
ax.grid(which='both', alpha=0.5)
ax.set_ylabel('Duration (h)', fontsize=fontsize - 2)
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(5))
y_ebb = (Ebb_df['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(abs(Ebb_df['Q'].dropna()), y_ebb.dropna(), marker='o', color=color0, s=20)
slope, intercept, r_value, p_value1, std_err = stats.linregress(abs(Ebb_df['Q'].dropna()), y_ebb.dropna())
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
label = 'r = ' + str(np.round(r_value, 2)) + ' , p = ' + " {:.2e}".format(p_value1)
ax.plot(x, slope * x + intercept, color='k', lw=4, label=label)
ax.legend()

ax = axs[1]
ax.set_title('Flood', fontsize = fontsize)
ax.grid(which='both', alpha=0.5)
ax.set_xlabel('Discharge (m$³$/s)', fontsize=fontsize - 2)
ax.set_ylabel('Duration (h)', fontsize=fontsize - 2)
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(5))
y_flood = (Flood_df['Duration'].dt.total_seconds() / 3600).astype(float)
ax.scatter(Flood_df['Q'][0:-1], y_flood.dropna(), marker='<', color=color1, s=20)
slope, intercept, r_value, p_value2, std_err = stats.linregress(Flood_df['Q'][0:-1], y_flood.dropna())
label = " {:.2e}".format(slope) + ' x + ' + str(np.round(intercept, 2)) + ', r=' + str(np.round(r_value, 2))
label = 'r = ' + str(np.round(r_value, 2)) + ' , p = ' + " {:.2e}".format(p_value2)
ax.plot(x, slope * x + intercept,  color='k', lw=4, label=label)
ax.legend()
outfile = 'Ebb_Flood_duration_TT_vs_discharge_alldata'
filter = False
if filter:
    outfile = outfile + '_filtered_at_' + str(treshold)
outfile = outfile + '.png'
fig.savefig(outfile, format='png')

y_ebb = (handle_outliers(Ebb_df.dropna(), 'Diff water level').dt.total_seconds() / 3600).astype(float)
y_flood = (handle_outliers(Flood_df.dropna(), 'Diff water level').dt.total_seconds() / 3600).astype(float)
#y_flood = (Flood_df['Diff water level'].dt.total_seconds() / 3600).astype(float)
#y_ebb = (Ebb_df['Diff water level'].dt.total_seconds() / 3600).astype(float)
slope, intercept, r_value, p_value, std_err = stats.linregress(Ebb_df.dropna()['Q'], y_ebb)
print(p_value, r_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(Flood_df.dropna()['Q'], y_flood)
print(p_value, r_value)


## 5/07
quantile = True
if quantile:
    deb = Ebb_df.quantile([0.25, 0.5, 0.75])['Q'].values
    TR = Ebb_df.quantile([0.25, 0.5, 0.75])['Amplitude HD'].abs().values
else:
    deb = [500, 1000, 1500]
    TR = [100, 200, 300]

cond1 = Ebb_df['Q'] < deb[0]
cond2 = (Ebb_df['Q'] > deb[0]) & (Ebb_df['Q'] < deb[1])
cond3 = (Ebb_df['Q'] > deb[1]) & (Ebb_df['Q'] < deb[2])
cond4 = Ebb_df['Q'] > deb[2]
# Condition on the amplitude
condA = (abs(Ebb_df['Amplitude HD']) < TR[0])
condB = (TR[0] < abs(Ebb_df['Amplitude HD'])) & (abs(Ebb_df['Amplitude HD']) <= TR[1])
condC = (TR[1] < abs(Ebb_df['Amplitude HD'])) & (abs(Ebb_df['Amplitude HD']) <= TR[2])
condD = (abs(Ebb_df['Amplitude HD']) > TR[2])



# Value of min and max discharge and correlation with TR and discharge

# Tidal range data
tidal_range = pd.DataFrame({'Time LT': time_local_min_HD, 'Val LT':local_minima_HD/100,
                            'Time HT': time_local_max_HD, 'Val HT': local_maxima_HD/100})
tidal_range['mean TR'] = abs(tidal_range['Val LT'] - tidal_range['Val HT'])
tidal_range['Date'] = tidal_range['Time LT'].dt.date


diff_discharge_init = pd.DataFrame()
diff_discharge = diff_discharge_init.reindex(tidal_range['Date'])
diff_discharge['TR'] = tidal_range['mean TR'].values

# Insertion of the discharge values
daily_mean_aligned = daily_mean.reindex(diff_discharge.index)
diff_discharge['Q'] = daily_mean_aligned['Q'].values

# Valeur des cycles min max
val = abs(df_min_discharge['Value'] - df_max_discharge['Value']).values
time_val = (df_min_discharge['Datetime']).values
df_amplitude = pd.DataFrame({'Datetime': time_val, 'Diff_min_max': val})
df_amplitude['Date'] = df_amplitude['Datetime'].dt.date

merged = pd.merge(diff_discharge, df_amplitude, on='Date', how='left')

merged_clean = merged.dropna(how='any')
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_clean['Diff_min_max'],
                                                               merged_clean['TR'])
print('corr with TR ','r_value ', r_value, 'p ', p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_clean['Diff_min_max'],
                                                               merged_clean['Q'])
print('corr with discharge ','r_value ', r_value, 'p ', p_value)

print(merged_clean.quantile([0.25,0.5,0.75]))
