# 29/05/24 : Je fais une nouvelle page propre dédiée aux lags, donc je ne calcule que les HW, LW, HWS, LWS, Vmin , Vmax.
# 7/06/24 : je reprends en construisant différemment mon tableau : en commencant par HW
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
# VARIABLE AND PARAMETER
# FIGURE PARAMETER
fontsize = 28
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 4
s = 25

#################################################################################""
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

#################################################################################################
# 1. Je détecte les max et min et calcule les durées à Hon Dau et à Trung Trang
print('Hello again ! I try to find the ebb and flood at TT ...')
# I calculate the min and max of the tides AT TT
# HW and LW AT TT
window_size = 17
local_minima, local_maxima, time_local_min, time_local_max, a, b = \
    find_local_minima(water_levels, water_levels.columns[0], window_size)
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
col2 = water_levels.columns[2]
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
    find_local_minima(water_levels, water_levels.columns[3], window_size)
df_min_discharge = pd.DataFrame(time_local_min_Q)
# the starting datetime is the beginning of the Ebb_HD i.e : max water levels at TT
df_max_discharge = pd.DataFrame(time_local_max_Q)
df_min_discharge = df_min_discharge.rename(columns={0: 'Datetime Vmin'})
df_max_discharge = df_max_discharge.rename(columns={0: 'Datetime Vmax'})

# SW detection with the resampled serie.
val_interp = 5
resampled_series = water_levels.copy()
resampled_series = resampled_series.set_index('Datetime')
resampled_series = resampled_series.resample(str(val_interp)+'T').asfreq()  # Resample by adding values every 5T
interpolated_series = resampled_series.interpolate(method='linear')
window_size_interp = int(17*60/val_interp)
local_minima_Q_interp, local_maxima_Q_interp, time_local_min_Q_interp, time_local_max_Q_interp, \
local_SW, time_local_SW = \
    find_local_minima(interpolated_series.reset_index(), interpolated_series.reset_index().columns[3],
                      window_size_interp, interp=True)

df_SW = pd.DataFrame(time_local_SW)
df_SW = df_SW.rename(columns={0: 'Datetime SW'})

#####################  FIN DEFINITION DES df de tous types

# Il ne me manque plus que la daily discharge, et le tidal range.
# Je rajoute une colonne daily Q, qui sera le débit journalier à la date de Vmin
# (arbitrairement mais choisi pour être sûre qu'il y en ait un différent à chaque ligne (i.e. soit Vmin, soit Vmax)
daily_mean['Date'] = daily_mean.index
# Create a dictionary mapping Datetime Vmin to Daily Mean Discharge
daily_mean_map = dict(zip(daily_mean['Date'], daily_mean['Q']))

# Et maintenant le tidal range.
# Je fais le choix pour le TR de 1. faire la moyenne de EBb et Flood. ou
# 2. de ne garder que les valeurs de Ebb, afin d'éviter les nan
# A HON DAU !!!!!!!!
daily_mean_map_TR = dict(zip(Ebb_HD['Datetime'].dt.date, (Flood_HD['Amplitude HD'] + abs(Ebb_HD['Amplitude HD']))/200))

#########################################################
### Merging the HW and the LW
merged_hw = pd.merge_asof(df_HW_HD[['Datetime HW HD']], df_HW[['Datetime HW']],
                           left_on='Datetime HW HD', right_on='Datetime HW',
                           direction='forward')

merged_hw['Q'] = merged_hw['Datetime HW HD'].dt.date.map(daily_mean_map)
merged_hw['TR'] = merged_hw['Datetime HW HD'].dt.date.map(daily_mean_map_TR)
merged_hw['TR'] = merged_hw['TR'].interpolate()

merged_lw = pd.merge_asof(df_LW_HD["Datetime LW HD"], df_LW[['Datetime LW']],
                           left_on='Datetime LW HD', right_on='Datetime LW',
                           direction='forward')

if merged_hw['Datetime HW HD'].iloc[0] < merged_lw['Datetime LW HD'].iloc[0] :
    combined_df = pd.concat([merged_hw, merged_lw], axis=1)
else :
    combined_df = pd.concat([merged_lw, merged_hw], axis=1)

# Etape de filtrage pour supprimer les marée que l'on ne détecte pas.
#combined_df = combined_df[(combined_df['Datetime HW']-combined_df['Datetime HW HD'])<pd.Timedelta(hours=5)]
#combined_df = combined_df[(combined_df['Datetime LW']-combined_df['Datetime LW HD'])<pd.Timedelta(hours=7)]

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

# Step 2 : insert the slack water inbetween the Vmin and the next Vmax
SW_after_min = pd.merge_asof(df_min_discharge,  df_SW,
    left_on='Datetime Vmin', right_on='Datetime SW', direction='forward')

# Filter the SW_after_max DataFrame to ensure slack water times are before the next Vmin
SW_after_min['Next Vmax'] = df_max_discharge['Datetime Vmax'].shift(-1, fill_value=pd.Timestamp.max)
filtered_SW_after_min = SW_after_min[SW_after_min['Datetime SW'] < SW_after_min['Next Vmax']]



if df_max_discharge['Datetime Vmax'].iloc[0] < df_min_discharge['Datetime Vmin'].iloc[0] :
    combined_df = pd.concat([df_max_discharge, df_min_discharge], axis=1)
else :
    combined_df = pd.concat([df_min_discharge, df_max_discharge], axis=1)

# Insert the filtered slack water times into the combined DataFrame
combined_df['LWS'] = filtered_SW_after_max['Datetime SW']
combined_df = combined_df[['Datetime Vmax', 'LWS', 'Datetime Vmin']] #Slack water after Vmax is LWS.
# LWS = Slack Water After Vmax


# Insert the filtered slack water times into the combined DataFrame
combined_df['HWS'] = filtered_SW_after_min['Datetime SW']
# HWS = Slack Water After Vmin
df_current_lag = combined_df.copy()

# Step 3 : Insérer les HW et LW.
# HW après Vmin et LW après Vmax
merged_hw = pd.merge_asof(df_HW[['Datetime HW HD']], df_HW[['Datetime HW']],
                           left_on='Datetime HW HD', right_on='Datetime HW',
                           direction='forward')

#merged_hw_HD = pd.merge_asof(df_min_discharge, df_HW_HD[['Datetime HW HD']],
#                           left_on='Datetime Vmin', right_on='Datetime HW HD',
#                           direction='forward')

merged_lw = pd.merge_asof(df_LW["Datetime LW HD"], df_LW[['Datetime LW']],
                           left_on='Datetime Vmax', right_on='Datetime LW',
                           direction='forward')

#merged_lw_HD = pd.merge_asof(df_max_discharge, df_LW_HD[['Datetime LW HD']],
#                           left_on='Datetime Vmax', right_on='Datetime LW HD',
#                           direction='forward')

# Reorder the columns as needed
combined_df['HW'] = merged_hw['Datetime HW']
combined_df['LW'] = merged_lw['Datetime LW']

combined_df = combined_df.rename(columns={'Datetime Vmin': 'Vmin', 'Datetime Vmax': 'Vmax'})
combined_df = combined_df[['Vmax', 'LW', 'LWS', 'Vmin', 'HW', 'HWS']] #Slack water after Vmax is LWS.
# Check, j'ai bien un cycle de LWS-LWS qui dure 25h09, 25h03 pour HWS-HWS, 25h01 pour la médiane des 2 cycles.
lag = 'HW'
median_cycle = (combined_df[lag] - combined_df[lag].shift(1)).median()



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
lag1 = 'Vmin'
lag2 = 'Vmax'
shift = True # If I the part to examine is in the following line
# Apply the shift conditionally
lag2_series = combined_df[lag2].shift(-1) if shift else combined_df[lag2]
# Find the indices of NaN values
list_nan1 = combined_df.index[combined_df[lag1].isna()].tolist()
list_nan2 = combined_df.index[lag2_series.isna()].tolist()
list_nan = list_nan1 + list_nan2
combined_df_filter = combined_df.drop(list_nan)
lag2_series = lag2_series.drop(list_nan) if shift else combined_df_filter[lag2]
# I want to check the impact of river discharge and TR on the different lags
print('lag ', lag1, lag2)
mean_val = (lag2_series- combined_df_filter[lag1]).mean()
q1, median_val, q4 = (lag2_series - combined_df_filter[lag1]).quantile([0.25, 0.5, 0.75])
print('mean', mean_val,'\n', 'Q1', q1,'\n', 'med', median_val, '\n','Q4', q4)
time_diffs_hours = (lag2_series - combined_df_filter[lag1]).dt.total_seconds()/3600
#np.array([(time_local_max[i+1] - time_local_max[i]).total_seconds() / 3600 for i in range(len(time_local_min) - 1)])
slope, intercept, r_value, p_value, std_err = stats.linregress(combined_df_filter['TR'], time_diffs_hours)
print('correlation with TR', r_value, p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(combined_df_filter['Q'], time_diffs_hours)
print('correlation with Q', r_value, p_value)


############################################################################################
############################################################################################
############################################################################################
# Je créé un df de toutes ces diffs.
df_diff = pd.DataFrame()
lag1 = 'Vmin'
lag2 = 'Vmax'
shift = True # If I the part to examine is in the following line
# Apply the shift conditionally
lag2_series = combined_df[lag2].shift(-1) if shift else combined_df[lag2]



############################################################################################
############################################################################################
############################################################################################
# Now that I do have all the values I want in the table, I can focus on the quartiles TR and Q
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
if quantile:
    deb_dis = combined_df.quantile([0.25, 0.5, 0.75])['Q'].values
    deb_TR = combined_df.quantile([0.25, 0.5, 0.75])['Amplitude HD'].abs().values
else:
    deb_dis = [500, 1000, 1500]
    deb_TR = [100, 200, 300]
print('Quantile = ', str(quantile), 'Discharge value = ' + str(deb_dis))
print('Quantile = ', str(quantile), 'TR value = ' + str(deb_TR))

cond1 = combined_df['Q'] < deb_dis[0]
cond2 = (combined_df['Q'] > deb_dis[0]) & (combined_df['Q'] < deb_dis[1])
cond3 = (combined_df['Q'] > deb_dis[1]) & (combined_df['Q'] < deb_dis[2])
cond4 = combined_df['Q'] > deb_dis[2]
Q1 = combined_df.loc[cond1]
Q2 = combined_df.loc[cond2]
Q3 = combined_df.loc[cond3]
Q4 = combined_df.loc[cond4]
print('')
print('len(Q1) et len(Q4) = ', len(Q1), len(Q4))
print('duration flood or ebb Q1 / Q4: ', Q1['Duration'].mean(), Q4['Duration'].mean())

condA = (abs(combined_df['Amplitude HD'])) < deb_TR[0]
condB = (abs(combined_df['Amplitude HD'] > deb_TR[0])) & (abs(combined_df['Amplitude HD']) < deb_TR[1])
condC = (abs(combined_df['Amplitude HD']) > deb_TR[1]) & (abs(combined_df['Amplitude HD']) < deb_TR[2])
condD = (abs(combined_df['Amplitude HD']) > deb_TR[2])
TR1 = combined_df.loc[condA]
TR2 = combined_df.loc[condB]
TR3 = combined_df.loc[condC]
TR4 = combined_df.loc[condD]
print('len(TR1) et len(TR4) = ', len(TR1), len(TR4))
print('duration flood or ebb TR1 / TR4: ', TR1['Duration'].mean(), TR4['Duration'].mean())

cols = ['Diff water level', 'Diff', combined_df.columns[-3]]
print('ATTENTION AU NOM DES COLONNES !!!!§')

for col in cols:
    print(col, 'a', str(a))
    df_recap_lag_Q[df_recap_lag_Q.columns[a]] = [Q[col].mean() for Q in [Q1,Q2,Q3,Q4]]
    df_recap_lag_Q_std[df_recap_lag_Q_std.columns[a]] = [Q[col].std() for Q in [Q1,Q2,Q3,Q4]]
    df_recap_lag_TR[df_recap_lag_TR.columns[a]] = [TR[col].mean() for TR in [TR1,TR2,TR3,TR4]]
    df_recap_lag_TR_std[df_recap_lag_TR_std.columns[a]] = [TR[col].std() for TR in [TR1,TR2,TR3,TR4]]
    print(str(TR1[col].mean()))
    a = a +1



########
######## TESTS
########
print('ok')
