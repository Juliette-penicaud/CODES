# 22/07. Objectif : calculer pourcentage de marée
# 16/05/2023 : Je veux généraliser la méthode de calcul pour que les HT et LT soient d'office detectées.
# Je veux travailler sur la relation entre % marée (temps) par rapp à % hauteur d'eau (dans estuaire? => Quelles val ?
# avec les données de ADCP Violaine ?) et l'intrusion saline (salinité aux points fixe?) et les vitesses de courants
# 22/11/2023 : je reprends le code avec la fonctione développée pour avoir detecter min et max en général.
# Puis pour avoir les valeurs avec les données effectives de HD et pas prédites
import numpy as np
import csv
import sys
import pandas as pd
from datetime import datetime, timedelta


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
    # time_local_min = [dt.replace(microsecond=0) for dt in time_local_min]
    # time_local_max = [dt.replace(microsecond=0) for dt in time_local_max]
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

    return np.array(local_minima2), np.array(local_maxima2), np.array(time_local_min2), np.array(time_local_max2)


# Function to find the closest datetime in df1 for each datetime in df2
def find_closest_datetime(dt, datetime_list):
    closest_dt = min(datetime_list, key=lambda x: abs((x - dt).total_seconds()))
    return closest_dt


# COnvention : 100% is HT, 0% is LT, + from L to HT and - from H to LT.
# convention 1 : From HT to LT, we begin by -99% near to HT, -1% near to LT.  CIRCLE
# Convention 2 : Begin by -1% next to HT to arrive to -99% newt to LT

# Trouble convention 1 : representation on graphs ..

#######################"           Tide Do Son   PREDITES    ################################
file_DoSon = '/home/penicaud/Documents/Data/Décharge_waterlevel/tide_DoSon_May-Nov2022.xlsx'
data_DoSon = pd.read_excel(file_DoSon, usecols=['Date', 'hour', 'Tide'])

####################### TIDE DO SON DATA ##################################
file = '/home/penicaud/Documents/Data/Décharge_waterlevel/Data_2021-2022.xlsx'

columns_to_load = list(range(25))
df = pd.read_excel(file, sheet_name='sea_level-HonDau_2021-2022', usecols=columns_to_load, skiprows=4)
df = df.rename(columns={'Unnamed: 0': 'Date'})
melted_df = pd.melt(df, id_vars=["Date"], var_name="Hour", value_name="Value")
melted_df['Datetime'] = pd.to_datetime(melted_df['Date']) + pd.to_timedelta(melted_df['Hour'], unit='h')
melted_df.sort_values("Datetime", inplace=True)
melted_df = melted_df.dropna()
# data_to_comp = melted_df.loc[(melted_df['Datetime'].dt.year == 2022 ) & (melted_df['Datetime'].dt.month >= 5) &
#                             (melted_df['Datetime'].dt.month < 9)].dropna().reset_index()
melted_df['Value'] = melted_df['Value'] / 100
data_DoSon = melted_df.copy()
data_DoSon = data_DoSon.drop('Hour', axis=1)
data_DoSon = data_DoSon.rename(columns={'Value': 'Tide'})  # 'Datetime' : 'hour'
data_DoSon = data_DoSon.reset_index()

# Interpolation des données à Do SOn
resampled_series = data_DoSon.copy()
resampled_series = resampled_series.set_index('Datetime')
resampled_series = resampled_series.resample('1T').asfreq()  # Resample by adding values every 1mn
resampled_series = resampled_series.drop('index', axis=1)
resampled_series['Date'] = resampled_series['Date'].interpolate(method='ffill')
resampled_series['Tide'] = resampled_series['Tide'].interpolate(method='linear')

# Calculation of the percentage on the whole dataset
local_minima_data, local_maxima_data, time_local_min_data, time_local_max_data = \
    find_local_minima(data_DoSon, 'Tide', 17)
Ebb_data = pd.DataFrame(time_local_max_data)
# the starting datetime is the beginning of the Ebb_data i.e : max water levels at TT
Flood_data = pd.DataFrame(time_local_min_data)
Ebb_data = Ebb_data.rename(columns={0: 'Datetime'})
Flood_data = Flood_data.rename(columns={0: 'Datetime'})
if time_local_max_data[0] > time_local_min_data[0]:  # To know which one we need to substract
    print('The first extremum is the minimum data, so it is the flood_data')
    Flood_data['Duration'] = time_local_max_data - time_local_min_data
    Flood_data['Amplitude'] = local_maxima_data - local_minima_data
    Ebb_data['Duration'] = np.roll(time_local_min_data, shift=-1) - time_local_max_data
    Ebb_data['Amplitude'] = np.roll(local_minima_data, shift=-1) - local_maxima_data
    Ebb_data['Duration'].iloc[-1] = np.nan
    Ebb_data['Amplitude'].iloc[-1] = np.nan
else:
    print('The first extremum is the MAX data, so it is the Ebb_data')
    Flood_data['Duration'] = np.roll(time_local_max_data, shift=-1) - time_local_min_data
    Flood_data['Amplitude'] = np.roll(local_maxima_data, shift=-1) - local_minima_data
    Flood_data['Duration'].iloc[-1] = np.nan
    Flood_data['Amplitude'].iloc[-1] = np.nan
    Ebb_data['Duration'] = time_local_min_data - time_local_max_data
    Ebb_data['Amplitude'] = local_minima_data - local_maxima_data

##################"
# VARIABLE AND PARAMETER
year = '2022'
list_month = ['June', 'August', 'Octobre']
i = 2  # 0 1 2
month = list_month[i]
rep = '/home/penicaud/Documents/Data/'
file = rep + 'Survey_' + month + '/Stations_' + month + '.xlsx'

dict_month = {'June': {'skiprow': 0, 'sheetname': 'All', 'nrows': 89},
              'August': {'skiprow': 2, 'sheetname': 'A1-AF38', 'nrows': 86},
              'Octobre': {'skiprow': 1, 'sheetname': 'O1-FO52', 'nrows': 98}}

print('file', file)
col_list = ["Stations", "Time"]  # "Station N", "Station E", "Time"]
dtypes = {'Stations': 'str', 'Time': 'str'}  # 'Station N': 'str', 'Station E': 'str',
parse_dates = ['Time']

df_month = pd.read_excel(file, usecols=col_list,
                         sheet_name=dict_month[month]['sheetname'],
                         parse_dates=parse_dates)  # nrows=dict_month[month]['nrows'],
df_month = df_month.dropna()
# df_month['Time'] = datetime.strptime(df_month['Date'], '%Y-%m-%d %H:%M:%S')
df_month['Date'] = pd.to_datetime(df_month['Time']).dt.date  # Date needed to compare with the DOSON file
print('df', df_month)

####################################""

list_percentage, list_tide = [], []
for sta in df_month['Stations']:
    print('sta', sta)
    df_sta = df_month.loc[df_month['Stations'] == sta].copy()  # df of one line with the Date , Time and station name
    # Je cherche ou j'ai la même date
    df_good_time = resampled_series.loc[resampled_series.index == df_sta['Time'].values[0]]
    # I look for the closest datetime in Ebb and Flood df
    time_sta = df_good_time.index.values[0]
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
            print('Case 3 : Je recalcule le flood, je suis en ebb') # Not useful to calculate the closest flood, but allow to manually cjeck
            closest_flood = find_closest_datetime(time_sta,
                                                  Flood_data['Datetime'].loc[
                                                      Flood_data['Datetime'] >= time_sta])
            duration_ebb = Ebb_data['Duration'].loc[Ebb_data['Datetime'] == closest_ebb]
            from_start = time_sta - closest_ebb
            percentage = - (100 - (from_start / duration_ebb * 100))
        elif (closest_both < time_sta) & (closest_both == closest_flood):
            print('Case 4 : Je recalcule le Ebb, je suis en flood') # Not useful to calculate the closest flood, but allow to manually cjeck
            closest_ebb = find_closest_datetime(time_sta,
                                                Ebb_data['Datetime'].loc[Ebb_data['Datetime'] >= time_sta])
            duration_flood = Flood_data['Duration'].loc[Flood_data['Datetime'] == closest_flood]
            from_start = time_sta - closest_flood
            percentage = from_start / duration_flood * 100
            break
        else :
            print('NO comprendo')
            break
    print(percentage)
    list_percentage.append(percentage.values[0])
    list_tide.append(df_good_time['Tide'].values[0])


df_percentage = pd.DataFrame(df_month['Stations'])
df_percentage = df_percentage.reset_index()
df_percentage = df_percentage.drop(['index'], axis=1)
df_percentage['Percentage of tide'] = list_percentage
df_percentage['Tide'] = list_tide
df_percentage = df_percentage.set_index('Stations')

outfile = rep + 'Survey_' + month + '/' + 'percentage_tide_basedondata_v2' + month + '.xlsx'
df_percentage.to_excel(outfile, header=True)

sys.exit(1)

############ 22/11 je teste une méthode avec les min et max déja detéectés.
list_percentage, list_tide = [], []
# Objectif 1 : obtain time and amplitude of previous and next HT and LT
count = 0
for sta in df_month['Stations']:
    print('count', count)
    print('sta', sta)
    df_sta = df_month.loc[df_month['Stations'] == sta].copy()  # df of one line with the Date , Time and station name
    # Je cherche ou j'ai la même date
    df_good_time = interpolated_series.loc[interpolated_series['']]

##############################
list_percentage, list_tide = [], []
# Objectif 1 : obtain time and amplitude of previous and next HT and LT
count = 0
for sta in df_month['Stations']:
    print('count', count)
    print('sta', sta)

    # Find the datetime of the station
    df_sta = df_month.loc[df_month['Stations'] == sta].copy()  # df of one line with the Date , Time and station name
    # Find corresponding time in DoSon file
    date_sta = df_sta['Date'][df_sta['Date'].index[0]].strftime(
        '%Y-%m-%d')  # transform date to string to compare with DoSon
    df_DoSon = data_DoSon.loc[data_DoSon['Date'] == date_sta]
    # Find the right time
    hour_sta = df_sta['Time'].item().hour
    mn_sta = df_sta['Time'].item().minute
    df2_DoSon = data_DoSon[df_DoSon.index[0]:df_DoSon.index[0] + 25]  # need to have +25 in case the hour is 23h
    # so that i+1 corresponds to sthg in df2_DoSon
    # df2 is the df of the whole day
    # I check inside what is the corresponding time
    for i in range(24):
        d = df2_DoSon['hour'].iloc[i].hour
        print('i', i, d, df_sta['Time'].item().hour)
        if d == df_sta['Time'].item().hour:
            print('It is the right time')
            break
    # The good time and best tide correponding is :
    df_goodtime = df2_DoSon.iloc[i]
    tide = df_goodtime['Tide']
    # Puis recalcul de lheure et de la hauteur d'eau ? Si 15:30, moitié entre 15 et 16
    tide_plus1 = df2_DoSon['Tide'].iloc[i + 1]
    tide_recalculate = tide + mn_sta / 60 * (tide_plus1 - tide)

    # which period are we ? Flood or EBB ?
    # Finding min and max period during the day before and day after
    period_DoSon = data_DoSon[df_goodtime.name - 14:df_goodtime.name + 14]  # period tide around 12h, more to be sure
    if tide_plus1 > tide:
        print("Flood tide")
        # FInd the HT in the next index
        HT = period_DoSon.loc[period_DoSon['Tide'].loc[df_goodtime.name:].idxmax()]
        LT = period_DoSon.loc[period_DoSon['Tide'].loc[:df_goodtime.name].idxmin()]
        # period_DoSon.loc[:df_goodtime.name].idxmin()[2] # Give the min index on the right slice of the df
        time_phase = HT['hour'] - LT['hour']
        time_sta = df_sta['Time'] - LT['hour']
        time_sta = time_sta[df_sta.index[0]]
        percentage_tide = time_sta / time_phase * 100
    elif tide_plus1 < tide:
        print("Ebb tide")
        HT = period_DoSon.loc[period_DoSon['Tide'].loc[:df_goodtime.name].idxmax()]
        LT = period_DoSon.loc[period_DoSon['Tide'].loc[df_goodtime.name:].idxmin()]
        time_phase = LT['hour'] - HT['hour']
        time_sta = df_sta['Time'] - HT['hour']
        time_sta = time_sta[df_sta.index[0]]
        percentage_tide = -(100 - (time_sta / time_phase * 100))
    else:
        print("Not clear, we are around slack water, perhaps closer to Neap T ? ")  # Nothing in August
        # Best solution (and probably the only correct : detext the first max and min around the sta.
        HT = period_DoSon.loc[period_DoSon['Tide'].idxmax()]
        LT = period_DoSon.loc[period_DoSon['Tide'].idxmin()]
    print(percentage_tide)
    list_percentage.append(percentage_tide)
    list_tide.append(tide_recalculate)
    # Now that I have LT and HT, I can calculate IN TIME the % of tide
    count = count + 1

df_percentage = pd.DataFrame(df_month['Stations'])
df_percentage = df_percentage.reset_index()
df_percentage = df_percentage.drop(['index'], axis=1)
df_percentage['Percentage of tide'] = list_percentage
df_percentage['Tide'] = list_tide
df_percentage = df_percentage.set_index('Stations')

outfile = rep + 'Survey_' + month + '/' + 'percentage_tide_basedondata' + month + '.xlsx'
df_percentage.to_excel(outfile, header=True)

sys.exit(1)
# Old version before may 2023:
month = '06'  # TO CHANGE
if month == '06':
    hour_ht = 18  # hour high tide
    file = "/home/penicaud/Documents/Data/Survey_16-21june2022/16-18june/Stations_june_v2.xlsx"
    skiprows = 0
    nrows = 89
    sheetname = 'All'
    case_floodtide_atnight = False  # Flood tide is or not during night

elif month == '08':
    hour_ht = 15
    file = "/home/penicaud/Documents/Data/survey_august/Stations_10-13_august.xlsx"
    skiprows = 2
    nrows = 86
    sheetname = 'A1-AF38'
    case_floodtide_atnight = False  # Flood tide is or not during night

elif month == '10':
    file = "/home/penicaud/Documents/Data/survey_october/Stations_2-5_octobre.xlsx"
    # hour high tide depends on day, see in the loop
    skiprow = 1
    nrows = 98
    sheetname = 'O1-FO52'
    case_floodtide_atnight = True  # Flood tide is during night

print('file', file)
col_list = ["Stations", "Time"]  # "Station N", "Station E", "Time"]
dtypes = {'Stations': 'str', 'Time': 'str'}  # 'Station N': 'str', 'Station E': 'str',
parse_dates = ['Time']

data = pd.read_excel(file, nrows=nrows, usecols=col_list, sheet_name=sheetname, parse_dates=parse_dates)  # sep=' ',
df = pd.DataFrame(data)
print('df', df)
# <class 'pandas._libs.tslibs.timestamps.Timestamp'>
# for i in df['Time'] :
#    df['Time'][i].to_pydatetime()

print('type time', type(df['Time']), df['Time'])
print('type time(0)', type(df['Time'][0]))
# time=pd.read_excel(file, sep='', usecols='Time')
time = df['Time'].tolist()  # time = df['Time'].values.tolist()

print('time', time)
print(type(time[0]))

print('day', time[0].day)

# sys.exit(1)

percentage = []
for i in range(len(time)):
    h = time[i].hour
    mn = time[i].minute
    print('time seen', h, mn)
    day = time[i].day

    if month == '06':
        print("day", day)
        if day == 16 and (h > 5 and h < 17 or (h == 17 and mn == 0)):
            print('case 1, flood tide')
            hour_lt = 5
            hour_ht = 17
        elif (day == 16 and h >= 17) or (day == 17 and (h < 6 or (h == 6 and mn == 0))):
            print('case 2, ebb tide')
            hour_lt = 6
            hour_ht = 17
        elif day == 17 and (h >= 6 and h < 18 or (h == 18 and mn == 0)):
            print('case 3, flood tide')
            hour_lt = 6
            hour_ht = 18
        elif (day == 17 and h >= 18) or (day == 18 and (h < 7 or (h == 17 and mn == 0))):
            print('case 4, ebb tide')
            hour_lt = 7
            hour_ht = 18
        elif day == 18 and (h >= 7 and h < 19 or (h == 19 and mn == 0)):
            print('case 5, flood tide')
            hour_lt = 7
            hour_ht = 19
        elif day == 18 and h >= 19:
            print('case 6, ebb tide')
            hour_lt = 8
            hour_ht = 19
        else:
            print('PB DAY')

    if month == '08':
        print("day", day)
        if day == 10 and (h < 15 or (h == 15 and mn == 0)):  # flood tide
            print('case 1, flood tide')
            hour_lt = 2
            hour_ht = 15
        elif (day == 10 and h >= 15) or (day == 11 and (h < 3 or (h == 3 and mn == 0))):  # ebb tide
            print('case 2, ebb tide')
            hour_lt = 3  # day after
            hour_ht = 15
        elif day == 11 and ((h >= 3 and h < 16) or (h == 16 and mn == 0)):  # flood tide
            print('case 3, flood tide')
            hour_lt = 3
            hour_ht = 16
        elif (day == 11 and h >= 16) or (day == 12 and h < 5 or (h == 5 and mn == 0)):  # ebb tide
            print('case 4, ebb tide')
            hour_lt = 5  # day after
            hour_ht = 16
        elif day == 12 and (h >= 5 and h < 17 or (h == 17 and mn == 0)):  # flood tide
            print('case 5, flood tide')
            hour_lt = 5
            hour_ht = 17
        elif (day == 12 and h >= 17) or (day == 13 and (h < 6 or (h == 6 and mn == 0))):  # ebb tide
            print('case 6, ebb tide')
            hour_lt = 6
            hour_ht = 17
        elif (day == 13 and h > 6):  # flood tide
            print('case 7, flood tide')
            hour_lt = 6
            hour_ht = 17
        else:
            print('PB DAY')

    if month == '10':  # not spring tide, hour of HT and LT changes day by day. Need to be specify
        # IDEA : new variable "plage" which will be the time taken by the flood tide. Can be different of 12h
        daytrick = day
        if day == 2:
            # WARNING : if hour_ht < hour_lt, need to fill the hour LT of the previous day, because we focus on the "plage" of FLOOD TIDE
            hour_ht = 8
            hour_lt = 20  # day before
        if day == 3:
            hour_ht = 9
            hour_lt = 21  # day before
        if day == 4:
            hour_ht = 10
            hour_lt = 22  # day before
        if day == 5:
            hour_ht = 12
            hour_lt = 23  # day before
        else:
            print('PB DAY')

    # Detect if we are in flood or ebb tide
    # first need : detect if we are on 2 days different
    if case_floodtide_atnight:
        hour_lt2 = hour_lt - 24  # change to negative number to know if it is flood tide
    else:
        hour_lt2 = hour_lt

    if (h > hour_lt2 or (h == hour_lt2 and mn > 0)) and (h < hour_ht or (h == hour_ht and mn == 0)):
        print('Hour is in the flood tide')
        hour_floodtide = True
    else:
        print('Hour is in the ebb tide')
        hour_floodtide = False
        print('change in the hour of the LT from', hour_lt)
        daytrick = day + 1  # shift the day to take the hour lt of the day +1 because we are in case_floodtide_atnight.

        # if month=='08': #give the low tide of the day after
        #     if daytrick == 11:
        #         hour_lt = 3
        #     if daytrick == 12:
        #         hour_lt =  5
        #     if daytrick == 13:
        #         hour_lt = 6
        #     if daytrick == 14:
        #         hour_lt =  7

        if month == '10':
            if daytrick == 3:
                hour_lt = 21  # day before
            if daytrick == 4:
                hour_lt = 22  # day before
            if daytrick == 5:
                hour_lt = 23  # day before
    print("hour lt ", hour_lt, 'hour_ht', hour_ht)
    if hour_floodtide:
        hour1 = hour_ht
        hour2 = hour_lt
    elif not hour_floodtide:
        hour1 = hour_lt
        hour2 = hour_ht
    plage = (hour1 - hour2) % 24
    print("hour 1 hour 2 = plage ", hour1, hour2, plage)
    # if hour1<hour2 :
    #     plage=24+hour1-hour2
    # else :
    #     plage=hour1-hour2

    # if hour_ht<hour_lt: #means that low tide filled is the one of the previous day, to have the flood interval.
    #     plage=24+hour_ht-hour_lt
    # else :
    #     plage=hour_ht-hour_lt

    # #TODO : change
    # else :
    #     plage=12
    #     hour_lt=hour_ht-plage

    # plage=abs(hour_ht-hour_lt)
    print('DAY : ', day, '/', month, 'hour of HT :', hour_ht, 'Hour of LT :', hour_lt)
    if h >= hour_lt2 and h < hour_ht:
        print('CASE pos, flood tide')
        mn2 = mn / 60  # convert from 0to60 to 0to100
        t2 = (plage - (hour_ht - (h + mn2))) / plage * 100
        # t2= (12-(hour_ht-(h+mn2)))/12 * 100
        # t2 = (h + mn2 - 6) / 12 * 100  # HT=100%, LT=0%
        t2 = round(t2, 2)
    elif h == hour_ht and mn == 0:  # supposition : HT is exactly at mn=0
        print('CASE HT')
        t2 = 100.00
    elif h == hour_lt and mn == 0:
        print('CASE LT')
        t2 = 0.00
    elif h >= hour_ht or h < hour_lt2:
        print('CASE Neg, ebb tide')
        if h <= 23 and h >= hour_ht:  # TODO : si hour_ht=23?
            print('NEG 1')
            mn2 = mn / 60
            t2 = -(100 - (h + mn2 - hour_ht) / plage * 100)  # Add the 100-expr to have the convention 1
            t2 = round(t2, 2)
        elif h <= hour_lt:
            print('special t', h)
            mn2 = mn / 60
            t2 = -(100 - (24 + h + mn2 - hour_ht) / plage * 100)
            # t2 = "{:.2f}".format(t2)
            t2 = round(t2, 2)
            # convention: negative, LT--> -100% but LT and HT are calculated in the positive convention
        else:
            print('HORS SERIE')
    else:
        print('PROBLEM to define t')

    print('t2', t2, '%')
    percentage.append(t2)

# print('percentage = ', percentage)
file = 'percentage_tide_' + month + '_vtest.csv'
percentage2 = pd.DataFrame(percentage)
percentage2.to_csv(file)

print(percentage2)

sys.exit(1)
#####################
# VERSION BEFORE 3march2022

# 22/07. Objectif : calculer pourcentage de marée
import numpy as np
import csv
import sys
import pandas
import pandas as pd

# TODO : script done for hour_ht > 12, adpat it to all values of hour_ht

# COnvention : 100% is HT, 0% is LT, + from L to HT and - from H to LT.
# convention 1 : From HT to LT, we begin by -99% near to HT, -1% near to LT.  CIRCLE
# Convention 2 : Begin by -1% next to HT to arrive to -99% newt to LT


month = '10'  # TO CHANGE
if month == '06':
    hour_ht = 18  # hour high tide
    file = "/home/penicaud/Documents/Data/Survey_16-21june2022/16-18june/Stations_june_v2.xlsx"
    skiprows = 0
    nrows = 47
    sheetname = 'S1-S39'
elif month == '08':
    hour_ht = 15
    file = "/home/penicaud/Documents/Data/survey_august/Stations_10-13_august.xlsx"
    skiprows = 2
    nrows = 86
    sheetname = 'A1-AF38'
elif month == '10':
    file = "/home/penicaud/Documents/Data/survey_october/Stations_2-5_octobre.xlsx"
    # hour high tide depends on day, see in the loop
    skiprow = 1
    nrows = 98
    sheetname = 'O1-FO52'

print('file', file)
col_list = ["Stations", "Time"]  # "Station N", "Station E", "Time"]
dtypes = {'Stations': 'str', 'Time': 'str'}  # 'Station N': 'str', 'Station E': 'str',
parse_dates = ['Time']

data = pd.read_excel(file, sep=' ', nrows=nrows, usecols=col_list, sheet_name=sheetname, parse_dates=parse_dates)
df = pd.DataFrame(data)
print('df', df)
# <class 'pandas._libs.tslibs.timestamps.Timestamp'>
# for i in df['Time'] :
#    df['Time'][i].to_pydatetime()

print('type time', type(df['Time']), df['Time'])
print('type time(0)', type(df['Time'][0]))
# time=pd.read_excel(file, sep='', usecols='Time')
time = df['Time'].tolist()  # time = df['Time'].values.tolist()

print('time', time)
print(type(time[0]))

print('day', time[0].day)

# sys.exit(1)

percentage = []
for i in range(len(time)):
    h = time[i].hour
    mn = time[i].minute
    print('time seen', h, mn)
    day = time[i].day
    if month == '10':  # not spring tide, hour of HT and LT changes day by day. Need to be specify
        # IDEA : new variable "plage" which will be the time taken by the flood tide. Can be different of 12h
        if day == 2:
            # WARNING : if hour_ht < hour_lt, need to fill the hour LT of the previous day, because we focus on the "plage" of FLOOD TIDE
            hour_ht = 8
            hour_lt = 20  # day before
        elif day == 3:
            hour_ht = 9
            hour_lt = 21  # day before
        elif day == 4:
            hour_ht = 10
            hour_lt = 22  # day before
        elif day == 5:
            hour_ht = 12
            hour_lt = 23  # day before
        if hour_ht < hour_lt:
            plage = 24 + hour_ht - hour_lt
        else:
            plage = hour_ht - hour_lt
    else:
        plage = 12
        hour_lt = hour_ht - plage

        # plage=abs(hour_ht-hour_lt)
    print('DAY : ', day, '/', month, 'hour of HT :', hour_ht, 'Hour of LT :', hour_lt)
    if h >= hour_lt and h < hour_ht:
        print('CASE pos, flood tide')
        mn2 = mn / 60  # convert from 0to60 to 0to100
        t2 = (plage - (hour_ht - (h + mn2))) / plage * 100
        # t2= (12-(hour_ht-(h+mn2)))/12 * 100
        # t2 = (h + mn2 - 6) / 12 * 100  # HT=100%, LT=0%
        t2 = round(t2, 2)
    elif h == hour_ht and mn == 0:  # supposition : HT is exactly at mn=0
        print('CASE HT')
        t2 = 100.00
    elif h == hour_lt and mn == 0:
        print('CASE LT')
        t2 = 0.00
    elif h >= hour_ht or h < hour_lt:
        print('CASE Neg, ebb tide')
        if h <= 23 and h >= hour_ht:  # TODO : si hour_ht=23?
            print('NEG 1')
            mn2 = mn / 60
            t2 = -(100 - (h + mn2 - hour_ht) / plage * 100)  # Add the 100-expr to have the convention 1
            t2 = round(t2, 2)
        elif h <= hour_lt:
            print('special t', h)
            mn2 = mn / 60
            t2 = -(100 - (24 + h + mn2 - hour_ht) / plage * 100)
            # t2 = "{:.2f}".format(t2)
            t2 = round(t2, 2)
            # convention: negative, LT--> -100% but LT and HT are calculated in the positive convention
        else:
            print('HORS SERIE')
    else:
        print('PROBLEM to define t')

    print('t2', t2, '%')
    percentage.append(t2)

# print('percentage = ', percentage)
file = 'percentage_tide_' + month + '.csv'
percentage2 = pd.DataFrame(percentage)
percentage2.to_csv(file)

print(percentage2)

sys.exit(1)

#########################""

time1 = pandas.DataFrame({'stations': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S10', 'S12', 'S13',
                                       'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24',
                                       'S25',
                                       'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36',
                                       'S37',
                                       'S38', 'S39', ],
                          'Time': [13.08, 13.32, 13.56, 14.23, 14.44, 15.19, 15.42, 16.02, 16.22, 16.35, 17.05, 18.0,
                                   18.28,
                                   18.52, 19.2, 19.35, 19.48, 12.16, 12.32, 12.4, 13, 13.11, 13.23, 13.4, 13.56, 14.14,
                                   14.31, 14.48, 20.31, 20.47, 21.05, 21.2, 21.34, 21.42, 21.52, 22, 22.22, 22.35,
                                   22.58]})

# Time of the Station S1 to S39
time = [13.08, 13.32, 13.56, 14.23, 14.44, 15.19, 15.42, 16.02, 16.22, 16.35, 17.05, 18.0, 18.28,
        18.52, 19.2, 19.35, 19.48, 12.16, 12.32, 12.4, 13, 13.11, 13.23, 13.4, 13.56, 14.14,
        14.31, 14.48, 20.31, 20.47, 21.05, 21.2, 21.34, 21.42, 21.52, 22, 22.22, 22.35,
        22.58]  # Attention, time in minutes (60) ==> 100

time = [13.25, 13.43, 14, 14.15, 14.3, 14.4, 14.5, 15, 15.1, 15.2, 15.3, 15.4, 15.5, 16, 16.1, 16.2, 16.3, 16.4, 16.5,
        17, 17.1, 17.2, 17.3,
        17.4, 17.5, 17.55, 18.04, 18.12, 18.17, 18.24, 18.33, 18.4, 18.45, 18.55, 19.05, 19.2, 19.32, 19.42]

print(type(time))
percentage = []
for t in time:
    print('t', t)
    if t >= 6 and t < 18:
        mn = t - int(t)  # only mn part left
        mn2 = mn * 100 / 60
        t2 = (int(t) + mn2 - 6) / 12 * 100  # HT=100%, LT=0%
        # t2="{:.2f}".format(t2)
        t2 = round(t2, 2)
    elif t == 18.0:
        t2 = 100.00
    elif t == 6.0:
        t2 = 0.00
    elif t >= 18 or t < 6:
        if t <= 23 and t > 18:
            mn = t - int(t)  # only mn part left
            mn2 = mn * 100 / 60
            t2 = -(int(t) + mn2 - 18) / 12 * 100
            t2 = round(t2, 2)

        else:
            print('special t', t)
            mn = t - int(t)  # only mn part left
            mn2 = mn * 100 / 60
            t2 = -(24 + int(t) + mn2 - 18) / 12 * 100
            # t2 = "{:.2f}".format(t2)
            t2 = round(t2, 2)

            # convention: negative, LT--> -100% but LT and HT are calculated in the positive convention
    else:
        print('PROBLEM to define t')

    print('t2', t2, '%')
    percentage.append(t2)

# print('percentage = ', percentage)
file = 'percentage_tide_1806.csv'
percentage2 = pd.DataFrame(percentage)
percentage2.to_csv(file)

print(percentage2)
