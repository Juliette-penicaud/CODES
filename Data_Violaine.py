# 29/01/24 : Exploiter les données de Violaine pour les comparer avec le modèle
# Il faudra le basculer sur CALMIP pour comparer tout ca :)

import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib as mpl
from openpyxl import load_workbook
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from scipy import stats
import scipy.signal as signal
import re
import gsw
import glob

list_aug_S1 = [pd.NaT, datetime(2017, 8, 26, 10, 8, 0), datetime(2017, 8, 26, 12, 8, 0), datetime(2017, 8, 26, 14, 9, 0),
               datetime(2017, 8, 26, 16, 11, 0), datetime(2017, 8, 26, 18, 8, 0), datetime(2017, 8, 26, 20, 16, 0),
               datetime(2017, 8, 26, 22, 16, 0), datetime(2017, 8, 27, 0, 8, 0), datetime(2017, 8, 27, 2, 11, 0),
               datetime(2017, 8, 27, 4, 11, 0), datetime(2017, 8, 27, 6, 13, 0)]
list_aug_S2 = [datetime(2017, 8, 27, 8, 32, 0), datetime(2017, 8, 27, 10, 8, 0), datetime(2017, 8, 27, 12, 5, 0),
               datetime(2017, 8, 27, 14, 2, 0), datetime(2017, 8, 27, 16, 6, 0), datetime(2017, 8, 27, 17, 57, 0),
               datetime(2017, 8, 27, 20, 5, 0), datetime(2017, 8, 27, 22, 13, 0), datetime(2017, 8, 28, 0, 16, 0),
               datetime(2017, 8, 28, 2, 22, 0), datetime(2017, 8, 28, 4, 10, 0), datetime(2017, 8, 28, 6, 9, 0)]
list_aug_S3 = [pd.NaT, datetime(2017, 8, 28, 10, 41, 0), datetime(2017, 8, 27, 12, 5, 0), datetime(2017, 8, 28, 14, 0, 0),
               datetime(2017, 8, 28, 16, 2, 0), datetime(2017, 8, 28, 18, 3, 0), datetime(2017, 8, 28, 20, 4, 0),
               datetime(2017, 8, 28, 22, 6, 0), datetime(2017, 8, 29, 0, 5, 0), datetime(2017, 8, 29, 2, 6, 0),
               datetime(2017, 8, 29, 4, 5, 0), datetime(2017, 8, 29, 6, 1, 0)]
list_sept_S1 = [datetime(2017, 9, 3, 8, 16, 0), datetime(2017, 9, 3, 10, 10, 0), datetime(2017, 9, 3, 12, 4, 0),
                datetime(2017, 9, 3, 14, 10, 0), datetime(2017, 9, 3, 16, 9, 0), datetime(2017, 9, 3, 18, 5, 0),
                datetime(2017, 9, 3, 20, 11, 0), datetime(2017, 9, 3, 22, 14, 0), datetime(2017, 9, 4, 0, 19, 0),
                datetime(2017, 9, 4, 2, 12, 0), datetime(2017, 9, 4, 4, 7, 0), datetime(2017, 9, 4, 6, 6, 0)]
list_sept_S2 = [datetime(2017, 9, 4, 8, 3, 0), datetime(2017, 9, 4, 10, 2, 0), datetime(2017, 9, 4, 11, 59, 0),
                datetime(2017, 9, 4, 14, 2, 0), datetime(2017, 9, 4, 16, 4, 0), datetime(2017, 9, 4, 18, 1, 0),
                datetime(2017, 9, 4, 20, 4, 0), datetime(2017, 9, 4, 22, 14, 0), datetime(2017, 9, 5, 0, 6, 0),
                datetime(2017, 9, 5, 2, 6, 0), datetime(2017, 9, 5, 4, 7, 0), datetime(2017, 9, 5, 6, 12, 0)]
list_sept_S3 = [datetime(2017, 9, 5, 8, 6, 0), datetime(2017, 9, 5, 10, 0, 0), datetime(2017, 9, 5, 12, 0, 0),
                datetime(2017, 9, 5, 13, 56, 0), datetime(2017, 9, 5, 16, 2, 0), datetime(2017, 9, 5, 17, 59, 0),
                datetime(2017, 9, 5, 19, 56, 0), datetime(2017, 9, 5, 22, 1, 0), datetime(2017, 9, 6, 0, 2, 0),
                datetime(2017, 9, 6, 2, 6, 0), datetime(2017, 9, 6, 4, 2, 0), datetime(2017, 9, 6, 6, 5, 0)]
list_decs_S1 = [datetime(2017, 12, 6, 9, 6, 0), datetime(2017, 12, 6, 10, 7, 0), datetime(2017, 12, 6, 12, 7, 0),
                datetime(2017, 12, 6, 14, 12, 0), datetime(2017, 12, 6, 16, 12, 0), datetime(2017, 12, 6, 18, 9, 0),
                datetime(2017, 12, 6, 20, 9, 0), datetime(2017, 12, 6, 22, 8, 0), datetime(2017, 12, 7, 0, 5, 0),
                datetime(2017, 12, 7, 2, 12, 0), datetime(2017, 12, 7, 4, 12, 0), datetime(2017, 12, 7, 6, 10, 0)]
list_decs_S2 = [datetime(2017, 12, 7, 8, 57, 0), datetime(2017, 12, 7, 10, 6, 0), datetime(2017, 12, 7, 12, 3, 0),
                datetime(2017, 12, 7, 14, 7, 0), datetime(2017, 12, 7, 16, 10, 0), datetime(2017, 12, 7, 18, 12, 0),
                datetime(2017, 12, 7, 20, 5, 0), datetime(2017, 12, 7, 22, 4, 0), datetime(2017, 12, 8, 0, 4, 0),
                datetime(2017, 12, 8, 2, 6, 0), datetime(2017, 12, 8, 4, 14, 0), datetime(2017, 12, 8, 6, 11, 0)]
list_decs_S3 = [datetime(2017, 12, 8, 8, 6, 0), datetime(2017, 12, 8, 10, 4, 0), datetime(2017, 12, 8, 12, 9, 0),
                datetime(2017, 12, 8, 14, 0, 0), datetime(2017, 12, 8, 16, 7, 0), datetime(2017, 12, 8, 18, 6, 0),
                datetime(2017, 12, 8, 20, 3, 0), datetime(2017, 12, 8, 22, 4, 0), datetime(2017, 12, 9, 0, 1, 0),
                datetime(2017, 12, 9, 2, 6, 0), datetime(2017, 12, 9, 4, 3, 0), datetime(2017, 12, 9, 6, 5, 0)]
list_decn_S1 = [datetime(2017, 12, 12, 8, 55, 0), datetime(2017, 12, 12, 10, 8, 0), datetime(2017, 12, 12, 12, 8, 0),
                datetime(2017, 12, 12, 14, 14, 0), datetime(2017, 12, 12, 16, 9, 0), datetime(2017, 12, 12, 18, 11, 0),
                datetime(2017, 12, 12, 20, 4, 0), datetime(2017, 12, 12, 22, 5, 0), datetime(2017, 12, 13, 0, 3, 0),
                datetime(2017, 12, 13, 2, 4, 0), datetime(2017, 12, 13, 4, 4, 0), datetime(2017, 12, 13, 6, 4, 0)]
list_decn_S2 = [datetime(2017, 12, 13, 8, 5, 0), datetime(2017, 12, 13, 10, 4, 0), datetime(2017, 12, 13, 12, 2, 0),
                datetime(2017, 12, 13, 14, 6, 0), datetime(2017, 12, 13, 16, 4, 0), datetime(2017, 12, 13, 18, 2, 0),
                datetime(2017, 12, 13, 20, 9, 0), datetime(2017, 12, 13, 22, 7, 0), datetime(2017, 12, 14, 0, 4, 0),
                datetime(2017, 12, 14, 2, 5, 0), datetime(2017, 12, 14, 4, 6, 0), datetime(2017, 12, 14, 6, 7, 0)]
list_decn_S3 = [datetime(2017, 12, 14, 8, 2, 0), datetime(2017, 12, 14, 10, 3, 0), datetime(2017, 12, 14, 11, 57, 0),
                datetime(2017, 12, 14, 14, 2, 0), datetime(2017, 12, 14, 16, 3, 0), datetime(2017, 12, 14, 18, 1, 0),
                datetime(2017, 12, 14, 20, 3, 0), datetime(2017, 12, 14, 22, 1, 0), datetime(2017, 12, 15, 0, 1, 0),
                datetime(2017, 12, 15, 2, 4, 0), datetime(2017, 12, 15, 4, 6, 0), datetime(2017, 12, 15, 6, 3, 0)]


# VARIABLE AND PARAMETER
year = '2022'
list_month = ['AUG', 'SEPT', 'DECS', 'DECN']
list_survey = ['WN', 'WS', 'DS', 'DN']
i = 0  # 0 1 2 3
survey = list_survey[i]
month_survey = list_month[i]
precise_hour = {'AUG': {'S1': list_aug_S1, 'S2': list_aug_S2, 'S3': list_aug_S3},
                'SEPT': {'S1': list_sept_S1, 'S2': list_sept_S2, 'S3': list_sept_S3},
                'DECS': {'S1': list_decs_S1, 'S2': list_decs_S2, 'S3': list_decs_S3},
                'DECN': {'S1': list_decn_S1, 'S2': list_decn_S2, 'S3': list_decn_S3}}
stations = ['S1', 'S2', 'S3']
rep = '/home/penicaud/Documents/Data/Survey_Violaine/'
# Rep Calmip :
rep = '/users/p13120/penicaud/DATA/Survey_Violaine/'

#################################################      CTD       ######################################################
columns_CTD = ["Depth", "Temperature", "Salinity", "Turbidity"]
units_CTD = ['(m)', '(°C)', '(PSU)', '(FTU)']
# CTD file
rep_CTD = rep + 'CTD/' + month_survey + '/'

hours = np.arange(2, 14, 2)
#moments = []
#for am_pm in ['am', 'pm']:
#    for hour in hours:
#        moments.append(str(hour) + am_pm)
moments = ['8am','10am', '12am', '2pm', '4pm', '6pm', '8pm', '10pm', '12pm', '2am','4am', '6am']
counts = np.arange(0,12)
dict_moment = {key: value for key, value in zip(counts, moments)}
test = True
if test :
    stations = ['S1']
    dict_moment = {7:'10pm', 8:'12pm'}
print(dict_moment)
len(dict_moment)
for i in range(len(dict_moment)):
    print(i)

for sta in stations:
    print('Station = ', sta)
    if sta == 'S1':
        station_name = 'ST1'
    elif sta == 'S2':
        station_name = 'ST2'
    elif sta == 'S3':
        station_name = 'ST3'
    # for moment in moments :
    for i in range(7, 9):  # range(len(dict_moment)):
        moment = dict_moment[i]
        print(moment)
        file_CTD = rep_CTD + month_survey + '_' + sta + '_' + moment + '.txt'
        print('file CTD : ', file_CTD)
        # Test pour savoir si fichier existe :
        if not os.path.exists(file_CTD):
            print(f"The file '{file_CTD}' does not exist in the directory.")

        else:
            print(f"The file '{file_CTD}' exists in the directory.")
            df_sta = pd.read_csv(file_CTD, sep='\t', header=None)  # , usecols=columns_CTD)
            df_sta.columns = [col for col in columns_CTD]
            # print(df_sta)

            # Here beggins the modelisation seeking ..
            rep_mod = '/tmpdir/penicaud/SYMPHONIE_v367_VANUC/VANUC_MUSTANG/GRAPHIQUES/'
            rep_sorties_concat = '/tmpdir/penicaud/sorties_nc/'
            sim = '14mois_2017_V1_1class'
            name_sim_extracted = 'sedextract_'  # + station_name + '_'

            # Je cherche la sortie la plus proche de l'heure réelle
            hour_obs = precise_hour[month_survey][sta][i]
            print(hour_obs)
            year = hour_obs.year
            month = hour_obs.month
            if month < 10:
                month_char = '0' + str(month)
            else:
                month_char = str(month)
            # Je créé une sécurité en faisant une recherche sur les fichiers de h-1 à h+1
            datetime_0 = (hour_obs - timedelta(hours=1))
            day_0 = datetime_0.day
            hour_0 = datetime_0.hour

            day_1 = hour_obs.day
            hour_1 = hour_obs.hour

            datetime_2 = (hour_obs + timedelta(hours=1))
            day_2 = datetime_2.day
            hour_2 = datetime_2.hour

            print('days : ', day_0, day_1, day_2)
            print('hours : ', hour_0, hour_1, hour_2)

            f_closest_time = []
            list_datetime_mod = []
            if day_0 == day_2:
                print('Same day0 and day 2')
                if day_0 < 10:
                    day_char = '0' + str(day_0)
                else:
                    day_char = str(day_0)
                for h in range(hour_0, hour_2 + 1):
                    print('hour = ', h)
                    if h < 10:
                        hour_char = '0' + str(h)
                    else:
                        hour_char = str(h)
                    name_file = rep_sorties_concat + sim + '/' + name_sim_extracted + str(
                        year) + month_char + day_char + '_' + hour_char + '*'
                    f_closest_time.append(glob.glob(name_file))
            else:
                print('Day0 and Day2 aredifferent ')
                if day_0 < 10:
                    day_char = '0' + str(day_0)
                else:
                    day_char = str(day_0)
                if hour_0 < 10:
                    hour_char = '0' + str(hour_0)
                else:
                    hour_char = str(hour_0)
                name_file = rep_sorties_concat + sim + '/' + name_sim_extracted + str(
                    year) + month_char + day_char + '_' + hour_char + '*'
                print(name_file)
                f_closest_time.append(glob.glob(name_file))

                if day_1 < 10:
                    day_char = '0' + str(day_1)
                else:
                    day_char = str(day_1)
                if hour_1 < 10:
                    hour_char = '0' + str(hour_1)
                else:
                    hour_char = str(hour_1)
                name_file = rep_sorties_concat + sim + '/' + name_sim_extracted + str(
                    year) + month_char + day_char + '_' + hour_char + '*'
                f_closest_time.append(glob.glob(name_file))

                if day_2 < 10:
                    day_char = '0' + str(day_2)
                else:
                    day_char = str(day_2)
                if hour_2 < 10:
                    hour_char = '0' + str(hour_2)
                else:
                    hour_char = str(hour_2)
                name_file = rep_sorties_concat + sim + '/' + name_sim_extracted + str(
                    year) + month_char + day_char + '_' + hour_char + '*'
                f_closest_time.append(glob.glob(name_file))

            # Flatten the list to a 1d list:
            f_closest_time = [item for sublist in f_closest_time for item in sublist]
            print('lent files closest time : ', len(f_closest_time))

            # select the datetime
            ind1 = 61  # len(name_file) - 12
            ind2 = 75  # len(name_file) + 3
            list_datetime_mod = []
            for f in f_closest_time:
                date_string = f[ind1:ind2]
                # print(date_string)
                hour_mod = datetime.strptime(date_string, "%Y%m%d_%H%M%S")
                # print(hour_mod)
                list_datetime_mod.append(hour_mod)

            df_datetime = pd.DataFrame({'file': f_closest_time, 'time': list_datetime_mod})
            closest_datetime = min(df_datetime['time'], key=lambda x: abs(x - hour_obs))




###############################################      ADCP       #####################################################

# ADCP FILE
rep_adcp = rep + 'ADCP/' + month + '/'

for sta in stations:
    file_adcp = rep_adcp + sta + '/'
    station_file_adcp = glob.glob(file_adcp + '*')
    for f_adcp in station_file_adcp :
        try:
            with open(f_adcp, 'rb') as file:
                content = file.read().decode('utf-16-le', errors='replace')
            print(content)
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            with open(f_adcp, 'r', encoding='utf-16') as file:
                content = file.read()
            print(content)
        except Exception as e:
            print(f"An error occurred: {e}")









