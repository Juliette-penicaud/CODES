# 29/01/24 : Exploiter les données de Violaine pour les comparer avec le modèle
# Il faudra le basculer sur CALMIP pour comparer tout ca :)

import pandas as pd
import numpy as np
import sys, os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib as mpl
#from openpyxl import load_workbook
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from scipy import stats
import scipy.signal as signal
import re
import xarray as xr
#import gsw
import glob
from scipy.ndimage import median_filter

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
#rep = '/users/p13120/penicaud/DATA/Survey_Violaine/'



###################### MODELE
# Here beggins the modelisation seeking ..
comp_sim = False
if comp_sim :
    rep_mod = '/tmpdir/penicaud/SYMPHONIE_v367_VANUC/VANUC_MUSTANG/GRAPHIQUES/'
    rep_sorties_concat = '/tmpdir/penicaud/sorties_nc/'
    sim = '14mois_2017_V1_1class'
    name_sim_extracted = 'sedextract_' # + station_name + '_'
    sim_extract = False
    if sim_extract :
        deb_name = rep_sorties_concat + sim + '/' + name_sim_extracted
    else :
        deb_name = rep_mod + sim + '/'

    ds_grid = xr.open_dataset(rep_sorties_concat+'grid.nc')

columns_CTD = ["Depth", "Temperature", "Salinity", "Turbidity"]
unit = {'Depth': '(m)', 'Temperature': '(°C)', "Salinity": '(PSU)', "Turbidity": '(FTU)'}
# CTD file
rep_CTD = rep + 'CTD/' + month_survey + '/'
rep_ADCP = rep + 'ADCP/' + month_survey + '/'

hours = np.arange(2, 14, 2)
# moments = []
# for am_pm in ['am', 'pm']:
#    for hour in hours:
#        moments.append(str(hour) + am_pm)
moments = ['8am', '10am', '12am', '2pm', '4pm', '6pm', '8pm', '10pm', '12pm', '2am', '4am', '6am']
counts = np.arange(0, 12)
dict_moment = {key: value for key, value in zip(counts, moments)}
stations = ['S1', 'S2', 'S3']
loop = len(dict_moment)
test = True
if test:
    stations = ['S1']
    dict_moment = {7: '10pm', 8: '12pm'}  # Test pour fighier à cheval sur 2 jours
    loop = (7, 9)
print(dict_moment)
len(dict_moment)

localisation = {'S1': {'i': 61, 'j': 456, 'color': 'gray'},
                'S2': {'i': 36, 'j': 454, 'color': 'blue'},
                'S3': {'i': 15, 'j': 459, 'color': 'violet'}}

# Parameter plot
fontsize = 15
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 4
s = 25
rep_fig = '/users/p13120/penicaud/SCRIPTS_PYTHON/FIGURES/'


def merge_first_three_rows(column):
    column = [str(value) for value in column[:3]]
    return ' '.join(column[:3])

def recursively_delete_word(column, word):
    return column.str.replace(word, '')

def round_down_seconds(dt):
    return dt.replace(second=0, microsecond=0)


def median_filter_numeric_columns(column):
    if pd.api.types.is_numeric_dtype(column.dtype):
        return median_filter(column, size=5)
    return column

# 08/02/24 : je rajoute cette focntion issue de plot_data_recap_excefile_xarray :
# TODO : upload dans CALMIP
def velocity_and_direction(type_fixe, month, data_fixe, name_sta, unit):
    # Several cases : # 16-08-23 introduction of case1 or case2 to discriminate the subcases i.e C and D
    # where we are sure to be in + or - case
    # CASE1:
    # Case A : nor and east are + : direction NE, sens value depends on the value of the angle
    # Case B : nor and east are - : direction SO, sens value depends on the value of the angle
    # Case C : nor + and east -, i.e angle value -, direction NO, sens is -
    # Case D : nor - and east +, i.e angle value -, direction SE, sens is +
    # CASE2 :
    # Case A : nor + east - : direction NE, sens value depends on the value of the angle
    # Case B : nor - east + : direction SO, sens value depends on the value of the angle
    # Case C : nor and east -, i.e angle value -, direction NO, sens is -
    # Case D : nor and east +, i.e angle value -, direction SE, sens is +
    # dict_angle = {'section1': {'angle': 50, 'case': 'case1'},
    #               'section2': {'angle': 0, 'case': 'case1'},
    #               'section3': {'angle': 15, 'case': 'case2'},
    #               'section4': {'angle': 25, 'case': 'case1'},
    #               'section5': {'angle': 77, 'case': 'case1'},
    #               'section6': {'angle': 50, 'case': 'case1'},
    #               'section7': {'angle': 10, 'case': 'case1'},
    #               'section8': {'angle': 65, 'case': 'case2'},
    #               'section9': {'angle': 28, 'case': 'case1'},
    #               'section10': {'angle': 80, 'case': 'case1'},
    #               'section11': {'angle': 33, 'case': 'case1'},
    #               'section12': {'angle': 44, 'case': 'case1'}}
    dict_angle = {'section2': {'angle': 55, 'case': 'case1'},
                  'section3': {'angle': 27, 'case': 'case2'}, # 18
                  'section4': {'angle': 80, 'case': 'case1'},
                  'section5': {'angle': 35, 'case': 'case1'},
                  'section6': {'angle': 20, 'case': 'case1'},
                  'section7': {'angle': 52, 'case': 'case1'},
                  'section8': {'angle': 25, 'case': 'case2'},
                  'section9': {'angle': 82, 'case': 'case1'},
                  'section10': {'angle': 25, 'case': 'case1'},
                  'section11': {'angle': 45, 'case': 'case1'},
                  'section12': {'angle': 75, 'case': 'case1'},
                  'section13': {'angle': 0, 'case': 'case1'},
                  "section14": {'angle': 35, 'case': 'case1'} }

    dict_section = {'section2': ['A7', 'O24', 'S17', 'O10', 'O31', 'O35', 'A14'],
                    'section3': ['O34', 'A15', 'S1', 'S32', 'S36', 'S34', 'S35', 'S33', 'O32', 'O11', 'O23', 'A16',
                                 'S37', 'O12', 'A8', 'A17', 'S38', 'S31'],
                    'section4': ['A9', 'O33'],
                    'section5': [''],
                    'section6': ['O22', 'O13', 'O22-2', 'A19', 'O14', "A11", 'A12', 'S39', 'A10', 'A18'],
                    'section7': ['SF1', 'A20'],
                    'section8': ['A21', 'O15', 'O16', 'A22', 'O17', 'A23', 'OF1'],
                    'section9': ['O18', 'A24'],
                    'section10': ["O19", 'A25', 'O20'],
                    'section11': ['A26', 'O21', 'A27'],
                    "section12": ["O29", 'S9', 'S23', "S10", "S22"], # S24, O6, O7
                    "section13": ["A33", "A34", "S18", "S19", "S20", "S21"],
                    'section14': ['']  # if not in other section, then it is section12}
                    }
    if type_fixe == 'fixe':
        case = 'case1'
        angle_seuil = 50  # angles for the station fixe values
    elif type_fixe == 'small fixe' and month == 'Octobre':
        case = 'case2'
        angle_seuil = 65
    elif type_fixe == 'small fixe' and month == 'August':
        angle_seuil = 10
        case = 'case1'
    else:
        s = 2
        while s < 14:
            if name_sta in dict_section['section' + str(s)]:
                break
            else:
                s = s + 1
        section = 'section' + str(s)
        angle_seuil = dict_angle[section]['angle']
        case = dict_angle[section]['case']

    vnor = 'vitesse north'
    veast = 'vitesse east'
    if case == 'case1':
        condA = (data_fixe[vnor] > 0) & (data_fixe[veast] > 0)
        condB = (data_fixe[vnor] < 0) & (data_fixe[veast] < 0)
        condC = (data_fixe[vnor] > 0) & (data_fixe[veast] < 0)
        condD = (data_fixe[vnor] < 0) & (data_fixe[veast] > 0)
        cond1 = (data_fixe['angle'].abs() > angle_seuil)
        cond2 = (data_fixe['angle'].abs() <= angle_seuil)
    elif case == 'case2':
        condD = (data_fixe[vnor] > 0) & (data_fixe[veast] > 0)
        condC = (data_fixe[vnor] < 0) & (data_fixe[veast] < 0)
        condA = (data_fixe[vnor] > 0) & (data_fixe[veast] < 0)
        condB = (data_fixe[vnor] < 0) & (data_fixe[veast] > 0)
        cond2 = (data_fixe['angle'].abs() > angle_seuil)
        cond1 = (data_fixe['angle'].abs() <= angle_seuil)
    new_v = 'Velocity sens'
    data_fixe[new_v] = np.nan  # data_fixe['Vel_'+v].values.copy()

    # Vel = -
    data_fixe.loc[condA & cond1, new_v] = -data_fixe.loc[condA & cond1, 'module vitesse u']
    data_fixe.loc[condB & cond2, new_v] = -data_fixe.loc[condB & cond2, 'module vitesse u']
    data_fixe.loc[condC, new_v] = -data_fixe.loc[condC, 'module vitesse u']
    # Vel +
    data_fixe.loc[condA & cond2, new_v] = data_fixe.loc[condA & cond2, 'module vitesse u']
    data_fixe.loc[condB & cond1, new_v] = data_fixe.loc[condB & cond1, 'module vitesse u']
    data_fixe.loc[condD, new_v] = data_fixe.loc[condD, 'module vitesse u']
    if unit=='m/s':
        data_fixe[new_v] = data_fixe[new_v]/1000



for sta in stations:
    print('Station = ', sta)
    if sta == 'S1':
        station_name = 'ST1'
    elif sta == 'S2':
        station_name = 'ST2'
    elif sta == 'S3':
        station_name = 'ST3'
    # for moment in moments :
    for i in range(7, 9):  # range(7,9)
        moment = dict_moment[i]
        print(moment)
        file_CTD = rep_CTD + month_survey + '_' + sta + '_' + moment + '.txt'
        file_ADCP = rep_ADCP + month_survey + '_' + sta + '_' + moment + '.txt' # chemin valable pour calmip
        file_ADCP = rep_ADCP + sta + '/' +  month_survey + '_' + sta + '_' + moment + '.txt' # chemin en local
        # Test pour savoir si fichier existe :
        if not os.path.exists(file_CTD):
            print(f"The file CTD'{file_CTD}' does not exist in the directory.")
            CTD_data = False
        else:
            # print(f"The file '{file_CTD}' exists in the directory.")
            df_sta = pd.read_csv(file_CTD, sep='\t', header=None)  # , usecols=columns_CTD)
            df_sta.columns = [col for col in columns_CTD]
            CTD_data = True
            # print(df_sta)
        if not os.path.exists(file_ADCP):
            print(f"The file ADCP '{file_ADCP}' does not exist in the directory.")
            ADCP_data = False
        else:
            ADCP_data = True
            df_adcp = pd.read_csv(file_ADCP, sep='\t', encoding='latin-1', skiprows=12, header=None)
            col_adcp = df_adcp.apply(merge_first_three_rows)
            col_adcp = recursively_delete_word(col_adcp, 'nan')  # This part becomes the header.
            col_adcp = recursively_delete_word(col_adcp, '  ')  # This part becomes the header.
            df_adcp.columns = [col for col in col_adcp]
            df_adcp.drop(index=df_adcp.index[:3], inplace=True)
            df_adcp = df_adcp.loc[:, ~df_adcp.columns.duplicated(keep='first')]
            df_adcp = df_adcp.drop(columns='')
            df_adcp = df_adcp.replace(',', '.', regex=True)
            df_adcp = df_adcp.apply(pd.to_numeric, errors='coerce')
            # Je mets toutes les données ADCP en GMT
            df_adcp['YR'] = 2017
            df_adcp['Datetime local'] = pd.to_datetime(df_adcp[['YR', 'MO', 'DA', 'HH', 'MM', 'SS']].rename(
                columns={'YR': 'year', 'MO': 'month', 'DA': 'day', 'HH': 'hour', 'MM': 'minute', 'SS': 'second'}),
                                                       errors='coerce')
            df_adcp['Datetime gmt'] = df_adcp['Datetime local'] - timedelta(hours=7)
            depth_bin = 0.25
            nb_bin = 60
            blank_1st_bin = 0.6 # officiellement 0.57
            dmax = depth_bin * nb_bin + blank_1st_bin
            interval_seconds = 5
            dmax = depth_bin * nb_bin + blank_1st_bin

        if (ADCP_data or CTD_data):
            # Je cherche la sortie la plus proche de l'heure réelle
            hour_obs_local = precise_hour[month_survey][sta][i]
            hour_obs_gmt = hour_obs_local - timedelta(hours=7)
            year = hour_obs_gmt.year
            month = hour_obs_gmt.month
            if month < 10:
                month_char = '0' + str(month)
            else:
                month_char = str(month)
            # Je créé une sécurité en faisant une recherche sur les fichiers de h-1 à h+1
            datetime_0 = (hour_obs_gmt - timedelta(hours=1))
            day_0 = datetime_0.day
            hour_0 = datetime_0.hour

            day_1 = hour_obs_gmt.day
            hour_1 = hour_obs_gmt.hour

            datetime_2 = (hour_obs_gmt + timedelta(hours=1))
            day_2 = datetime_2.day
            hour_2 = datetime_2.hour
            # print('days : ', day_0, day_1, day_2)
            # print('hours : ', hour_0, hour_1, hour_2)

            f_closest_time = []
            list_datetime_mod = []
            if day_0 == day_2:
                print('Same day0 and day 2')
                if day_0 < 10:
                    day_char = '0' + str(day_0)
                else:
                    day_char = str(day_0)
                for h in range(hour_0, hour_2 + 1):
                    # print('hour = ', h)
                    if h < 10:
                        hour_char = '0' + str(h)
                    else:
                        hour_char = str(h)
                    name_file = deb_name + str(year) + month_char + day_char + '_' + hour_char + '*'
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
                name_file = deb_name + str(year) + month_char + day_char + '_' + hour_char + '*'
                f_closest_time.append(glob.glob(name_file))

                if day_1 < 10:
                    day_char = '0' + str(day_1)
                else:
                    day_char = str(day_1)
                if hour_1 < 10:
                    hour_char = '0' + str(hour_1)
                else:
                    hour_char = str(hour_1)
                name_file = deb_name + str(year) + month_char + day_char + '_' + hour_char + '*'
                f_closest_time.append(glob.glob(name_file))

                if day_2 < 10:
                    day_char = '0' + str(day_2)
                else:
                    day_char = str(day_2)
                if hour_2 < 10:
                    hour_char = '0' + str(hour_2)
                else:
                    hour_char = str(hour_2)
                name_file = deb_name + str(year) + month_char + day_char + '_' + hour_char + '*'
                f_closest_time.append(glob.glob(name_file))

            # Flatten the list to a 1d list:
            f_closest_time = [item for sublist in f_closest_time for item in sublist]
            # print('lent files closest time : ', len(f_closest_time))

            # select the datetime
            ind1 = 85  # 61 # len(name_file) - 12, 85 is ok for sim='14mois_...'
            list_datetime_mod = []
            for f in f_closest_time:
                date_string = f[ind1:ind1 + 15]
                # print(date_string)
                hour_mod = datetime.strptime(date_string, "%Y%m%d_%H%M%S")
                # print(hour_mod)
                list_datetime_mod.append(hour_mod)

            df_datetime = pd.DataFrame({'file': f_closest_time, 'time': list_datetime_mod})
            closest_datetime = min(df_datetime['time'], key=lambda x: abs(x - hour_obs_gmt))

            print('OBS : ', hour_obs_gmt)
            print('MOD :', closest_datetime)
            # I open the corresponding file
            file = df_datetime[df_datetime['time'] == closest_datetime]['file'].values[0]
            data_mod = xr.open_dataset(file)

            depth = ds_grid.depth_t.sel(ni_t=localisation[sta]['i'], nj_t=localisation[sta]['j'])
            depth = depth - depth[-1]  # recale  à 0 en enlevant la surface libre pour avoir une comp avec prof CTD

            figure_salinity = False
            if figure_salinity:
                var = 'Salinity'
                sal = data_mod.sal.sel(ni_t=localisation[sta]['i'], nj_t=localisation[sta]['j'])[0]
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('white')
                string_hour_obs_local = hour_obs_local.strftime("%d/%m/%Y %H:%M:%S")
                string_hour_obs_local_out = hour_obs_local.strftime("%Y%m%d_%H%M%S")
                fig.suptitle(sta + ' ' + string_hour_obs_local)
                ax.set_xlabel(var + ' ' + unit[var], fontsize=fontsize)
                ax.set_xlim(-1, 30)
                ax.set_ylabel('Depth (m)', fontsize=fontsize)
                ax.set_ylim(-12, 0)
                ax.plot(sal, depth, color=localisation[sta]['color'], label='Model')
                ax.plot(df_sta[var].values, -df_sta['Depth'].values, color=localisation[sta]['color'], ls='--',
                        label='Obs')
                ax.legend()
                outfile = rep_fig + 'Salinity_mod_obs_' + month_survey + '_' + sta + '_' + string_hour_obs_local_out + '.png'
                fig.savefig(outfile)


            # Traitement de données de la vitesse.
            # 1. Je dois selectionner le profil qui correspond à l'heure du modèle.
            closest_datetime = datetime(2017, 8, 26, 15, 12, 51) # Comme pas accès aux sorties modèle, je fixe
            rounded_datetime = closest_datetime.replace(second=0)  # Arrondi, pour ne pas prendre en compte les secondes
            df_adcp['Rounded Datetime gmt'] = df_adcp['Datetime gmt'].apply(round_down_seconds) # nouvelle colomne avec
            # toutes les datetime arrondies.
            condition = (df_adcp['Rounded Datetime gmt'] == rounded_datetime)
            # Filtre sur les données globales
            filter = True
            if filter : # TODO : a retravailler
                #data_vitesse = signal.medfilt2d(data_vitesse_nonfiltré.values, 5)
                df_adcp2 = df_adcp.apply(median_filter_numeric_columns, axis=0)
            else :
                df_adcp2 = df_adcp.copy()
            # 2. Je sélectionne ce profil :
            list_drop = ['Ens', 'YR', 'MO', 'DA', 'HH', 'MM','SS','Datetime gmt', 'Datetime local','Rounded Datetime gmt']
            data_vitesse = df_adcp2[condition].drop(columns=list_drop)

            mag, dir, eas, nor = [], [], [], []
            for i in range(1, nb_bin+1):
                string1 = "Mag mm/s " + str(i)
                string2 = "Dir deg " + str(i)
                string3 = "Eas mm/s " + str(i)
                string4 = "Nor mm/s " + str(i)
                mag.append(string1)
                dir.append(string2)
                eas.append(string3)
                nor.append(string4)

            # Création du df avec les valeurs moyennes sur une minutes (12 valeurs)
            data_vitesse_mean = pd.DataFrame()
            depth = np.arange(blank_1st_bin, dmax, depth_bin)
            data_vitesse_mean['Depth'] = depth
            data_vitesse_mean['Nor mm/s'] = data_vitesse[nor].mean().values
            data_vitesse_mean['Eas mm/s'] = data_vitesse[eas].mean().values
            data_vitesse_mean['Mag mm/s'] = data_vitesse[mag].mean().values
            data_vitesse_mean['Dir mm/s'] = data_vitesse[dir].mean().values

            # Ajout colonne du module de vitesse, grad vitesse u et angle
            data_vitesse_mean['module vitesse u'] = np.sqrt(np.square(data_vitesse_mean['Eas mm/s']) +
                                                            np.square(data_vitesse_mean['Nor mm/s']))
            data_vitesse_mean['grad vitesse horiz'] = np.gradient(data_vitesse_mean['module vitesse u'], depth_bin)
            data_vitesse_mean['angle'] = np.rad2deg(np.arctan(data_vitesse_mean['Nor mm/s'] /
                                                              data_vitesse_mean['Eas mm/s']))  # abs of nor and east ?

            # Sens de la vitesse :
            angle_seuil = {'S1': 45, 'S2': 68, 'S3': 55}
            # Several cases :
            # Case A : nor and east are + : direction NE, sens value depends on the value of the angle
            # Case B : nor and east are - : direction SO, sens value depends on the value of the angle
            # Case C : nor + and east -, i.e angle value -, direction NO, sens is -
            # Case D : nor - and east +, i.e angle value -, direction SE, sens is +
            condA = (data_vitesse_mean['Nor mm/s'] > 0) & (data_vitesse_mean['Eas mm/s'] > 0)
            condB = (data_vitesse_mean['Nor mm/s'] < 0) & (data_vitesse_mean['Eas mm/s'] < 0)
            condC = (data_vitesse_mean['Nor mm/s'] > 0) & (data_vitesse_mean['Eas mm/s'] < 0)
            condD = (data_vitesse_mean['Nor mm/s'] < 0) & (data_vitesse_mean['Eas mm/s'] > 0)
            cond1 = (angle_seuil[sta] < data_vitesse_mean['angle'].abs())
            cond2 = (angle_seuil[sta] >= data_vitesse_mean['angle'].abs())
            new_v = 'Sens velocity'
            # Il me faut continuer ca.. v étant c1 c2 ou mean, il faut que je sache à quoi cela correspond 
            data_vitesse_mean.loc[condA & cond1, new_v] = data_vitesse_mean.loc[condA & cond1, '']
            data_vitesse_mean.loc[condA & cond2, new_v] = -data_vitesse_mean.loc[condA & cond2, 'Vel_' + v]
            data_vitesse_mean.loc[condB & cond1, new_v] = -data_vitesse_mean.loc[condB & cond1, 'Vel_' + v]
            data_vitesse_mean.loc[condB & cond2, new_v] = data_vitesse_mean.loc[condB & cond2, 'Vel_' + v]
            data_vitesse_mean.loc[condC, new_v] = -data_vitesse_mean.loc[condC, 'Vel_' + v]
            data_vitesse_mean.loc[condD, new_v] = data_vitesse_mean.loc[condD, 'Vel_' + v]




            figure_velocity = False
            if figure_velocity:
                print('no')


