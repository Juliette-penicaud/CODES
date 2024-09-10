# 22/05/23 : Je créée les plots pour visualiser les données du file recap all data
# mai : plot du paramètre de Simpson fonction distance à embouchure et tentative fonction du percentage de marée
# 2/06 : plot de la variation en temps de : niveau eau, vitesse (surface? ou colonne ?) et arrivée sal
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
from matplotlib.lines import Line2D

#fontsize = 28
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
#plt.rcParams['xtick.labelsize'] = fontsize - 4
#plt.rcParams['ytick.labelsize'] = fontsize - 4
#plt.rcParams['legend.fontsize'] = fontsize - 4
#s = 25

# VARIABLE AND PARAMETER
year = '2022'
list_month = ['June', 'August', 'Octobre']
i = 2 # 0 1 2 a voir pour faire une boucle si besoin
month = list_month[i]

rep = '/home/penicaud/Documents/Data/Survey_' + month
file = rep + '/Recap_all_param_' + month + '.xlsx'
dict_month = {'June': {'nrows': 87},
              'August': {'nrows': 95},
              'Octobre': {'nrows': 111}}

file_station = rep + '/Stations_' + month + '.xlsx'
df_global = pd.read_excel(file_station, sheet_name=0, nrows=dict_month[month]['nrows'])  # read the stations name
df_global = df_global.dropna(subset=['Stations'])
list_sheet = df_global['Stations'].values
list_sheet = [col for col in list_sheet if not '-' in col]  # remove the 'bistations' to have a unique liste
# corresponding to the devices stations
# EXCLUDE the SXX-25
# df_global = pd.ExcelFile(file)  # df du LISST
# list_sheet = df_global.sheet_names
# WARNING : Commented method is NOT A GOOD ONE BECAUSE BASED ON LISST STATIONS, so missing stations

# I load the depth data
file_depth = rep + '/diff_depth_surface_allinstrum_allstations_' + month + '.xlsx'
df_depth = pd.read_excel(file_depth)
df_depth = df_depth.set_index('Unnamed: 0')
print('ok for the file depth')

add_value = False  # Slower than doing again the tab ...
if add_value:
    for i in list_sheet[16:]:
        station = i
        print('station ', station)
        df = pd.read_excel(file, station)
        df2 = df.copy()
        eas = df2['vitesse east']
        nor = df2['vitesse north']
        df2['angle'] = np.rad2deg(np.arctan(nor / eas))
        # file2 = rep + '/Recap_all_param_' + month + '_copy.xlsx'
        book = load_workbook(file)
        writer = pd.ExcelWriter(file, engine='openpyxl', mode='a', if_sheet_exists='overlay')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        df2.to_excel(writer, sheet_name=station, startrow=0, startcol=0, index=False)  # , header=header)
        writer.save()

print('Phase 1 ok')
critere_prof = 0.6
list_critere_prof = np.arange(0.5,1,0.1) # If prof of instrum < prof max (usually ADCP) : not taken into account. The highest, the strictest
list_critere_prof = [0.7]
layer_2m = True  # if True, the first layer calculated for data of 1st layer will be the nanmean of values < 2m. If
# False, value of the first half layer (until prof_max, max of the devices depth). Anyway, 2d layer is the lower half

# list_val, list_name, list_dist, list_tide, list_transect, list_exclude, \
# list_tide_level, list_time, list_velocity, sal_c1_LOG, sal_c2_LOG, sal_bot_LOG, sal_mean_LOG,\
# sal_c1_IMER, sal_c2_IMER, sal_bot_IMER, sal_mean_IMER, vel_c1, vel_c2, list_prof_ADCP, \
# mean_angle, angle_c1, angle_c2, vel_nor, vel_east, vel_nor_c1, vel_east_c1, vel_nor_c2, vel_east_c2, list_prof_max = \
#     [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[]
# list_tab_all = [list_val, list_name, list_dist, list_tide, list_transect, list_exclude, list_tide_level, list_time,
#                 list_velocity, sal_c1_LOG, sal_c2_LOG, sal_bot_LOG, sal_mean_LOG, sal_c1_IMER, sal_c2_IMER,
#                 sal_bot_IMER, sal_mean_IMER,vel_c1, vel_c2, list_prof_ADCP,
#                 mean_angle, angle_c1, angle_c2, vel_nor, vel_east, vel_nor_c1, vel_east_c1, vel_nor_c2, vel_east_c2]

dict_figure = {'June': {'fixe': {'hlim_inf': 2, 'hlim_sup': 4, 'shift_percentage': 15, 'sal_max': 26,
                                 'vel_min': -1.000, 'vel_max': 1.000, 'day': '/0'}},
               # vel min -800 permet de centrer et faire apparaitre le 0
               'August': {'small fixe': {'hlim_inf': 2., 'hlim_sup': 4, 'shift_percentage': 5, 'sal_max': 25,
                                         'vel_min': -1.200,
                                         'vel_max': 1.800, 'day': '/0'},
                          'fixe': {'hlim_inf': 0, 'hlim_sup': 4, 'shift_percentage': 12, 'sal_max': 26,
                                   'vel_min': -0.800,
                                   'vel_max': 2.200, 'day': '/0'}},
               'Octobre': {'small fixe': {'hlim_inf': 2, 'hlim_sup': 3.2, 'shift_percentage': 5, 'sal_max': 15,
                                          'vel_min': -1.800, 'vel_max': 1.800, 'day': '/'},
                           # every 0.2 instead of 0.4m for level height
                           'fixe': {'hlim_inf': 0, 'hlim_sup': 4, 'shift_percentage': 12, 'sal_max': 26,
                                    'vel_min': -1.200, 'vel_max': 2.000, 'day': '/'}}
               }

def set_newticks(twin_axes):
    new_tick = []
    for i in range(len(twin_axes.get_xticks())):
        tick = twin_axes.get_xticks()[i]
        if tick <= 100:
            print('do not change')
            new_tick.append(tick)
        elif 100 < tick <= 200:
            print('Ebb tide :')
            new_tick.append(-(200 - tick))
        elif tick > 200:
            print('New flood tide')
            new_tick.append(tick - 200)
        else:
            print('PROBLEM IN FUNCTION NEW TICKS')
    return new_tick

# for i in range(len(list_sheet[85:])):  # range(16,18) # :
for critere_prof in list_critere_prof :
    list_val, list_name, list_dist, list_tide, list_transect, list_exclude, \
    list_tide_level, list_time, list_velocity, sal_c1_LOG, sal_c2_LOG, sal_bot_LOG, sal_mean_LOG, \
    sal_c1_IMER, sal_c2_IMER, sal_bot_IMER, sal_mean_IMER, vel_c1, vel_c2, list_prof_ADCP, \
    mean_angle, angle_c1, angle_c2, vel_nor, vel_east, vel_nor_c1, vel_east_c1, vel_nor_c2, vel_east_c2, list_prof_max = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    list_tab_all = [list_val, list_name, list_dist, list_tide, list_transect, list_exclude, list_tide_level, list_time,
                    list_velocity, sal_c1_LOG, sal_c2_LOG, sal_bot_LOG, sal_mean_LOG, sal_c1_IMER, sal_c2_IMER,
                    sal_bot_IMER, sal_mean_IMER, vel_c1, vel_c2, list_prof_ADCP,
                    mean_angle, angle_c1, angle_c2, vel_nor, vel_east, vel_nor_c1, vel_east_c1, vel_nor_c2, vel_east_c2]

    for i in list_sheet:
        # station = list_sheet[i]
        station = i
        print('station', station)
        list_prof_ADCP.append(df_depth['depth ADCP'].loc[df_depth.index == station].values[0])
        bad_sta_IMER = ['AF12', 'AF22', 'A17', 'A34']
        bad_sta_LOG = []
        df = pd.read_excel(file, station)
        case_LISST = df['D50'].last_valid_index()  # NOT USED for the moment
        case_vel = df['module vitesse u'].last_valid_index()
        case_sal_IMER = df['Salinity'].last_valid_index()
        case_sal_LOG = df['Salinity LOG'].last_valid_index()
        case_simpson = df['Simpson'].last_valid_index()  # toujours avoir une valeur,
        # car avec pb de prof, [1] pas tj bonne valeur
        # Calcul de la prof max des 3 instrum pour avoir un idic et exclure val simpson pas représentatives
        prof_max = -1
        c = -1
        for u in ['module vitesse u', 'Density']:  # j'enlève 'D50' tant que l'on n'a pas réglé les soucis de LISST
            if df[u].last_valid_index() is not None:
                c = df[u].last_valid_index()
            else:
                c = 0
            prof_max = max(prof_max, df['Depth'].iloc[c])
        list_prof_max.append(prof_max)
        #print('pmax', prof_max)
        # Besoin : indicateur pour exclure des stations si la profondeur n'est pas assez importante.
        # prof_max = max(df['Depth'].iloc[df[u]].last_valid_index() for u in ['module vitesse u', 'D50', 'Density'] if not df[u].last_valid_index() == None )
        list_name.append(station)
        list_time.append(df['Time'][0])
        list_dist.append(df['Distance'][0])
        list_tide.append(df['Percentage of tide'][0])
        list_tide_level.append(df['Tide level'][0])
        list_transect.append(df['Transect'][0])
        val = np.nan
        if (case_simpson is None) or (df['Depth'].iloc[df['Density'].last_valid_index()] < critere_prof * prof_max) or \
                df['Simpson'].iloc[case_simpson] <= 0:  # TODO : check why Simpson val < 0
            # list_exclude.append(list_name[i-1])
            list_val.append(val)
        else:
            list_val.append(df['Simpson'][case_simpson])
        if case_vel is None:  # No other indication because of the depth of ADCP is supposed to be good (WARN : average over time)
            list_tab = [list_velocity, vel_c1, vel_c2, mean_angle, angle_c1, angle_c2, vel_nor, vel_nor_c1, vel_nor_c2,
                        vel_east, vel_east_c1, vel_east_c2]
            for tab in list_tab:
                tab.append(val)
        else:
            # mean
            list_velocity.append(np.nanmean(df['module vitesse u']))
            mean_angle.append(np.nanmean(df['angle']))
            vel_nor.append(np.nanmean(df['vitesse north']))
            vel_east.append(np.nanmean(df['vitesse east']))
            # Second layer
            vel_c2.append(np.nanmean(df['module vitesse u'].loc[int(case_vel / 2):]))
            angle_c2.append(np.nanmean(df['angle'].loc[int(case_vel / 2):]))
            vel_nor_c2.append(np.nanmean(df['vitesse north'].loc[int(case_vel / 2):]))
            vel_east_c2.append(np.nanmean(df['vitesse east'].loc[int(case_vel / 2):]))
            if layer_2m:
                vel_c1.append(np.nanmean(df['module vitesse u'].loc[df['Depth'] < 2]))  # 2 first meter
                angle_c1.append(np.nanmean(df['angle'].loc[df['Depth'] < 2]))
                vel_nor_c1.append(np.nanmean(df['vitesse north'].loc[df['Depth'] < 2]))
                vel_east_c1.append(np.nanmean(df['vitesse east'].loc[df['Depth'] < 2]))
            else:
                vel_c1.append(np.nanmean(df['module vitesse u'].loc[:int(case_vel / 2)]))
                angle_c1.append(np.nanmean(df['angle'].loc[:int(case_vel / 2)]))
                vel_nor_c1.append(np.nanmean(df['vitesse north'].loc[:int(case_vel / 2)]))
                vel_east_c1.append(np.nanmean(df['vitesse east'].loc[:int(case_vel / 2)]))

        if (case_sal_LOG is None) or (prof_max < 2) or (station in bad_sta_LOG):
            # Detected as wrong for CTD Imer by comparison with CTD LOG):
            # or (df['Depth'].loc[case_sal] < df['Depth'].loc[df['Depth'] == prof_max].values[0]/2) :
            # or (prof_max < 5) : # if the surface is not representative, do not take into account.
            sal_mean_LOG.append(val)
            sal_c1_LOG.append(val)
            sal_c2_LOG.append(val)
            sal_bot_LOG.append(val)
        elif df['Depth'].loc[case_sal_LOG] > critere_prof * prof_max:
            sal_mean_LOG.append(np.nanmean(df['Salinity LOG']))
            if layer_2m:
                sal_c1_LOG.append(np.nanmean(df['Salinity LOG'].loc[df['Depth'] < 2]))  # 2 first meter
            else:
                sal_c1_LOG.append(np.nanmean(df['Salinity LOG'].loc[df['Depth'] < prof_max / 2]))
            sal_c2_LOG.append(np.nanmean(df['Salinity LOG'].loc[df['Depth'] > prof_max / 2]))
            sal_bot_LOG.append(np.nanmean(df['Salinity LOG'].loc[df['Depth'] > prof_max-2]))
        else:  # prof >2 and < crit_prof*prof_max
            # Split the case : is salinity is not None, we can load the surface sal
            sal_c2_LOG.append(val)
            sal_mean_LOG.append(val)
            sal_bot_LOG.append(val)
            if layer_2m:
                sal_c1_LOG.append(np.nanmean(df['Salinity LOG'].loc[df['Depth'] < 2]))  # 2 first meter
            else:
                sal_c1_LOG.append(np.nanmean(df['Salinity LOG'].loc[df['Depth'] < prof_max / 2]))  # 06/06 : ajout condition
                # pour avoir c1 que sur 2 m (plus de stations et toujours la même taille de couche)
            # sal_c1.append(np.nanmean(df['Salinity LOG'].loc[df['Depth'] < prof_max / 2]))  # 05/06 I change to have the layer
            # divided in two accoridng to depth max, and not to the device, i.e instead of next line.
            # sal_c1.append(np.nanmean(df['Salinity LOG'].loc[:int(case_sal / 2)]))  # case_sal/2 or depth max /2 ?

        if (case_sal_IMER is None) or (prof_max < 2) or (station in bad_sta_IMER):
            # Detected as wrong for CTD Imer by comparison with CTD LOG):
            # or (df['Depth'].loc[case_sal] < df['Depth'].loc[df['Depth'] == prof_max].values[0]/2) :
            # or (prof_max < 5) : # if the surface is not representative, do not take into account.
            sal_mean_IMER.append(val)
            sal_c1_IMER.append(val)
            sal_c2_IMER.append(val)
            sal_bot_IMER.append(val)
        elif df['Depth'].loc[case_sal_IMER] > critere_prof * prof_max:
            sal_mean_IMER.append(np.nanmean(df['Salinity']))
            if layer_2m:
                sal_c1_IMER.append(np.nanmean(df['Salinity'].loc[df['Depth'] < 2]))  # 2 first meter
            else:
                sal_c1_IMER.append(np.nanmean(df['Salinity'].loc[df['Depth'] < prof_max / 2]))
            sal_c2_IMER.append(np.nanmean(df['Salinity'].loc[df['Depth'] > prof_max / 2]))
            sal_bot_IMER.append(np.nanmean(df['Salinity'].loc[df['Depth'] > prof_max-2]))
        else:  # prof >2 and < crit_prof*prof_max
            # Split the case : is salinity is not None, we can load the surface sal
            sal_c2_IMER.append(val)
            sal_mean_IMER.append(val)
            sal_bot_IMER.append(val)
            if layer_2m:
                sal_c1_IMER.append(np.nanmean(df['Salinity'].loc[df['Depth'] < 2]))  # 2 first meter
            else:
                sal_c1_IMER.append(np.nanmean(df['Salinity'].loc[df['Depth'] < prof_max / 2]))

    data = pd.DataFrame(list_val, index=list_name)
    # data = data.assign(simpson=list_val, dist = list_dist)
    data = data.rename(columns={0: 'Simpson'})
    data = data.assign(Transect=list_transect, Distance=list_dist, Tide=list_tide, Tide_level=list_tide_level,
                       Time=list_time, Vel_mean=list_velocity, Sal_mean_LOG=sal_mean_LOG, Sal_c1_LOG=sal_c1_LOG,
                       Sal_c2_LOG=sal_c2_LOG, Sal_bot_LOG=sal_bot_LOG,Sal_mean_IMER=sal_mean_IMER, Sal_c1_IMER=sal_c1_IMER,
                       Sal_c2_IMER=sal_c2_IMER, Sal_bot_IMER=sal_bot_IMER,Vel_c1=vel_c1, Vel_c2=vel_c2, depth_ADCP=list_prof_ADCP,
                       Angle_mean=mean_angle,
                       Angle_c1=angle_c1, Angle_c2=angle_c2, mean_nor=vel_nor, mean_east=vel_east,
                       c1_nor=vel_nor_c1, c1_east=vel_east_c1, c2_nor=vel_nor_c2, c2_east=vel_east_c2)
    # TREATMENT : if percentage is -98 ; convert it into 102 to have continuity
    data['Tide 2'] = data['Tide'].values.copy()
    data['Tide 2'].where(data['Tide 2'] > 0, other=100 + (data['Tide 2'] + 100), inplace=True)

    # 21/07 Add of Sp parameter to qualify the estuary
    # 27/09 Modif : I change the min calcultation to a max (so that we expect having the most robust mean, espectially
    # for the bottom average, having an important mean means probably that we have more layers.
    data['Sp'] = (data[['Sal_bot_IMER','Sal_bot_LOG']].max(axis=1) - data[['Sal_c1_IMER','Sal_c1_LOG']].max(axis=1)) \
                 / data[['Sal_mean_IMER','Sal_mean_LOG']].max(axis=1)

    fontsize = 10
    dict_transect_color = {'T1': 'blue', 'T2': 'orange', 'T3': 'green', 'T4': 'cyan'}
    dict_transect = {'T1': 'x', 'T2': 'o', 'T3': '^', 'T4': '*'}

    dict_tide = {'June': {'T1': 'LF', 'T2': 'EE', 'T3': 'MF', 'T4': 'ME'},
                 'August': {'T1': 'MF', 'T2': 'LF', 'T3': 'HT', 'T4': 'EF'},
                 'Octobre': {'T1': 'LE', 'T2': 'EE', 'T3': 'ME', 'T4': 'LF'}}
    dict_tide = {'June': {'T1': '1790', 'T2': '1790', 'T3': '1737', 'T4': '1737'},
                 'August': {'T1': '687', 'T2': '687', 'T3': '930', 'T4': '1577'},
                 'Octobre': {'T1': '856', 'T2': '642', 'T3': '692', 'T4': '692'}}

    # 02/06 : Graph depending on the time of water height , velocity (mean) , and bottom salinity
    type_fixe = 'fixe'  # small fixe or fixe
    condition = (data['Transect'] == type_fixe)  # Warning : need to be on the same day
    data_fixe = data.loc[condition].copy()

    # TODO : NOT ENOUGH because we have the values  mprove the values
    if month == 'August' and type_fixe == 'fixe':
        data_fixe['Tide 2'].where(data_fixe['Tide 2'] > 30, other=data_fixe['Tide 2'] + 200, inplace=True)
    if month == 'Octobre' and type_fixe == 'fixe':
        data_fixe['Tide 2'].where(data_fixe['Tide 2'] > 90, other=data_fixe['Tide 2'] + 200, inplace=True)

    if layer_2m:
        label = ['surface 2m-layer', 'half bottom layer', 'mean']
    else:
        label = ['1st layer', '2d layer', 'mean']

    date1 = data_fixe['Time'].min() - timedelta(minutes=10)
    date2 = data_fixe['Time'].max() + timedelta(minutes=10)

    dict_ADCP = {'June': {"liminf": 7.5, "limsup": 10.5},
                 'August': {"liminf": 6.5, "limsup": 11.5},
                 'Octobre': {"liminf": 8.5, "limsup": 11.5}}

    # 21/06/2023 : Treatment of angle of velocity
    # dict_velocity = {'Mean': 'Vel_mean', 'c1': 'Vel_c1', 'c2': 'Vel_c2' }

    dict_angle = {'section1 ': 40, 'section3': 15, 'section4': 25, 'section5': 77, 'section6': 50,
                  'section7': 10, 'section8': 65, 'section9': 28 , 'section10': 80, 'section11': 33, 'section12': 44}

    section1 = ['fixe']
    section2 = ['A7', 'O24', 'S17', 'S31', 'O10', 'O31', 'O35', 'A27']
    section3 = ['O34', 'A28', 'S1', 'S32', 'S36', 'S34', 'S35', 'S33', 'O32', 'O11', 'O23', 'A29', 'S37', 'O12', 'S37',
                'A8', 'A30', 'S38']
    section4 = ['A9', 'O33']
    section5 = ['S39', 'A10', 'A31']
    section6 = ['O22', 'O13', 'O22-2', 'A32', 'O14', "A11", 'A12']
    section7 = ['SF1', 'A33']
    section8 = ['A34', 'O15', 'O16', 'A35', 'O17', 'A36' , 'OF1']
    section9 = ['O18', 'A37']
    section10 = ["O19", 'A38', 'O20']
    section11 = ['A39', 'O21', 'A40']

    if type_fixe == 'fixe' :
        angle_seuil = 50  # angles for the station fixe values
    elif type_fixe == 'small fixe' and month=='Octobre':
        angle_seuil = 65
    elif type_fixe == 'small fixe' and month=='August':
        angle_seuil = 10
    for v in ['mean', 'c1', 'c2']:
        # Several cases :
        # Case A : nor and east are + : direction NE, sens value depends on the value of the angle
        # Case B : nor and east are - : direction SO, sens value depends on the value of the angle
        # Case C : nor + and east -, i.e angle value -, direction NO, sens is -
        # Case D : nor - and east +, i.e angle value -, direction SE, sens is +
        vnor = v + '_nor'
        veast = v + '_east'
        condA = (data_fixe[vnor] > 0) & (data_fixe[veast] > 0)
        condB = (data_fixe[vnor] < 0) & (data_fixe[veast] < 0)
        condC = (data_fixe[vnor] > 0) & (data_fixe[veast] < 0)
        condD = (data_fixe[vnor] < 0) & (data_fixe[veast] > 0)
        new_v = 'Vel_' + v + '_sens'
        data_fixe[new_v] = np.nan  # data_fixe['Vel_'+v].values.copy()
        cond1 = (angle_seuil < data_fixe['Angle_' + v].abs())
        cond2 = (angle_seuil >= data_fixe['Angle_' + v].abs())
        data_fixe.loc[condA & cond1, new_v] = data_fixe.loc[condA & cond1, 'Vel_' + v]
        data_fixe.loc[condA & cond2, new_v] = -data_fixe.loc[condA & cond2, 'Vel_' + v]
        data_fixe.loc[condB & cond1, new_v] = -data_fixe.loc[condB & cond1, 'Vel_' + v]
        data_fixe.loc[condB & cond2, new_v] = data_fixe.loc[condB & cond2, 'Vel_' + v]
        data_fixe.loc[condC, new_v] = -data_fixe.loc[condC, 'Vel_' + v]
        data_fixe.loc[condD, new_v] = data_fixe.loc[condD, 'Vel_' + v]


    # Outlier de ADCP
    # if month == 'August':
    #     s_to_delete = "AF21"
    #     data_fixe['depth_ADCP'].loc[s_to_delete] = np.nan
    # Problem of sign in Octobre : TODO improve the code to make it more robust
    if month == 'Octobre' and type_fixe=='fixe' :
        data_fixe['Vel_c2_sens'].loc['OF7'] = - data_fixe['Vel_c2_sens'].loc['OF7']
        data_fixe['Vel_c1_sens'].loc['OF44'] = - data_fixe['Vel_c1_sens'].loc['OF44']
        data_fixe['Vel_c1_sens'].loc['OF7'] = - data_fixe['Vel_c1_sens'].loc['OF7']
        data_fixe['Vel_mean_sens'].loc['OF10'] = - data_fixe['Vel_mean_sens'].loc['OF10']
        data_fixe['Vel_c1_sens'].loc['OF10'] = - data_fixe['Vel_c1_sens'].loc['OF10']



    # 21/07 : Figure of Sp for transects fuction of tides ==> DOESNOT WORK BECAUSE OF THE VALUES NOT AVERAGED
    list_transect = ['T1', 'T2', 'T3']
    if month == 'June':
        list_transect.append('T4')
    cmap = 'Spectral'
    # fig, ax = plt.subplots()
    # fig.suptitle(month)
    # ax.set_xlabel('Distance (m)', fontsize=fontsize)
    # ax.set_ylabel('Sp = $\delta$ s /<s>', fontsize=fontsize)
    # ax.grid(True, which='major')
    # ax.grid(True, which='minor')
    # for t in list_transect:
    #     condition = (data['Transect'] == t)  # & (data.index != 'S24') & ( data.index != "S10"))
    #     mean_val = data['Tide'].loc[condition].mean()
    #     x = data['Distance'].loc[condition]
    #     y = data['Sp'].loc[condition]
    #     p1 = ax.scatter(x, y, marker=dict_transect[t], alpha=0.8, c=data['Tide'].loc[condition] , cmap=cmap,
    #                     vmin=-100, vmax=100,  label=t + ' ' + dict_tide[month][t])
    #     ax.plot(x,y, zorder=0.5, color = 'gray')
    #     # for i, txt in enumerate(data['distance'].loc[condition].index):
    #     #    ax.annotate(txt, (x[i], y[i]))
    # cbar = plt.colorbar(p1)
    # ax.set_xlim(-16000,13500)
    # cbar.ax.set_ylabel('% of tide') # , rotation=270)
    # plt.legend()
    # fig.savefig('Sp_transect_' + month + 'critere_' + str(critere_prof) + '.png')
    #
    # # 27/09 : Sp parameter on the fixed stations, test on the depth criterion
    # fig, ax = plt.subplots()
    # fig.suptitle(month)
    # ax.set_xlabel('Time', fontsize=fontsize)
    # ax.set_ylabel('Sp = $\delta$ s /<s>', fontsize=fontsize)
    # ax.grid(True, which='major')
    # ax.grid(True, which='minor')
    # #ax.set_ylim(-15, 300)
    # start_date = datetime(2022, 6, 18, 13, 0, 0) # TO CHANGE
    # end_date = datetime(2022, 6, 18, 20, 0, 0)
    # ax.set_xlim(start_date, end_date)
    # date_form = DateFormatter("%H:%M")  # Define the date format
    # # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax.xaxis.set_major_formatter(date_form)
    # p1 = ax.scatter(data_fixe['Time'], data_fixe['Sp'], marker='x', alpha=0.8, c=data_fixe['Tide'] , cmap=cmap,
    #                     vmin=-100, vmax=100, label=str(critere_prof*100) + ' %')
    # #ax.plot(data_fixe['Time'], data_fixe['Sp'], zorder=0.5, color = 'gray')
    # plt.legend()
    # cbar = plt.colorbar(p1)
    # fig.savefig('Sp_fixe_station_' + month + '_' + str(critere_prof) + 'depth' + '.png')
    if month == 'June':
        start_date = datetime(2022, 6, 18, 13, 0, 0) # TO CHANGE
        end_date = datetime(2022, 6, 18, 20, 0, 0)
    elif month == 'August':
        start_date = datetime(2022, 8, 12, 9, 0, 0) # TO CHANGE
        end_date = datetime(2022, 8, 13, 9, 0, 0)
    elif month == 'Octobre':
        start_date = datetime(2022, 10, 4, 9, 0, 0)
        end_date = datetime(2022, 10, 5, 10, 0, 0)

    figure_simpson=False
    if figure_simpson :
    # 26/09 : simpson parameter on the fixed stations, test on the depth criterion
        fig, ax = plt.subplots()
        fig.suptitle(month)
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
        ax.grid(True, which='major')
        ax.grid(True, which='minor')
        ax.set_ylim(-15, 300)
        ax.set_xlim(start_date, end_date)
        date_form = DateFormatter("%H:%M")  # Define the date format
        # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_formatter(date_form)
        p1 = ax.scatter(data_fixe['Time'], data_fixe['Simpson'], marker='x', alpha=0.8, c=data_fixe['Tide'] , cmap=cmap,
                            vmin=-100, vmax=100, label=str(critere_prof*100) + ' %', zorder=0.5)
        #plt.legend()
        cbar = plt.colorbar(p1)
        ax.set_yscale('log')
        ax.set_ylim([10 ** -1, 2.5 * 10 ** 3])
        fig.savefig('Simpson_fixe_station_' + month + '_' + str(critere_prof) + 'depth' + '.png')

        # 12/07 : I add != marker for each transect, and a color of lineplot corresponding to the mean tide% of the transect
        list_transect = ['T1', 'T2', 'T3']
        if month == 'June':
            list_transect.append('T4')
        cmap = 'Spectral'
        fig, ax = plt.subplots()
        fig.suptitle(month)
        ax.set_xlabel('Distance (m)', fontsize=fontsize)
        ax.set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
        ax.grid(True, which='major')
        ax.grid(True, which='minor')
        data_simspon_plot = data.dropna(subset=['Distance', 'Simpson', 'Tide'])
        list_label = []
        handles = []
        for t in list_transect:
            condition = (data['Transect'] == t)  # & (data.index != 'S24') & ( data.index != "S10"))
            mean_val = data['Tide'].loc[condition].mean()
            x = data_simspon_plot['Distance'].loc[condition]
            y = data_simspon_plot['Simpson'].loc[condition]
            name_transect = t[0] + month[0] + t[1]
            label = name_transect + ' Q=' + dict_tide[month][t] + ' m$^{3}$/s'
            list_label.append(label)
            p1 = ax.scatter(x, y, marker=dict_transect[t], alpha=0.8, c=data_simspon_plot['Tide'].loc[condition] ,
                            cmap=cmap, vmin=-100, vmax=100,  label=label, zorder=15)
            # for i, txt in enumerate(data['distance'].loc[condition].index):
            #    ax.annotate(txt, (x[i], y[i]))
            # custom_legend = [plt.Line2D([0], [0], marker=dict_transect[t], color='gray', label=label, markersize=10,
            #                            markerfacecolor='none')]
            #handles.append(p1)
            ax.plot(x, y, zorder=0.5, color = 'gray')
            #ax.legend(handles=custom_legend)
        cbar = plt.colorbar(p1)
        cbar.ax.set_ylabel('% of tide') # , rotation=270)
        # ax.set_xlim(-8000, 13500)
        ax.set_xlim(-16000,13500)
        ax.set_yscale('log')
        ax.set_ylim([10 ** -1, 2.5 * 10 ** 3])
        #plt.legend(handles=custom_legend)
        custom_legend = [plt.Line2D([0], [0], marker=dict_transect[t], color='gray', label=label, markersize=10,
                                markerfacecolor='none') for t, label in zip(list_transect, list_label)]
        plt.legend(handles = custom_legend)
        fig.savefig('Simpson_log_filtered_' + month + '_' + str(critere_prof) + 'depth' + '.png')
    ###############################################################################################################
fontsize = 22
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['xtick.labelsize'] = fontsize - 4
plt.rcParams['ytick.labelsize'] = fontsize - 4
plt.rcParams['legend.fontsize'] = fontsize - 5
s=10

# 2 subplots
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(15,10))
ax=axs[0]
ax.grid(which='both')
# pour octobre, il faut mettre data_fixe['Time'] et twin2 en ax
p2, = ax.plot(data_fixe['Time'], data_fixe['Vel_mean_sens']/1000, alpha=0.8, color='red', marker='x', ls='--',
                     label=label[2])
p2bis, = ax.plot(data_fixe['Time'], data_fixe['Vel_c2_sens']/1000, alpha=0.8, color='red', marker='o', ls = ':',
                        label=label[1])
p2ter, = ax.plot(data_fixe['Time'], data_fixe['Vel_c1_sens']/1000, alpha=0.8, color='red',marker='d',
                        label=label[0])
ax.set_ylim(-2.0, 2.5) # dict_figure[month][type_fixe]['sal_max'])
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.axhline(y = 0, color = 'black')
ax.set_ylabel('Velocity (m/s)', fontsize=fontsize)
ax.legend()
twin_axes=ax.twiny()
shift = 1.5
twin_axes.set_xticklabels(set_newticks(twin_axes))
ax.set_xlim(date1, date2)  # IMPORTANT so that we know the margins
shift = 1.5  # dict_figure[month][type_fixe]['shift_percentage'] # shift is in 1.28% 1.38% or 1.51% depending on
# length of Ebb or Flood tide (13-11h)
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']) - shift,
                   np.nanmax(data_fixe['Tide 2']) + shift)  # 12 pour august fixe, 5 august small
l = ax.get_ylim()
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']), np.nanmax(data_fixe['Tide 2']))
twin_axes.set_xticklabels(set_newticks(twin_axes))
date_form = DateFormatter("%H:%M")  # Define the date format
ax.xaxis.set_major_formatter(date_form)

ax=axs[1]
ax.grid(which='both')
# pour octobre, il faut mettre data_fixe['Time'] et twin2 en ax
p2, = ax.plot(data_fixe['Time'], data_fixe['Sal_mean_IMER'], alpha=0.8, color='blue', marker='x', ls='--',
                     label=label[2])
p2bis, = ax.plot(data_fixe['Time'], data_fixe['Sal_c2_IMER'], alpha=0.8, color='blue', marker='o', ls = ':',
                        label=label[1])
p2ter, = ax.plot(data_fixe['Time'], data_fixe['Sal_c1_IMER'], alpha=0.8, color='blue',marker='d',
                        label=label[0])
ax.set_ylim(0, 25) # dict_figure[month][type_fixe]['sal_max'])
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))
ax.axhline(y = 20, color = 'black')
ax.set_ylabel('Salinity (PSU)', fontsize=fontsize)
ax.legend()
twin_axes=ax.twiny()
shift = 1.5
twin_axes.set_xticklabels(set_newticks(twin_axes))
ax.set_xlim(date1, date2)  # IMPORTANT so that we know the margins
shift = 1.5  # dict_figure[month][type_fixe]['shift_percentage'] # shift is in 1.28% 1.38% or 1.51% depending on
# length of Ebb or Flood tide (13-11h)
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']) - shift,
                   np.nanmax(data_fixe['Tide 2']) + shift)  # 12 pour august fixe, 5 august small
l = ax.get_ylim()
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']), np.nanmax(data_fixe['Tide 2']))
twin_axes.set_xticklabels(set_newticks(twin_axes))
date_form = DateFormatter("%H:%M")  # Define the date format
ax.xaxis.set_major_formatter(date_form)
outfile = 'Temporal_salinity_velocity_evolution_' + type_fixe + month
if layer_2m:
    outfile = outfile + '_layer_2m'
outfile = outfile + '.png'
fig.savefig(outfile)



# 19/01 : Je change, je ne fais q'un seul subplot pour la salinité, et sans le tidal elevation à côté
# Make several subplots : salt and tide evolution, vel and tide evolution and water discharge
fig, ax = plt.subplots(figsize=(15,10))
ax.grid(True)
# pour octobre, il faut mettre data_fixe['Time'] et twin2 en ax
p2, = ax.plot(data_fixe['Time'], data_fixe['Sal_mean_IMER'], alpha=0.8, color='blue', marker='x', ls='--',
                     label=label[2])
p2bis, = ax.plot(data_fixe['Time'], data_fixe['Sal_c2_IMER'], alpha=0.8, color='blue', marker='o', ls = ':',
                        label=label[1])
p2ter, = ax.plot(data_fixe['Time'], data_fixe['Sal_c1_IMER'], alpha=0.8, color='blue',marker='d',
                        label=label[0])
ax.set_ylim(0, 25) # dict_figure[month][type_fixe]['sal_max'])
ax.set_ylabel('Salinity (PSU)', fontsize=fontsize)
ax.legend()
twin_axes=ax.twiny()
shift = 1.5
twin_axes.set_xticklabels(set_newticks(twin_axes))
ax.set_xlim(date1, date2)  # IMPORTANT so that we know the margins
shift = 1.5  # dict_figure[month][type_fixe]['shift_percentage'] # shift is in 1.28% 1.38% or 1.51% depending on
# length of Ebb or Flood tide (13-11h)
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']) - shift,
                   np.nanmax(data_fixe['Tide 2']) + shift)  # 12 pour august fixe, 5 august small
l = ax.get_ylim()
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']), np.nanmax(data_fixe['Tide 2']))
twin_axes.set_xticklabels(set_newticks(twin_axes))
date_form = DateFormatter("%H:%M")  # Define the date format
ax.xaxis.set_major_formatter(date_form)
outfile = 'Temporal_salinity_evolution_' + type_fixe + month
if layer_2m:
    outfile = outfile + '_layer_2m'
fig.savefig(outfile)

#19/01 second fig with velocity evolution
fig, ax = plt.subplots(figsize=(15,10))
ax.grid(True)
# pour octobre, il faut mettre data_fixe['Time'] et twin2 en ax
p2, = ax.plot(data_fixe['Time'], data_fixe['Vel_mean_sens']/1000, alpha=0.8, color='red', marker='x', ls='--',
                     label=label[2])
p2bis, = ax.plot(data_fixe['Time'], data_fixe['Vel_c2_sens']/1000, alpha=0.8, color='red', marker='o', ls = ':',
                        label=label[1])
p2ter, = ax.plot(data_fixe['Time'], data_fixe['Vel_c1_sens']/1000, alpha=0.8, color='red',marker='d',
                        label=label[0])
ax.set_ylim(-2.0, 2.0) # dict_figure[month][type_fixe]['sal_max'])
ax.set_ylabel('Velocity (m/s)', fontsize=fontsize)
ax.legend()
twin_axes=ax.twiny()
shift = 1.5
twin_axes.set_xticklabels(set_newticks(twin_axes))
ax.set_xlim(date1, date2)  # IMPORTANT so that we know the margins
shift = 1.5  # dict_figure[month][type_fixe]['shift_percentage'] # shift is in 1.28% 1.38% or 1.51% depending on
# length of Ebb or Flood tide (13-11h)
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']) - shift,
                   np.nanmax(data_fixe['Tide 2']) + shift)  # 12 pour august fixe, 5 august small
l = ax.get_ylim()
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']), np.nanmax(data_fixe['Tide 2']))
twin_axes.set_xticklabels(set_newticks(twin_axes))
date_form = DateFormatter("%H:%M")  # Define the date format
ax.xaxis.set_major_formatter(date_form)
outfile = 'Temporal_velocity_evolution_' + type_fixe + month
if layer_2m:
    outfile = outfile + '_layer_2m'
fig.savefig(outfile)

################################################################################




# Make several subplots : salt and tide evolution, vel and tide evolution and water discharge
fig, axs = plt.subplots(nrows=2, sharex=True)
fig.suptitle('Fixed station ' + str(data_fixe['Time'][0].day) + dict_figure[month][type_fixe]['day'] +
             str(data_fixe['Time'][0].month))
fig.subplots_adjust(right=0.85)
################  First subplot : salinity
ax = axs[0]
ax.grid(True)
twin2 = ax.twinx()
twin2.grid(None)
ax.set_ylabel('Tide level (m)', fontsize=fontsize)
p0, = ax.plot(data_fixe['Time'], data_fixe['Tide_level'], alpha=0.8, color='k')  # , marker='o')

#if month == 'June' :
#    ax.set_yticks(np.arange(dict_figure[month][type_fixe]['hlim_inf'], dict_figure[month][type_fixe]['hlim_sup'] + 0.1,
#                            0.4))
#else :
ax.set_ylim(dict_figure[month][type_fixe]['hlim_inf'], dict_figure[month][type_fixe]['hlim_sup'])

# p0height = twin_height.plot(data_fixe['Time'], data_fixe['depth_ADCP'], color='grey')
# p1, = twin1.plot(data_fixe['time'], data_fixe['Simpson'], alpha=0.8,  color='orange')
twin_axes = twin2.twiny()

# pour octobre, il faut mettre data_fixe['Time'] et twin2 en ax
p2, = twin2.plot(data_fixe['Time'], data_fixe['Sal_mean_IMER'], alpha=0.8, color='blue', ls='--',
                     label=label[2])
p2bis, = twin2.plot(data_fixe['Time'], data_fixe['Sal_c2_IMER'], alpha=0.8, color='blue', ls = ':',
                        label=label[1])
p2ter, = twin2.plot(data_fixe['Time'], data_fixe['Sal_c1_IMER'], alpha=0.8, color='blue', ls='-.',
                        label=label[0])
twin2.set_ylim(0, dict_figure[month][type_fixe]['sal_max'])
# Color : blue, navy, turquoise
twin2.legend()

shift = 1.5  # dict_figure[month][type_fixe]['shift_percentage'] # shift is in 1.28% 1.38% or 1.51% depending on
# length of Ebb or Flood tide (13-11h)
#twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']) - shift,
#                   np.nanmax(data_fixe['Tide 2']) + shift)  # 12 pour august fixe, 5 august small

l2 = twin2.get_ylim()
l = ax.get_ylim()
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']), np.nanmax(data_fixe['Tide 2']))
f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
ticks = f(ax.get_yticks())
twin2.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
twin_axes.set_xticklabels(set_newticks(twin_axes))

twin2.set(ylabel="Salinity (PSU)")
ax.yaxis.label.set_color(p0.get_color())
twin2.yaxis.label.set_color(p2.get_color())
twin2.tick_params(axis='y', colors=p2.get_color())
ax.tick_params(axis='y', colors=p0.get_color())
#twin_axes.legend(fontsize=fontsize - 2, loc='upper left')
ax.set_xticklabels('')

################"""  2d subplot : velocity
ax = axs[1]
ax.set_ylabel('Tide level (m)', fontsize=fontsize)
ax.grid(True)
ax.set_xlabel('Time', fontsize=fontsize)
twin3 = ax.twinx()
twin_axes = twin3.twiny()
# twin_height = ax.twinx()
# twin_height.spines.left.set_position(("axes", 0.2))  # Used to shift the 3d y axis
# twin_height.grid(None)
#twin3.spines.right.set_position(("axes", 1.2))  # Used to shift the 3d y axis
twin3.grid(None)

ax.plot(data_fixe['Time'], data_fixe['Tide_level'], alpha=0.8, color='k')  # , marker='o')
#if month == 'June' :
#    ax.set_yticks(np.arange(dict_figure[month][type_fixe]['hlim_inf'], dict_figure[month][type_fixe]['hlim_sup'] + 0.1,
#                            0.4))
#else :
ax.set_ylim(dict_figure[month][type_fixe]['hlim_inf'], dict_figure[month][type_fixe]['hlim_sup'])
# ptest, = twin2.plot(data_fixe['Time'], data_fixe['Salinity'], alpha=0.8, color='green', marker='+', label='mean')
p3, = twin3.plot(data_fixe['Time'], data_fixe['Vel_mean_sens']/1000, alpha=0.8, color='red', ls = '--',
                 label=label[2])  # , marker='o')
p4, = twin3.plot(data_fixe['Time'], data_fixe['Vel_c1_sens']/1000, alpha=0.8, color='red', label=label[0], ls = ':')  # , marker='o')
p5, = twin3.plot(data_fixe['Time'], data_fixe['Vel_c2_sens']/1000, alpha=0.8, color='red', label=label[1], ls='-.')  # , marker='o')
# color = orange, pink, red
#twin2.set_ylim(0, dict_figure[month][type_fixe]['sal_max'])
twin3.set_ylim(dict_figure[month][type_fixe]['vel_min'], dict_figure[month][type_fixe]['vel_max'])
ax.set_xlim(date1, date2)  # IMPORTANT so that we know the margins
shift = 1.5  # dict_figure[month][type_fixe]['shift_percentage'] # shift is in 1.28% 1.38% or 1.51% depending on
# length of Ebb or Flood tide (13-11h)
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']) - shift,
                   np.nanmax(data_fixe['Tide 2']) + shift)  # 12 pour august fixe, 5 august small

l = ax.get_ylim()
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']), np.nanmax(data_fixe['Tide 2']))
l3 = twin3.get_ylim()
f = lambda x: l3[0] + (x - l[0]) / (l[1] - l[0]) * (l3[1] - l3[0])
ticks2 = f(ax.get_yticks())
twin3.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks2))
twin_axes.set_xticklabels(set_newticks(twin_axes))

# twin1.set(ylabel="Simspon")
# Management of the twin axis : title colors and ticks
twin3.set(ylabel="Velocity (m/s)")
ax.yaxis.label.set_color(p0.get_color())
twin3.yaxis.label.set_color(p3.get_color())
ax.tick_params(axis='y', colors=p0.get_color())
twin3.tick_params(axis='y', colors=p3.get_color())

date_form = DateFormatter("%H:%M")  # Define the date format
ax.xaxis.set_major_formatter(date_form)
#twin_axes.legend(fontsize=fontsize - 2, loc='upper left')
twin3.legend(fontsize=fontsize - 2, loc='lower center')
twin3.axhline(y = 0, color = 'grey', linestyle = '--')

outfile = 'TEST_subplots_Water_level_velocitysens_salt_' + type_fixe + month
if layer_2m:
    outfile = outfile + '_layer_2m'
fig.savefig(outfile)

# 5/12 :

# 5/12 : plot
df_percentage_salinity = pd.DataFrame({'Salinity_last0':[80.5,79.2, 70.8],'Salinity_firstnon0':[81.9,83.3, 75], 'first_23PSU_salinity':[89.58, 100, 92], 'discharge':[1954,1577,691]})
df_percentage_velocity = pd.DataFrame({'percentage_velocity+to-':[68.1,], 'percentage_velocity-to+': [np.nan, ],
                                       'percentage_min':[89.6,] ,'value_min':[] ,  'percentage_max':[np.nan, ] ,
                                       'value_max':[np.nan,],
                                       'discharge': [1954, 1577, 691]})
# 26/01 :
df_percentage_salinity = pd.DataFrame({'Salinity_entrance':[80.0,80.1, 75.6], 'first_23PSU_salinity':[89.58, 100, 92], 'discharge':[1954,1577,691]})


# FIgure of the percentage vs discharge for the salinity parameters
fig, ax = plt.subplots(figsize=(18, 10))
ax.grid(True, alpha=0.5)
ax.set_ylabel('Arrival salinity (% of tide)', fontsize=fontsize - 2)
ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=fontsize - 2)
ax.set_ylim(65,100)
s=55
ax.scatter(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_last0'], color='grey', s=s, label='Last 0 salinity')
ax.scatter(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_firstnon0'], color='grey', s=s, marker = 'd', label='First non 0 salinity')
ax.scatter(df_percentage_salinity["discharge"],  (df_percentage_salinity['Salinity_firstnon0']+df_percentage_salinity['Salinity_last0'])/2, color='black', s=s)
ax.fill_between(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_last0'], df_percentage_salinity['Salinity_firstnon0'],
                color='blue', alpha = 0.2, label = 'Arrival salinity')

ax.scatter(df_percentage_salinity["discharge"],  df_percentage_salinity['first_19PSU_salinity'] , color='green', marker='+', s=s, label = 'First 19 salinity')
ax.fill_between(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_firstnon0'], df_percentage_salinity['first_19PSU_salinity'],
                color='green', alpha = 0.2, label='Salinity instauration')
x = np.arange(500, 2000, 10)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_last0'])
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + '% of tide, r=' + str(np.round(r_value, 3))
print(p_value)
ax.plot(x, slope * x + intercept, lw=1, color='grey', label=label)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_salinity["discharge"], df_percentage_salinity['Salinity_firstnon0'])
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1, color='violet', label=label)
print(p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_percentage_salinity["discharge"],
                                                               (df_percentage_salinity['Salinity_firstnon0']+df_percentage_salinity['Salinity_last0'])/2)
label = "{:.1e}".format(slope) + ' discharge + ' + str(np.round(intercept, 2)) + ' % of tide, r=' + str(np.round(r_value, 3))
ax.plot(x, slope * x + intercept, lw=1, color='orange', label=label)
print(p_value)
legend = ax.legend(loc='upper left', ncol=2)
# Set the font size for the legend labels
for label in legend.get_texts():
    label.set_fontsize(12)  # Set the desired font size
fig.savefig('Percentage_last0salinity_vs_discharge.png', format='png')










sys.exit(1)

# Figure with Salt, velocity, tidal height
fig, ax = plt.subplots()
fig.suptitle('Fixed station ' + str(data_fixe['Time'][0].day) + dict_figure[month][type_fixe]['day'] +
             str(data_fixe['Time'][0].month))
fig.subplots_adjust(right=0.75)
ax.set_xlabel('Time', fontsize=fontsize)
ax.set_ylabel('Tide level (m)', fontsize=fontsize)
ax.grid(True)
twin2 = ax.twinx()
twin3 = ax.twinx()
# twin_height = ax.twinx()
# twin_height.spines.left.set_position(("axes", 0.2))  # Used to shift the 3d y axis
# twin_height.grid(None)
twin3.spines.right.set_position(("axes", 1.2))  # Used to shift the 3d y axis
twin2.grid(None)
twin3.grid(None)

p0, = ax.plot(data_fixe['Time'], data_fixe['Tide_level'], alpha=0.8, color='k')  # , marker='o')
# p0height = twin_height.plot(data_fixe['Time'], data_fixe['depth_ADCP'], color='grey')
# p1, = twin1.plot(data_fixe['time'], data_fixe['Simpson'], alpha=0.8,  color='orange')
twin_axes = twin2.twiny()
p2, = twin_axes.plot(data_fixe['Tide 2'], data_fixe['Salinity'], alpha=0.8, color='blue', marker='+', label=label[2])
p2bis, = twin_axes.plot(data_fixe['Tide 2'], data_fixe['Sal_c2'], alpha=0.8, color='navy', marker='+', label=label[1])
p2ter, = twin_axes.plot(data_fixe['Tide 2'], data_fixe['Sal_c1'], alpha=0.8, color='turquoise', marker='+',
                        label=label[0])
# ptest, = twin2.plot(data_fixe['Time'], data_fixe['Salinity'], alpha=0.8, color='green', marker='+', label='mean')
p3, = twin3.plot(data_fixe['Time'], data_fixe['Vel_mean_sens'], alpha=0.8, color='orange',
                 label=label[2])  # , marker='o')
p4, = twin3.plot(data_fixe['Time'], data_fixe['Vel_c1_sens'], alpha=0.8, color='pink', label=label[1])  # , marker='o')
p5, = twin3.plot(data_fixe['Time'], data_fixe['Vel_c2_sens'], alpha=0.8, color='red', label=label[0])  # , marker='o')

ax.set_ylim(dict_figure[month][type_fixe]['hlim_inf'], dict_figure[month][type_fixe]['hlim_sup']) # Only for June
# ax.set_yticks(
#    np.arange(dict_figure[month][type_fixe]['hlim_inf'], dict_figure[month][type_fixe]['hlim_sup'] + 0.1, 0.4))
twin2.set_ylim(0, dict_figure[month][type_fixe]['sal_max'])
twin3.set_ylim(dict_figure[month][type_fixe]['vel_min'], dict_figure[month][type_fixe]['vel_max'])
ax.set_xlim(date1, date2)  # IMPORTANT so that we know the margins
shift = 1.5  # dict_figure[month][type_fixe]['shift_percentage'] # shift is in 1.28% 1.38% or 1.51% depending on
# length of Ebb or Flood tide (13-11h)
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']) - shift,
                   np.nanmax(data_fixe['Tide 2']) + shift)  # 12 pour august fixe, 5 august small

l = ax.get_ylim()
l2 = twin2.get_ylim()
twin_axes.set_xlim(np.nanmin(data_fixe['Tide 2']), np.nanmax(data_fixe['Tide 2']))
l3 = twin3.get_ylim()
f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
ticks = f(ax.get_yticks())
twin2.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
twin2.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
f = lambda x: l3[0] + (x - l[0]) / (l[1] - l[0]) * (l3[1] - l3[0])
ticks2 = f(ax.get_yticks())
twin3.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks2))
twin_axes.set_xticklabels(set_newticks(twin_axes))

# twin1.set(ylabel="Simspon")
# Management of the twin axis : title colors and ticks
twin2.set(ylabel="Salinity (PSU)")
twin3.set(ylabel="Velocity (mm/s)")
ax.yaxis.label.set_color(p0.get_color())
ax.yaxis.label.set_color(p0.get_color())
twin2.yaxis.label.set_color(p2.get_color())
twin3.yaxis.label.set_color(p3.get_color())
ax.tick_params(axis='y', colors=p0.get_color())
twin2.tick_params(axis='y', colors=p2.get_color())
twin3.tick_params(axis='y', colors=p3.get_color())

date_form = DateFormatter("%H:%M")  # Define the date format
ax.xaxis.set_major_formatter(date_form)
twin_axes.legend(fontsize=fontsize - 2, loc='upper left')
twin3.legend(fontsize=fontsize - 2, loc='lower left')
outfile = 'TEST_Water_level_velocitysens_salt_' + type_fixe + month
if layer_2m:
    outfile = outfile + '_layer_2m'
fig.savefig(outfile)

sys.exit(1)


# Figure to compare height differences : ADCP - Hon Dau
fig, ax = plt.subplots()
fig.suptitle('Fixed station ' + str(data_fixe['Time'][0].day) + dict_figure[month][type_fixe]['day'] +
             str(data_fixe['Time'][0].month))
ax.set_xlabel('Time', fontsize=fontsize)
ax.set_ylabel('Tide level (m)', fontsize=fontsize)
ax.grid(True)
twin_height = ax.twinx()
twin_height.grid(None)
twin_height.set_ylim(dict_ADCP[month]['liminf'], dict_ADCP[month]['limsup'])  # IMPORTANT so that we know the margins

p0, = ax.plot(data_fixe['Time'], data_fixe['Tide_level'], alpha=0.8, color='k')  # , marker='o')
p0height = twin_height.plot(data_fixe['Time'], data_fixe['depth_ADCP'], color='grey')
ax.set_yticks(
    np.arange(dict_figure[month][type_fixe]['hlim_inf'], dict_figure[month][type_fixe]['hlim_sup'] + 0.1, 0.4))
ax.yaxis.label.set_color(p0.get_color())
ax.yaxis.label.set_color(p0.get_color())
ax.set_xlim(date1, date2)  # IMPORTANT so that we know the margins
date_form = DateFormatter("%H:%M")  # Define the date format
ax.xaxis.set_major_formatter(date_form)
l = ax.get_ylim()
l2 = twin_height.get_ylim()
f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
ticks = f(ax.get_yticks())
twin_height.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))

out = 'Test_different_height_' + month
fig.savefig(out)

# plot transect par transect, avec % marée, distance à embouchure et phi
fig, ax = plt.subplots()
fig.suptitle(month)
ax.set_xlabel('Distance to the mouth (m)', fontsize=fontsize)
ax.set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for t in ['T1', 'T2', 'T3', 'T4']:
    condition = (data['Transect'] == t)  # & (data.index != 'S24') & ( data.index != "S10"))
    x = data['Distance'].loc[condition]
    y = data['Simpson'].loc[condition]
    ax.plot(x, y, marker=dict_transect[t], alpha=0.8, color=dict_transect[t], label=t + ' ' + dict_tide[month][t])
    # for i, txt in enumerate(data['distance'].loc[condition].index):
    #    ax.annotate(txt, (x[i], y[i]))
ax.set_yscale('log')
# ax.set_xlim(-8000, 13500)
ax.set_xlim(-16000,4000)
ax.set_ylim([10 ** -1, 10 ** 3])
plt.legend()
fig.savefig('test_simpson_log_filtered_cirtere60%pmax_'+month)

# 12/07 : I add != marker for each transect, and a color of lineplot corresponding to the mean tide% of the transect
list_transect = ['T1', 'T2', 'T3']
if month == 'June':
    list_transect.append('T4')
cmap = 'Spectral'
fig, ax = plt.subplots()
fig.suptitle(month)
ax.set_xlabel('Distance (m)', fontsize=fontsize)
ax.set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
ax.grid(True, which='major')
ax.grid(True, which='minor')
for t in list_transect:
    condition = (data['Transect'] == t)  # & (data.index != 'S24') & ( data.index != "S10"))
    mean_val = data['Tide'].loc[condition].mean()
    x = data['Distance'].loc[condition]
    y = data['Simpson'].loc[condition]
    p1 = ax.scatter(x, y, marker=dict_transect[t], alpha=0.8, c=data['Tide'].loc[condition] , cmap=cmap,
                    vmin=-100, vmax=100,  label=t + ' ' + dict_tide[month][t])
    ax.plot(x,y, zorder=0.5, color = 'gray')
    # for i, txt in enumerate(data['distance'].loc[condition].index):
    #    ax.annotate(txt, (x[i], y[i]))
ax.set_yscale('log')
# ax.set_xlim(-8000, 13500)
ax.set_xlim(-16000,13500)
ax.set_ylim([10 ** -1, 10 ** 3])
fig.colorbar(p1)
plt.legend()
fig.savefig('test_cmap_simpson_log_filtered_cirtere60%pmax_'+month)



# plot of simpson parameter for fixed station
fig, ax = plt.subplots()
fig.suptitle(month)
ax.set_xlabel('Tide percentage', fontsize=fontsize)
ax.set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
ax.grid(True)
ax.set_xlim(-100, 100)
ax.set_ylim(-15, 200)
condition = (data['Transect'] == 'fixe')
ax.plot(data['tide'].loc[condition], data['Simpson'].loc[condition], marker='x', alpha=0.8, color='k')
fig.savefig('test_simpson_fixe')

# plot of simpson parameter for fixed station
fig, ax = plt.subplots()
ax.set_xlabel('Tide percentage', fontsize=fontsize)
ax.set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
ax.grid(True)
condition = (data['Transect'] == 'fixe')
ax.plot(data['tide'].loc[condition], data['Simpson'].loc[condition], marker='x', alpha=0.8, color='k')
ax.set_yscale('log')
ax.set_xlim(-100, 100)
fig.savefig('test_simpson_fixe_log')

sys.exit(1)

fig, axs = plt.subplots(ncols=2, nrows=2)
fig.title = month
axs[1, 0].set_xlabel('Distance to the mouth (m)', fontsize=fontsize)
axs[1, 1].set_xlabel('Distance to the mouth (m)', fontsize=fontsize)
axs[0, 0].set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
axs[1, 0].set_ylabel('$\Phi$ ($J.m^{-3}$)', fontsize=fontsize)
axs.set_xlim(-1500, 6200)
axs.set_ylim(0, 600)
i = 0
for t in ['T1', 'T2', 'T3', 'T4']:
    condition = (data['Transect'] == t)
    if i == 0:
        ax = axs[0, 0]
    if i == 1:
        ax = axs[0, 1]
    if i == 2:
        ax = axs[1, 0]
    if i == 3:
        ax = axs[1, 1]
    ax.scatter(data['distance'].loc[condition], data['simpson'].loc[condition], marker='x', alpha=0.8,
               color=dict_transect[t])
    i = i + 1
fig.savefig('test_simpson_'+month)



# 18/10/23  : Df of the transitions values of velocity
# 1st value June, August, Octobre

vitesse_postoneg = [75,70,55]
vitesse_negtopos = [np.nan, -85,-75]
vitesse_posmax = [np.nan, -50, -45]
val_posmax = [np.nan,1.45,1.2]
vitesse_posmaxlast = [np.nan, np.nan ,20]
discharge = [1954,1577,692]

data_transition_velocity = pd.DataFrame(vitesse_postoneg)
data_transition_velocity = data_transition_velocity.rename(columns={0: 'Vitesse_postoneg'})
data_transition_velocity = data_transition_velocity.assign(Vitesse_negtopos=vitesse_postoneg,
                                                           Vitesses_posmaxfirst=vitesse_posmaxfirst,
                                                           Vitesse_posmaxlast=vitesse_posmaxlast,
                                                           Val_posmax=val_posmax,
                                                           Discharge=discharge)
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(data_transition_velocity["Vitesse_postoneg"], data_transition_velocity["Discharge"])
x=np.arange(0,100,1)

fig,ax = plt.subplots()
ax.scatter(discharge,vitesse_postoneg)
ax.plot(x,slope*x+intercept)

plt.show()

#########################################################"
# 23/01, new version of the velocities positive to negative
Discharge=[1954,1577,691]
Reverse_pos_to_neg_surf=[71.44, 67.17, 54.53]
Reverse_pos_to_neg_bot=[66.18, 64.69, 54.53]
Reverse_neg_to_pos_surf=[np.nan, -99.36, -79.7]
Reverse_neg_to_pos_bot=[np.nan, -88.76, -72.2]
df_test = pd.DataFrame({'Discharge': Discharge, "Reverse_pos_to_neg_surf":Reverse_pos_to_neg_surf,
                        "Reverse_pos_to_neg_bot":Reverse_pos_to_neg_bot,
                        "Reverse_neg_to_pos_surf":Reverse_neg_to_pos_surf,
                        "Reverse_neg_to_pos_bot":Reverse_neg_to_pos_surf})
slope, intercept, r_value, p_value, std_err = stats.linregress(df_test["Discharge"], df_test['Reverse_pos_to_neg_surf'])

print(str(np.round(slope,3))+' x + '+str(np.round(intercept, 3))+', r = '+str(np.round(r_value, 3))+
      ' p_value = '+str(np.round(p_value, 3))+' std = '+str(np.round(std_err, 3)))
