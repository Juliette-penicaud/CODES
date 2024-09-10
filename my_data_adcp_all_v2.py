# 14/02/24 : Je rajoute la fonction pour calculer le sens du courant.
# Test de la
# 27/07/2022 Treat ADCP DATA
# 24/07/23 : Improve previous prog
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import cmcrameri as cmc
import scipy.signal as signal
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
# mpl.use('TkAgg')
import math
# from adcploader import *
from mpl_toolkits import mplot3d
import warnings
from scipy.ndimage import gaussian_filter


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


def velocity_and_direction_for_entire_transect(month, df1, name_transect, nb_bin):
    case = False
    dict_angle = {'section1': {'angle': 55, 'case': True},
                  'section2': {'angle': 15, 'case': True}} # -22 is ok for the stations J31 to J38, but not for J29-J30
                  # 'section2': {'angle': -22, 'case': True}}

    dict_section = {'section1': [("T1", "06"),("T2", "06"), ("T3", "06"), ("T1", "08") , ("T4", "08"), ("T1", "10"),
                                 ("T3", "10")],
                    # TJ1, sans S1, et sans S9 S10,
                    # TJ3, sans S21 à S24
                    # TO3 juste à partir de O24 ou O25
                    'section2': [("T4", "06"), ("T4", "10")]
                    # TO4: sans le O30, à partir de O31, et attention, le transect est fait dans les deux sens.
                    # Trop complexe pour attribuer un sens : TA2, TA3, TO2
                    }

    for s in ['section1', 'section2']:
        if ((name_transect, month) in dict_section[s]):
            section = s
            case = True
    if not case :
        print('No angle value, exit')
    if case :
        df = df1.copy()
        angle1 = dict_angle[section]['angle']
        angle2 = angle1 + 180
        angle1bis = angle1 % 360
        sens_vel, mag, dir = [], [], []
        for i in range(1, nb_bin):
            string1 = "Mag, mm/s, " + str(i)
            string2 = "Dir, deg, " + str(i)
            string3 = "Sens vel " + str(i)
            mag.append(string1)
            dir.append(string2)
            sens_vel.append(string3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[sens_vel] = df[mag].copy()

            if not (df[dir].dtypes == np.float64).all() :
                df[dir] = df[dir].replace(',', '.', regex=True).astype(float)
        condition = (df[dir] < angle1bis) | (df[dir] > angle2)

        # Change the sign of the oriented velocities in the selected rows
        # df.loc[condition.any(axis=1), sens_vel] *= -1
        df_sens_vel = df[sens_vel].copy()
        condition.columns = df_sens_vel.columns # I change the header so I can change the map
        df_sens_vel[condition] *= -1
        return df_sens_vel

# LIST OF variables : station, station2 same but with '' inbetween to plot on the 2d graph, profilevertical : the number of the rows of the table

month = '06'  # TO COMPLETE
year = '2022'
transects = ['T4'] # ['T1', 'T2', 'T3', 'T4']  # T1 to T4 or small fixe or fixe
part = 1  # 1 or 2 for the fixed station cut : August, fixe
No_save = True
filter = True
skiprow = 11
draw_sta = 1
plot_signed_vel = True
unit = 'm/s' # put the values in mm/s in m/s
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


for transect in transects :
    if month == '06':
        shift=5
        nfile=1
        sep = ' '
        survey = 'June'
        depth_bin = 0.3
        blank_1st_bin = 0.6
        nb_bin = 45  # TO COMPLETE
        moy = 16  # EVEN NUMBER #nb of PROFILES (and because 1hz, of secondes also !) we want to average.
        # 16 for the june survey where 1 measure per second, august, 1 measure every 5 seconds
        # Limit of the layers if layer=true
        lim1 = 7  # nb of the limit bin between layer 1 and 2
        lim2 = 17  # nb of the limit bin between layer 2 and 3 #FOR 1806 : 17 contain every intense bins, 15 around 80 % (a vue de nez)
        if transect == 'T1' or transect == 'T2':
            day = '16'
            T = ''
            type = 'transect'
        elif transect == 'T3' or transect == 'T4':
            day = '17'
            T = '_' + transect
            type = 'transect'
        elif transect == 'fixe':
            day = '18'
            T = ''
            type = 'fixed'
        else:
            print('problem with the transect')
        file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + T + '_alldata_BT.csv'

    elif month == '08':
        sep = ','
        survey = 'August'
        depth_bin = 0.1
        blank_1st_bin = 0.66
        moy = 6  # WARNING : fq=0.2HZ, it is the nb of PROFILES we average, and moy*5 is the period
        nb_bin = 180
        if transect == 'small fixe':
            lim1 = 115
        elif transect == 'T1':
            lim1 = 120  # WARNING : should be a mooving part, because thin layer not horizontal
        if transect == 'T1':
            type = 'transect'
            tide = 'Mid Flood'
            nfile = 1
            day = '10'
            hour_deb = '10h53'
            hour_end = '12h24'
            shift = 25
        elif transect == 'T2':
            type = 'transect'
            tide = 'Late Flood'
            nfile = 1
            day = '10'
            hour_deb = '10h53'
            hour_end = '12h24'
            hour_deb = '12h41'
            hour_end = '14h35'
            shift = 10
        elif transect == 'small fixe':
            type = 'fixed'
            tide = 'HT-ME'
            nfile = 1
            day = '10'
            hour_deb = '14h36'
            hour_end = '20h14'
            shift = 50
            # First values of current of the fixed station on the previous file
        elif transect == 'T3':
            type = 'transect'
            tide = 'HT'
            nfile = 1
            day = '11'
            hour_deb = '15h01'
            hour_end = '18h31'
            shift = 65
        elif transect == 'T4':
            type = 'transect'
            tide = 'Early Flood'
            nfile = 1
            day = '12'
            hour_deb = '06h46'
            hour_end = '07h06'
            shift = 3
            # first minutes on another file
        elif transect == 'fixe' and part == 1:
            type = 'fixed'
            tide = 'cycle MF-EE'
            nfile = 1
            day = '12'
            hour_deb = '11h48'
            hour_end = '19h49'
            shift = 10
        elif transect == 'fixe' and part == 2:
            type = 'fixed'
            tide = 'cycle ME-MF'
            nfile = 1
            day = '12'
            hour_deb = '19h50'
            hour_end = '08h55'
            shift = 10
        else:
            print('PROBLEM TRANSECT august')
            sys.exit(1)

        if nfile == 1:
            file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_' + hour_deb + '-' + hour_end + '_BT.csv'
        elif nfile == 2:
            file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_' + hour_deb + '-' + hour_end + '_BT.csv'
            # file2 = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_'+hour_deb2+'-'+hour_end2+ '_BT.csv'

    elif month == '10':
        survey = 'Octobre'
        sep = ','
        depth_bin = 0.1
        blank_1st_bin = 0.66
        moy = 6  # WARNING : fq=0.2HZ, it is the nb of PROFILES we average, and moy*5 is the period
        nb_bin = 120

        if transect == 'T1':
            nb_bin = 120
            type = 'transect'
            tide = 'Late Ebb'
            nfile = 1
            day = '02'
            hour_deb = '17h19'
            hour_end = '18h28'
            shift = 25  # A voir
        elif transect == 'T2':
            nb_bin = 160
            type = 'transect'
            tide = 'Early Ebb'
            nfile = 1
            day = '03'
            hour_deb = '10h20'
            hour_end = '18h00'
            shift = 10  # a voir
        elif transect == 'T3':  # pareil, voir comment faire ?
            nb_bin = 160
            type = 'transect'
            tide = 'Mid Ebb'
            nfile = 1
            day = '03'
            hour_deb = '10h20'
            hour_end = '18h00'
            shift = 50  # a voir
            # First values of current of the fixed station on the previous file
        elif transect == 'T4':
            nb_bin = 160
            type = 'transect'
            tide = 'Late Flood'
            nfile = 1
            day = '04'
            hour_deb = '07h40'
            hour_end = '08h24'
            shift = 20
        elif transect == 'fixe':
            nb_bin = 160
            # fichier sépareé en 2, voir si tout mettre ne 1 ? en changeant les valeurs des ensembles ?
            type = 'fixed'
            tide = '1 cycle'
            nfile = 1
            day = '04'
            shift = 3
            day_deb = '04'
            day_end = '05'
            hour_deb = '10h19'
            hour_end = '09h32'
        elif transect == 'small fixe':  # fichier sépareé en 2, voir si tout mettre ne 1 ? en changeant les valeurs des ensembles ?
            nb_bin = 160
            type = 'fixed'
            tide = 'Mid Ebb'
            nfile = 1
            day = '03'
            hour_deb = '10h20'
            hour_end = '18h00'
            shift = 10
        else:
            print('PROBLEM TRANSECT octobre')

        if (transect == 'T2') or (transect == 'T3') or (transect=='small fixe'):
            transect2 = 'T2_T3'
        else:
            transect2 = transect

        if nfile == 1:
            file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + month + day + '_' + transect2 + '_' + hour_deb + '_' + hour_end + '.csv'
            if transect == 'fixe':
                file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + month + day + '_FO_.csv'
        elif nfile == 2:
            file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_' + hour_deb + '-' + hour_end + '_BT.csv'

    else:
        print("PROBLEM WITH THE MONTH")
    print(file)

    # 24/07 creation of a loop to get back the information of transect, time,
    # 21/02 : pourquoi j'utilise le recap plutôt que le "station" ?
    file_recap = '/home/penicaud/Documents/Data/Survey_' + survey + '/Recap_all_param_' + survey + '.xlsx'
    df_global = pd.ExcelFile(file_recap)  # df of all stations
    list_sheet = df_global.sheet_names
    list_time, list_transect, list_percentage = [], [], []
    for i in list_sheet:
        df_recap = pd.read_excel(file_recap, sheet_name=i)
        # station = list_sheet[i]
        station = i
        print('station', station)
        list_time.append(df_recap['Time'][0])
        list_transect.append(df_recap['Transect'][0])
        list_percentage.append(df_recap['Percentage of tide'][0])
    df_sta = pd.DataFrame(list_sheet)
    df_sta = df_sta.rename(columns={0: 'Stations'})
    if month == '06':
        df_sta['Stations'] = df_sta['Stations'].str.replace('S', 'J')
    df_sta = df_sta.assign(Transect=list_transect, Percentage=list_percentage, Time=list_time)

    #############################            PARAMETER TO FILL         ###############################"
    var = 'profile'  # 'dir or profile #TO COMPLETE
    dim = '3d'  # or '3d' #TO COMPLETE
    if var == 'dir':
        dim = '2d'
    CTD_to_observe = 'LOG'  # 'LOG' or 'IMER'

    layer = 0  # 0 1 or 2 depending on the number of layer we want to show #Bool to add the averaged velocity of the layers, need to define the lim of the layers
    remove_last_val = 1  # remove the last bin value
    color_arrow_dir = 0  # 0 or 1, if 1 add a color to the profile 3D arrows to have a better idea of the direction
    correct_dir_segment = 0  # 0 or 1, Change the north south orientation of 3D profile into the "along segment" direction, to not take into account the meanders
    #######################################################################################""
    # load the data of the csv file of ADCP data
    col_list = ["Ens", "HH", "MM", "SS"]
    sta = []  # stations

    for i in range(1, nb_bin + 1):
        string = "Eas, mm/s, " + str(i)
        col_list.append(string)
    for i in range(1, nb_bin + 1):
        string = "Nor, mm/s, " + str(i)
        col_list.append(string)
    for i in range(1, nb_bin + 1):
        string = "Mag, mm/s, " + str(i)
        col_list.append(string)
    for i in range(1, nb_bin + 1):
        string = "Dir, deg, " + str(i)
        col_list.append(string)

    df = pd.read_csv(file, skiprows=skiprow, low_memory=False, sep=sep)  # , usecols=col_list)
    # df = pd.DataFrame(data)
    hour = df["HH"]
    mn = df["MM"]
    sec = df["SS"]
    ens = df["Ens"]

    mag, dir, sens_vel = [], [], []
    for i in range(1, nb_bin):
        string1 = "Mag, mm/s, " + str(i)
        string2 = "Dir, deg, " + str(i)
        string3 = "Sens vel " + str(i)
        mag.append(string1)
        dir.append(string2)
        sens_vel.append(string3)

    # manip on the data to have the values of the first bins on the top of the figure
    magnitude = pd.DataFrame(df, columns=mag)
    magnitude2 = magnitude.transpose() # 24/11/23 : I change .transpose() to .T, because it was upside down...
    magnitude2 = magnitude2.iloc[::-1] # Cette manip met les permiers bins en bas du df (dir, deg, 1) pour que le
    # pcolormesh fonctionne bien (il plot par défaut la premiere ligne en bas)
    # 14/02 : in fact that was ok, I put back the transpose()
    # magnitude2 = magnitude2.iloc[::-1]
    direction = pd.DataFrame(df, columns=dir)
    direction2 = direction.transpose()
    direction2 = direction2.iloc[::-1] # Cette manip met les permiers bins en bas du df (dir, deg, 1) pour que le
    # pcolormesh fonctionne bien (il plot par défaut la premiere ligne en bas)

    # 21/02
    if plot_signed_vel:
        vel_sens = velocity_and_direction_for_entire_transect(month, df, transect, nb_bin)
        # vel_sens = pd.DataFrame(df_signed, columns=sens_vel)
        vel_sens2 = vel_sens.transpose()
        vel_sens2 = vel_sens2.iloc[::-1]

    if nfile == 2:
        data_file2 = pd.read_csv(file2, skiprows=skiprow, low_memory=False, sep=',', usecols=col_list)
        df_file2 = pd.DataFrame(data_file2)
        hour_file2 = df_file2["HH"]
        mn_file2 = df_file2["MM"]
        sec_file2 = df_file2["SS"]
        ens_file2 = df_file2["Ens"]

        magnitude_file2 = pd.DataFrame(data_file2, columns=mag)
        magnitude2_file2 = magnitude_file2.T
        # magnitude2_file2 = magnitude2_file2.iloc[::-1]
        direction_file2 = pd.DataFrame(data_file2, columns=dir)
        direction2_file2 = direction_file2.transpose()
        direction2_file2 = direction2_file2.iloc[::-1]

    ######################################################################################
    # DF of the stations of the transects
    dict_label = {'June': {'T1': 1, 'T2': 0, 'T3': 0, 'T4': 0},
                  'August': {'T1': 1, 'T2': 0, 'T3': 0, 'T4': 1},
                  'Octobre': {'T1': 1, 'T2': 0, 'T3': 1, 'T4': 0}}

    if type == 'transect' :
        label_case = dict_label[survey][transect]
        if label_case == 0:
            label2 = 'River'
            label1 = 'Ocean'
        if label_case == 1:
            label1 = 'River'
            label2 = 'Ocean'

    condition = (df_sta['Transect'] == transect)  # Warning : need to be on the same day
    df2 = df_sta.loc[condition].copy()
    print(df2)
    station = df2['Stations'].copy()
    hour_station = df2['Time'].copy()

    # Add the numero of the ensemble of each hour
    num = []
    # print('hour deb, hour end ', int(hour_deb[0:2]), 'h',int(hour_deb[3:5]), int(hour_end[0:2]), 'h', int(hour_end[3:5]))
    # print('hour deb2, hour end ', int(hour_deb2[0:2]), int(hour_deb2[3:5]), int(hour_end2[0:2]), int(hour_end2[3:5]))
    for h in hour_station:
        hour_sta = h.hour
        mn_sta = h.minute
        day_sta = h.day
        print('hour sta', hour_sta, mn_sta)
        # sec_sta = 30 #sec= 30 for file where 1measure/sec, the first measure corresponding to HH and MM
        # if hour_sta>=int(hour_deb[0:2]) and hour_sta<=int(hour_end[0:2]) mn_sta>=int(hour_deb[0:2]):
        if month == '10' and type == 'fixed':
            condition1 = (datetime(2022,10,4,10,20,0) < h < datetime(2022,10,5,9,32,0))
        elif month == '06' :
            print('TODO here')
            condition1 = True # True car l'ADCP a toujours fonctionné, pas de stations non faites
        else :
            condition1 = (hour_sta == int(hour_deb[0:2]) and mn_sta >= int(hour_deb[3:5])) or (
                hour_sta == int(hour_end[0:2]) and mn_sta <= int(hour_end[3:5])) or (
                    int(hour_deb[0:2]) < hour_sta < int(hour_end[0:2]))
        if condition1:  # TODO : attention si jamais sur station 24h
            print('case1')
            condition = ((df.HH == hour_sta) & (df.MM == mn_sta) & (df.DA == day_sta))  # & (df.SS == sec_sta)) #
            l = df.index[condition]
            # gives the index of the several measures at the time HH MM (for august, one measure every 5s)
            print('l', l)
            l = l[0] + 1  # +1 because it gives the index and indexing begins by 0
            print('l', l)
        if not (condition1):
            if nfile == 2:
                print('Case 2, 2 files')
                condition2 = (hour_sta == int(hour_deb2[0:2]) and mn_sta >= int(hour_deb2[3:5])) or (
                        hour_sta == int(hour_end2[0:2]) and mn_sta <= int(hour_end2[3:5])) or (
                                     hour_sta > int(hour_deb2[0:2]) and hour_sta < int(hour_end2[0:2]))
                if condition2:
                    print('hour station in the 2d file')
                    condition = ((df_file2.HH == hour_sta) & (df_file2.MM == mn_sta))  # & (df.SS == sec_sta))
                    l = df_file2.index[condition]
                    print('l', l)
                    l = l[0] + 1  # +1 because it gives the index and indexing begins by 0
                else:
                    print('No station found in the 2d file')
                    l = -10

            else:
                print('No station found')
                l = -10

        print('hour', h, l)
        num.append(l)

    df2.insert(2, 'numero', num)
    print(df2)

    if var == 'profile' and dim == '3d':  # va créer des fleches 3D.
        df2_station = df2[df2.numero != -10]
        station_CTD = df2_station['Stations']  # ['stations']
        station2_CTD = station_CTD.copy()
        station2_CTD = station2_CTD.values.tolist()  # from df or series to list
        station2_CTD.insert(0, '')
        i = 1
        while i < 2 * len(station_CTD):  # need this trick only for 3D, to plot the vertical profiles on the values of
            station2_CTD.insert(i, '')
            i += 2
        profilevertical_CTD = df2_station['numero']

    print('df2 station', df2[df2.numero != -10])

    cmap = cmc.cm.imola
    cmap = cmc.cm.hawaii_r
    cmap2 = plt.cm.plasma
    cmap2 = plt.cm.twilight
    fontsize = 20 #13

    # print((df.iloc[-1]))#==24186) #gives a serie
    # print(df[df.columns[0]].count()) #Warning : if the file doesnt begin by 0 or 1, doesn't work
    # print(ens.iloc[-1]) #THE GOOD SOLUTION

    # X axis
    # x=np.linspace(1,df2.at[len(df2['stations'])-1,'numero'],len(df2['stations']), dtype=int)
    # TODO : what if first or 2d values are -10

    incr = 0  # TODO Check if it works everytime or just if it depends on the number of values in the 1st file
    if nfile == 1:
        if (month == '06') & (transect == 'T4'):
            # I cut the beginning and the end of transect
            deb = df2['numero'].iloc[0]
            end = df2['numero'].iloc[-2]
        else :
            deb = df2['numero'].iloc[0]
            end = df2['numero'].iloc[-1]
        for n2 in df2['numero']:  # loop over the whole stations to change if limits are -10
            if deb == -10:
                print('deb = -10')
                deb = df2['numero'].iloc[incr]
            if end == -10:
                end = df2['numero'].iloc[-1 - incr]
            incr = incr + 1
        print('deb end', deb, end)

    elif nfile == 2:
        n1 = 0  # n1 is the previous value to compare with the new one. If n2 < n1 that means that we changed file
        deb = df2['numero'].iloc[0]  # deb is the first value of the first file
        end2 = df2['numero'].iloc[-1]  # end is the last value of the last file
        deb2 = 0  # has to be initialize to increment the values if is equal to -10
        for n2 in df2['numero']:  # loop over the whole stations to determine the limit between the two files
            if end2 == -10:
                end2 = df2['numero'].iloc[-1 - incr]
            if deb == -10:
                deb = df2['numero'].iloc[incr]
            if deb2 == -10:
                deb2 = df2['numero'].iloc[incr]
            if n2 < n1:
                if n1 == 0:
                    end = deb  # that means that there is only one and only value of station in the first file
                else:
                    end = n1
                deb2 = n2
            n2 = n1
            incr = incr + 1
        print('2 files, deb end, deb2, end2', deb, end, deb2, end2)

    # Fix the limits of the plot by adding if possible some values before and after the last station, to have a better overview
    # TODO : problem if 2 files : the1st value at the end of 1st file > values after ... ==> Comment concatener les 2 fichiers ? #How to determine the good values ? Is linespace going to work ?
    if survey == 'June':
        step_file = 200
    else:
        step_file = 50
    if deb != 1:
        if deb - ens.iloc[0] > step_file:
            deb = deb - step_file  # 200 was ok for the 1meas/sec
        else:
            deb = ens.iloc[0]
    if nfile == 1:
        if end != ens.iloc[-1]:
            if ens.iloc[-1] - end > step_file:
                end = end + step_file
            else:
                end = ens.iloc[-1] - 1
        print(' 1 file : deb, end', deb, end)
        x = np.linspace(deb, end, 4, dtype=int)
        print('x', x)
    elif nfile == 2:
        if end2 != ens_file2.iloc[-1]:
            if ens_file2.iloc[-1] - end2 > step_file:
                end2 = end2 + step_file
            else:
                end2 = ens_file2.iloc[-1] - 1
            x1 = deb  # First guess : only one value in the first file
            x2 = np.linspace(deb2, end2, 4, dtype=int)
            x = np.insert(x2, 0, x1)
            print('2 files, x', x)

    if transect == 'fixe' and month == '08':
        deb = 1
        end = ens.iloc[-1] - 1
        print('case SF24 deb end ', deb, end)
        x = np.linspace(deb, end, 4, dtype=int)
        print('x', x)

    time = []
    xi1 = 0
    for xi in x:
        xi = xi - 1  # to speak in numero, beginning by 1 and not by 0
        if nfile == 1:
            if hour[xi] < 10:
                h = '0' + str(hour[xi])
            else:
                h = str(hour[xi])
            if mn[xi] < 10:
                m = '0' + str(mn[xi])
            else:
                m = str(mn[xi])
            if sec[xi] < 10:
                s = '0' + str(sec[xi])
            else:
                s = str(sec[xi])
            time.append(h + ':' + m)  # 24/07 comment the part + ':' + s)
        elif nfile == 2:
            print('TIME creation axis, 2 files')
            if xi < xi1:  # means that we are in the 2d file #TODO : WARNING what if the first file is really short ?
                if hour_file2[xi] < 10:
                    h = '0' + str(hour_file2[xi])
                else:
                    h = str(hour_file2[xi])
                if mn_file2[xi] < 10:
                    m = '0' + str(mn_file2[xi])
                else:
                    m = str(mn_file2[xi])
                if sec_file2[xi] < 10:
                    s = '0' + str(sec_file2[xi])
                else:
                    s = str(sec_file2[xi])
            else:  # still in the first file

                if hour[xi] < 10:
                    h = '0' + str(hour[xi])
                else:
                    h = str(hour[xi])
                if mn[xi] < 10:
                    m = '0' + str(mn[xi])
                else:
                    m = str(mn[xi])
                if sec[xi] < 10:
                    s = '0' + str(sec[xi])
                else:
                    s = str(sec[xi])
            time.append(h + ':' + m + ':' + s)
            xi1 = xi  # memory of the n-1 value
        # print('hour ', hour[xi], min[xi], sec[xi])
    print('time', time)

    # Axis y
    y = np.linspace(0, nb_bin - 1, 3)
    print('y', y)
    dmax = depth_bin * nb_bin + blank_1st_bin
    depth = np.around(np.arange(-dmax, -depth_bin, depth_bin), decimals=1)
    # print('depth', depth)
    depth2 = [depth[0], -(depth[-1] - depth[0]) / 2, depth[-1]]
    #depth2 = depth[::int(nb_bin / 2)]
    print('depth2', depth2)  # better to use this one, the other is not right every time
    print('depth', depth[::int(nb_bin / 2)])
    yplot = np.arange(1, nb_bin + 1, 1)
    # print('yplot', yplot)
    # parameter for figures
    n = 1  # indicator for quiver plot legend
    labelcolor = 'black'
    incr = 0
    print('deoth2', depth2)

    ###############################################################################################################
    # 21/02 : velocity oriented figures
    gaussian = False
    fig, ax = plt.subplots(figsize=(10, 4))  # constrained_layout=True)
    profilevertical = df2['numero']
    if type == 'transect':
        name_transect = transect[0] + survey[0] + transect[1]
        if transect == 'T4' and month == '06':
            profilevertical = df2['numero'][:-1]
    else:
        name_transect = transect
    vmax = 1500  # 2000
    if plot_signed_vel:
        disp = 'Velocity'
        var = vel_sens2
        cmap = cmc.cm.vik
        extend = 'both'
        vmin = -vmax
    else:
        disp = 'Magnitude'
        var = magnitude2
        extend = 'max'
        vmin = 0
    disp = disp + ' ' + unit
    if filter:
        if gaussian:
            sigma = 1.0
            variable = gaussian_filter(var, sigma)
        else:
            variable = signal.medfilt2d(var, 5)  # calcul de la turbidité avec filter median 5
    else:
        variable = var
    if unit == 'm/s':
        vmax = vmax / 1000
        vmin = vmin / 1000
        variable = variable / 1000
    month_letter = 'june' if month == '06' else 'august' if month == '08' else 'october' if month == '10' else None
    ax.set_title(day + ' ' + month_letter + ', ' + name_transect, fontsize=fontsize, pad=20)
    ax.set_ylabel('Depth (m)', fontsize=fontsize - 2)
    ax.set_ylim(0, nb_bin)
    ax.set_yticks(y)
    ax.set_yticklabels(labels=depth2)  # depth[::int(nb_bin/2)])
    # ax.set_yticklabels(labels=depth[::int(nb_bin/2)], fontsize=fontsize-2)
    ax.set_xlabel('Time', fontsize=fontsize - 2)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_xticks(x)
    ax.set_xticklabels(labels=time, fontsize=fontsize - 4)
    # date_form = DateFormatter("%H:%M")  # Define the date format
    # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax.xaxis.set_major_formatter(date_form)
    # Je teste pcolor plutôt que pcolormesh pour lisser plus.
    p1 = ax.pcolor(variable, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(p1, ax=ax, extend=extend, pad=0.05)  # , ticks=1)#ax=ax
    cbar.set_label(label=disp, fontsize=fontsize - 2)
    cbar.ax.tick_params(labelsize=fontsize - 5)
    # contour_levels = np.arange(-2,2.5, 0.5)
    # plt.contour(variable,  levels=contour_levels, colors='k')  # Add contour lines
    if type == 'transect':  # ocean and river sides
        if dim == '2d':
            ax.text(x[0] - 10, -20, label1, fontsize=fontsize - 4, fontweight='bold')
            ax.text(x[-1] - 10, -20, label2, fontsize=fontsize - 4, fontweight='bold')
        elif dim == '3d':
            ax.text(x[0] - 10, -40, label1, fontsize=fontsize - 4, fontweight='bold')
            ax.text(x[-1] - 10, -40, label2, fontsize=fontsize - 4, fontweight='bold')

    # Plot the vertical profiles only
    xver = 0  # increment only for the list of "stations"
    if draw_sta == 1:
        for xvertical in profilevertical:  # I changed here to test
            # s = station[xver]
            s = df2['Stations'].iloc[xver]
            # percentage =
            print('station', s)
            if df2['numero'].iloc[xver] == -10:  # skipped because too many station in one place
                print('station skipped', df2['numero'].iloc[xver])
            else:
                ax.axvline(xvertical, color='black', lw=0.5)
                if (transect == 'SF_24.1' and s in ['AF12', 'AF13', 'AF14', 'AF15', 'AF16', 'AF17', 'AF18']):
                    print('station SF24.1 skipped')
                else:
                    ax.text(xvertical - shift, y=np.max(yplot) + 1, s=s, color='k', fontsize=fontsize - 5)
                    # WARNING : the xvertical-300 depends on the xaxis size!!! 300 is ok for T4
                    # ax.text(xvertical - shift, y= 1, s=s, color='k',fontsize=fontsize - 4)
            xver = xver + 1
    # fig.set_size_inches(11, 5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 5)
    plt.subplots_adjust(right=0.98, left=0.1, bottom=0.15, top=0.85)  # Adjust layout to leave space for colorbar
    # plt.tight_layout()
    outfile = 'article_ADCP_' + month + '_' + transect
    if plot_signed_vel:
        outfile = outfile + '_sens_vel'
    if filter:
        outfile = outfile + '_filtered'
        if gaussian:
            outfile = outfile + '_gaussian'
    outfile = outfile + '.png'
    fig.savefig(outfile, format='png')
    ##########################################################################################################

    # TODO : attention on peut etre ne 2D avec des stations
    if dim == '2d':
        fig, ax = plt.subplots(constrained_layout=True)#nrows=2, sharex=True, constrained_layout=True)
        # ax = axs[0]
        posx = [0.09, 0.25, 0.45, 0.65, 0.85, 0.97]  # position for the 2d graph for the legend of the averaged layer
        posy = [0.05, 0.28, 0.65]
        profilevertical = df2['numero']
    elif dim == '3d':
        fig, ax = plt.subplots(figsize = (18,15)) #constrained_layout=True)
        # fig = plt.figure()
        # ax = fig.add_subplot(2, 1, 1)  # TODO : 2 plots différents pour les aligner
        station2 = station.copy()
        station2 = station2.values.tolist()  # from df or series to list
        station2.insert(0, '')
        i = 1
        while i < 2 * len(station):  # need this trick only for 3D, to plot the vertical profiles on the values of
            station2.insert(i, '')
            i += 2
        profilevertical = df2['numero']

    # 1st plot of the magnitude of current
    # fig.suptitle('Current magnitude and velocity profiles '+transect)

    if type == 'transect':
        name_transect = transect[0] + survey[0] + transect[1]
    else:
        name_transect = transect
    vmax = 1500  # 2000
    if plot_signed_vel:
        disp = 'Velocity'
        var = vel_sens2
        cmap = cmc.cm.vik
        extend = 'both'
        vmin = -vmax
    else:
        disp = 'Magnitude'
        var = magnitude2
        extend = 'max'
        vmin = 0
    disp = disp + ' ' + unit

    if filter:
        variable = signal.medfilt2d(var, 5)  # calcul de la turbidité avec filter median 5
    else:
        variable = var
    if unit == 'm/s':
        vmax = vmax / 1000
        vmin = vmin/1000
        variable = variable / 1000

    ax.set_ylabel('Depth (m)', fontsize=fontsize - 2)
    # ax.set_yticks(y)
    # ax.set_yticklabels(labels=depth2, fontsize=fontsize-2)
    ax.set_ylim(0, nb_bin)
    ax.set_yticks(y)
    ax.set_yticklabels(labels=depth2)  # depth[::int(nb_bin/2)])

    # ax.set_yticklabels(labels=depth[::int(nb_bin/2)], fontsize=fontsize-2)
    # ax.set_xlabel('Time', fontsize=fontsize)
    # TODO : for the 2 files
    if nfile == 1:
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_xticks(x)
        ax.set_xticklabels(labels=time, fontsize=fontsize - 4)
        #date_form = DateFormatter("%H:%M")  # Define the date format
        #plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        #ax.xaxis.set_major_formatter(date_form)
    # ax.set_xlim(0, np.max(x) + 1)
    # ax.set_xticks(ticks=x)  # , labels=stations)
    # ax.set_xticklabels(stations)  # , minor=True)
    #if transect == 'T4' or transect == 'TA4' or transect == 'SF_24.2':
    #    vmax = 1750
    #elif transect == 'TA2' or transect == 'TA3' or transect == 'SF1':
    #    vmax = 1250
    # elif transect=='SF_24.1':
    #    vmax=900
    #else:
    p1 = ax.pcolormesh(variable, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(p1, label=disp, ax=ax, extend=extend)  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=8)
    if layer == 1:
        print('horizontal line on the graph layer=1')
        lineh2 = ax.axhline(lim1, xmin=0, xmax=1, color='k')
    elif layer == 2:
        lineh2 = ax.axhline(lim1, xmin=0, xmax=1, color='k')
        lineh3 = ax.axhline(lim2, xmin=0, xmax=1, color='k')
    if type == 'transect':  # ocean and river sides
        if dim == '2d':
            ax.text(x[0] - 10, -20, label1, fontsize=fontsize - 4, fontweight='bold')
            ax.text(x[-1] - 10, -20, label2, fontsize=fontsize - 4, fontweight='bold')
        elif dim == '3d':
            ax.text(x[0] - 10, -40, label1, fontsize=fontsize - 4, fontweight='bold')
            ax.text(x[-1] - 10, -40, label2, fontsize=fontsize - 4, fontweight='bold')

    # TODO : for the 2 files
    # Plot the vertical profiles only
    xver = 0  # increment only for the list of "stations"
    if draw_sta == 1 :
        for xvertical in profilevertical:  # I changed here to test
            # s = station[xver]
            s = df2['Stations'].iloc[xver]
            # percentage =
            print('station', s)
            if df2['numero'].iloc[xver] == -10:  # skipped because too many station in one place
                print('station skipped', df2['numero'].iloc[xver])
            else:
                ax.axvline(xvertical, color='black')
                if (transect == 'SF_24.1' and s in ['AF12', 'AF13', 'AF14', 'AF15', 'AF16', 'AF17', 'AF18']):
                    print('station SF24.1 skipped')
                else:
                    ax.text(xvertical - shift, y=np.max(yplot) + 1, s=s, color='k', fontsize=fontsize - 5)
                    # WARNING : the xvertical-300 depends on the xaxis size!!! 300 is ok for T4
                    # ax.text(xvertical - shift, y= 1, s=s, color='k',fontsize=fontsize - 4)

            xver = xver + 1
    #fig.set_size_inches(11, 5)
    #outfile = 'ADCP_magnitude_filtered_' + month + '_' + transect + '.png'
    outfile = 'ADCP_'+month+'_'+transect
    if plot_signed_vel :
        outfile = outfile + '_sens_vel'
    if filter :
        outfile = outfile + '_filtered'
    outfile = outfile + '.png'
    fig.savefig(outfile, format='png')
    #fig.savefig('outfile_magnitude_' + month + '_' + transect, format='png')

    ############################" 24/11 ##############" New figure with GOOD DATA not upside down
    fig, ax = plt.subplots(constrained_layout=True)
    #fig.suptitle(day + '/' + month + ', ' + transect + ' ' + tide)
    disp = 'Magnitude'
    disp = disp + ' (m/s)'
    ax.set_ylabel('Depth (m)', fontsize=fontsize - 2)
    ax.set_ylim(0, nb_bin)
    ax.set_yticks(y)
    ax.set_yticklabels(labels=depth2)  # depth[::int(nb_bin/2)])
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_xticks(x)
    ax.set_xticklabels(labels=time, fontsize=fontsize - 4)
    vmax = 1
    variable = signal.medfilt2d(magnitude2, 5)  # calcul de la turbidité avec filter median 5
    variable = variable / 1000
    cbar = plt.colorbar(p1, label=disp, ax=ax, extend='max')  # , ticks=1)#ax=ax
    cbar.ax.tick_params(labelsize=8)

    if var == 'dir':
        # 2d plot with the direction in degrees
        ax = axs[1]
        disp = 'Direction (degree)'
        p2 = ax.pcolormesh(direction2, cmap=cmap2, vmin=0, vmax=360)
        ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.set_ylim(0, nb_bin)
        ax.set_yticks(y)
        ax.set_yticklabels(labels=depth2)  # depth[::int(nb_bin/2)])
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_xticks(x)
        ax.set_xticklabels(labels=time)
        # 24/07 : I add the percentage of the stations.
        # ax.text(x[-1] - 10, -40, label2, fontsize=fontsize - 4)
        cbar = plt.colorbar(p2, label=disp, ax=ax)  # , ticks=60)#ax=ax
        cbar.ax.tick_params(labelsize=8)
        # Plot the vertical profiles only
        if draw_sta :
            xver = 0  # increment only for the list of "stations"
            for xvertical in profilevertical:  # I changed here to test
                # s = station[xver]
                s = df2['Stations'].iloc[xver]
                print('station', s)
                if df2['numero'].iloc[xver] == -10:  # skipped because too many station in one place
                    print('station skipped', df2['numero'].iloc[xver])
                else:
                    ax.axvline(xvertical, color='black')
                    if (transect == 'SF_24.1' and s in ['AF12', 'AF13', 'AF14', 'AF15', 'AF16', 'AF17', 'AF18']):
                        print('station SF24.1 skipped')
                    else:
                        ax.text(xvertical - shift, y=np.max(yplot) + 1, s=s, color='k',
                                fontsize=fontsize - 4)  # WARNING : the xvertical-300 depends on the xaxis size!!! 300 is ok for T4
                xver = xver + 1

        outfile = 'current_magnitude_direction_stations_cmaptwilight' + transect + '_' + day + month + '.png'
        fig.savefig(outfile, format='png')
        # plt.show()




        sys.exit(1)

    elif var == 'profile':
        ###################################################################################################################
        # Parameters for the 2d plot
        if transect == 'T4' and month == '08':
            limvel = 1.2  # 0.6 #for values in m/s, every limvel, a new profile on the 3d plot
        else:
            limvel = 0.8
        length = 0.001  # for values in m/s, length of the arrow
        val = limvel / 2  # 0.3 #Value of the unit
        labelpad = 3  # distance between ticks and the axis

        if dim == '2d':
            ax = axs[1]
            # disp=' quiver plot '
            ax.set_ylabel('Depth (m)', fontsize=fontsize)
            ax.set_ylim(0, nb_bin + 30)  # +2 to see the arrows going up
            ax.set_yticks(y)
            ax.set_yticklabels(labels=depth2)  # depth[::int(nb_bin/2)])
            ax.set_xlabel('Time', fontsize=fontsize)
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_xticks(x)
            ax.set_xticklabels(labels=time, fontsize=fontsize - 4)
            if layer == 1:
                print('horizontal line on the graph layer=1')
                lineh2 = ax.axhline(lim1, xmin=0, xmax=1, color='k')
            elif layer == 2:
                lineh2 = ax.axhline(lim1, xmin=0, xmax=1, color='k')
                lineh3 = ax.axhline(lim2, xmin=0, xmax=1, color='k')
            # cbar = plt.colorbar(p2, label=disp, ax=ax )#, ticks=60)#ax=ax
            # cbar.ax.tick_params(labelsize=8)

        elif dim == '3d':
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #fig.set_size_inches(11, 5)
            ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.8, 0.8, 1]))
            # ax = fig.add_subplot(2, 1, 2, projection=dim )  # , sharex='row')#,sharey = ax1,sharez = ax1)
            ax.set_zlabel('Depth (m)', fontsize=fontsize - 3, labelpad=labelpad - 4)
            ax.set_zlim(0, 35)
            ax.set_zticks(y)
            ax.set_zticklabels(depth2, fontsize=fontsize - 4)  # depth[::int(nb_bin/2)]
            # ax.set_xlabel('Velocity East (m/s)', fontsize=fontsize-2, labelpad=labelpad)
            ax.set_xlabel('Profiles', fontsize=fontsize - 2, labelpad=labelpad)
            ax.set_xlim(-limvel, limvel * 2)
            ax.set_xlim(0, len(station2_CTD))
            x_absis = np.arange(0, (len(station2_CTD) + 0.1) * limvel, limvel)
            ax.set_xticks(x_absis)
            # ax.set_xticks(np.linspace(0, len(station2_CTD)*limvel, len(station2_CTD)))
            # ax.set_xticks(np.linspace(-limvel, len(station2_CTD)*limvel, len(station2_CTD)))
            # ax.set_xticks(np.arange(-limvel, len(station2_CTD) * limvel + 0.1, val))
            station2_CTD.append('')
            ax.set_xticklabels(station2_CTD)
            # ax.set_ylabel('Velocity North (m/s)', fontsize=fontsize-2, labelpad=labelpad-6)
            # ax.set_ylabel('', fontsize=fontsize-2, labelpad=labelpad-6)
            ax.set_ylim(-2 * limvel, 2 * limvel)
            # ax.set_yticks(np.arange(-limvel, limvel + 0.1, val))
            y_absis = np.linspace(-2 * limvel, 2 * limvel, 5)
            ax.set_yticks(y_absis)
            ax.set_yticklabels([])
            ax.tick_params(axis='x', pad=-1)
            # ax.set_ylim(-limvel, limvel)
            # ax.set_yticks(np.arange(-limvel, limvel + 0.1, val))

            # 3D parameters
            azim = -90
            dist = 7
            elev = 25
            ax.azim = azim
            ax.dist = dist
            ax.elev = elev

            if correct_dir_segment:
                # qNor = ax.quiver3D(-limvel, 0, 0, 0, val, 0, color='grey', arrow_length_ratio=0.3)
                # qEas = ax.quiver3D(-limvel, 0, 0, val, 0, 0, color='grey', arrow_length_ratio=0.3)
                # ax.text(-3.2 * val, 0.5 * val, 1, s=str(val) + '\nm/s lateral', fontsize=fontsize - 5)
                # ax.text(-3.2 * val, -0.9 * val, 0, s=str(val) + 'm/s to ocean', fontsize=fontsize - 5)  # -1.8*val
                qNor = ax.quiver3D(0, 0, 0, 0, limvel, 0, color='grey', arrow_length_ratio=0.3)
                qEas = ax.quiver3D(0, 0, 0, limvel, 0, 0, color='grey', arrow_length_ratio=0.3)
                ax.text(-3.2 * val, 0.5 * val, 1, s=str(limvel) + '\nm/s N', fontsize=fontsize - 5)
                ax.text(-3.2 * val, -0.9 * val, 0, s=str(limvel) + 'm/s E', fontsize=fontsize - 5)
                # plot échelle à station 0=-limvel/s
            else:
                #qNor = ax.quiver3D(-limvel, 0, 0, 0, val, 0, color='grey', arrow_length_ratio=0.3)
                #qEas = ax.quiver3D(-limvel, 0, 0, val, 0, 0, color='grey', arrow_length_ratio=0.3)
                #ax.text(-3.2 * val, 0.5 * val, 1, s=str(val) + '\nm/s N', fontsize=fontsize - 5)
                #ax.text(-3.2 * val, -0.9 * val, 0, s=str(val) + 'm/s E', fontsize=fontsize - 5)  # -1.8*val
                qNor = ax.quiver3D(0, 0, 0, 0, limvel, 0, color='grey', arrow_length_ratio=0.3)
                qEas = ax.quiver3D(0, 0, 0, limvel, 0, 0, color='grey', arrow_length_ratio=0.3)
                ax.text(-3.2 * val, 0.5 * val, 1, s=str(limvel) + '\nm/s N', fontsize=fontsize - 5)
                ax.text(-3.2 * val, -0.9 * val, 0, s=str(limvel) + 'm/s E', fontsize=fontsize - 5)  # -1.8*val
        n = 2
        for xplot in profilevertical_CTD:  # _CTD:
            xplot2 = np.ones(np.shape(yplot))
            xplot2 = xplot2 * xplot
            # print('xplot2', xplot2)
            # Define vx (east) and vy (north) in our case for each profile (x2)
            vx = (df.iloc[[xplot], 4:4 + nb_bin])  # extract the line of East Eas, mm/s # for mag 65:96])
            vx = vx.to_numpy()

            if deb == 1:
                vxmoy = df.iloc[int(xplot):int(xplot + moy),
                        4:4 + nb_bin]  # do the average on the 16seconds after, because we cannot center on the 1st value
            else:  # TODO : see if this condition is enough
                vxmoy = df.iloc[int(xplot - moy / 2):int(xplot + moy / 2),
                        4:4 + nb_bin]  # average velocity x over the moy in sec determined
            vxmoy = vxmoy.mean()  # TODO : check si ok avec nan ?
            vxmoy = vxmoy.to_numpy()

            vy = (df.iloc[[xplot], 4 + nb_bin:4 + 2 * (nb_bin)])  # for dir 96:127])
            vy = vy.to_numpy()

            if deb == 1:
                vymoy = df.iloc[int(xplot):int(xplot + moy), 4 + nb_bin:4 + 2 * (
                    nb_bin)]  # do the average on the 16seconds after, because we cannot center on the 1st value
            else:  # TODO : see if this condition is enough
                vymoy = df.iloc[int(xplot - moy / 2):int(xplot + moy / 2),
                        4 + nb_bin:4 + 2 * (nb_bin)]  # average velocity x over the moy in sec determined

            # vymoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 34:65]
            vymoy = vymoy.mean()  # TODO : check si ok avec nan ?
            vymoy = vymoy.to_numpy()
            # print('vxmoy', len(vxmoy), vxmoy)
            # sys.exit(1)
            # last_val=(~np.isnan(vx)).cumsum(1).argmax(1) #last value non nan : to remove. WORKS ONLY FOR VX, shape [[]]
            if vxmoy[-1] == np.nan:  # if the last value is != nan, no need of this loop
                last_val = np.argwhere(np.isnan(vxmoy))[0]  # find the last value different of nan
                print('nb bin ', nb_bin, last_val[0])
                for i in range(last_val[0] - 1, nb_bin - 1):
                    if vxmoy[i] != np.nan:
                        print('vxmoy diff nan', vxmoy[i])
                        vxmoy[i] = np.nan
                        vymoy[i] = np.nan
            else:
                last_val = nb_bin - 1
            # last_valy=(~np.isnan(vymoy)).cumsum(0).argmax(1) #last value non nan : to remove
            # print('vxmoy', vxmoy)
            # print('vymoy', vymoy)
            print('vx last before nan ', last_val)  # df[].last_valid_index() )
            if remove_last_val:
                vxmoy[last_val - 1] = np.nan
                vymoy[last_val - 1] = np.nan
                # print('vxmoy removed last', vxmoy)
                # print('vymoy removed last', vymoy)

            # vd=df.iloc[[xplot] ,4+5*nb_bin:4+6*nb_bin ] # 96:127] #DOESNT WORK : EMPTY
            # print('vd', vd)
            # vd=vd.to_numpy()

            vx = np.flip(vx)  # to have the data of the bottom in first lines
            vxmoy = np.flip(vxmoy)
            vy = np.flip(vy)
            vymoy = np.flip(vymoy)
            # vd=np.flip(vd)
            # print('vx, ' , vx)
            # print('vy, ' , vy)

            if layer == 2:
                # average by layer, excluding the nan values
                vymoylayer1 = np.nanmean(vymoy[0:lim1])  # bottom
                vymoylayer2 = np.nanmean(vymoy[lim1:lim2])  # intermediate
                vymoylayer3 = np.nanmean(vymoy[lim2:31])  # surface #TODO : why 31 ?
                vxmoylayer1 = np.nanmean(vxmoy[0:lim1])
                vxmoylayer2 = np.nanmean(vxmoy[lim1:lim2])
                vxmoylayer3 = np.nanmean(vxmoy[lim2:31])
                # print('vxmoylayer1', vxmoylayer1, vymoylayer1)
                # print('vxmoylayer2', vxmoylayer2, vymoylayer2)
                # print('vxmoylayer3', vxmoylayer3, vymoylayer3)
                # print('vxmoy', vxmoy, vymoy)
            elif layer == 1:
                # average by layer, excluding the nan values
                vymoylayer1 = np.nanmean(vymoy[0:lim1])  # bottom
                vymoylayer2 = np.nanmean(vymoy[lim1:-1])  # intermediate
                vxmoylayer1 = np.nanmean(vxmoy[0:lim1])
                vxmoylayer2 = np.nanmean(vxmoy[lim1:-1])
                # print('vxmoylayer1', vxmoylayer1, vymoylayer1)
                # print('vxmoylayer2', vxmoylayer2, vymoylayer2)
                # print('vxmoylayer3', vxmoylayer3, vymoylayer3)
                # print('vxmoy', vxmoy, vymoy)

            vx = np.nan_to_num(vx, nan=0)
            vxmoy = np.nan_to_num(vxmoy, nan=0)
            vy = np.nan_to_num(vy, nan=0)
            vymoy = np.nan_to_num(vymoy, nan=0)
            # vd=np.nan_to_num(vd, nan=0)

            if dim == '2d':
                if month == '06':
                    scale = 9
                elif month == '08':
                    scale = 25  # 15
                ax.scatter(xplot2, yplot, marker='x', s=10, color='k')
                ax.set_title(str(moy) + 'profiles averaged, ' + str(moy * 5) + 'sec')
                # q = ax.quiver(zplot+(n-1)*limvel, zplot, yplot, vxmoy, vymoy, zmoy, color='grey',  arrow_length_ratio=0.3,
                #              length=length)#, scale=9, scale_units='dots')
                q = ax.quiver(xplot2, yplot, vxmoy, vymoy, width=0.003, color='grey', scale=scale, scale_units='dots')
                if layer == 2:
                    qlayer1 = plt.quiver(xplot, 4, vxmoylayer1, vymoylayer1, width=0.005, color='green', scale=scale,
                                         scale_units='dots', zorder=3)
                    qlayer2 = plt.quiver(xplot, 12, vxmoylayer2, vymoylayer2, width=0.005, color='red', scale=scale,
                                         scale_units='dots', zorder=3)
                    qlayer3 = plt.quiver(xplot, 24, vxmoylayer3, vymoylayer3, width=0.005, color='blue', scale=scale,
                                         scale_units='dots', zorder=3)
                    avglen1 = np.around(np.sqrt(np.array(vxmoylayer1) ** 2 + np.array(vymoylayer1) ** 2).mean(), 2)
                    avglen2 = np.around(np.sqrt(np.array(vxmoylayer2) ** 2 + np.array(vymoylayer2) ** 2).mean(), 2)
                    avglen3 = np.around(np.sqrt(np.array(vxmoylayer3) ** 2 + np.array(vymoylayer3) ** 2).mean(), 2)
                    print('avglen', avglen1, avglen2, avglen3)
                    displen = 250
                    plt.quiverkey(qlayer1, posx[incr], posy[0], -1, '{}'.format(np.around(avglen1 / 1000, 2)),
                                  color='white',
                                  labelcolor='green')
                    plt.quiverkey(qlayer2, posx[incr], posy[1], 0, '{}'.format(np.around(avglen2 / 1000, 2)), color='white',
                                  labelcolor='red')
                    plt.quiverkey(qlayer3, posx[incr], posy[2], 0, '{}'.format(np.around(avglen3 / 1000, 2)), color='white',
                                  labelcolor='blue')
                elif layer == 1:
                    if month == '06':
                        xl1 = 40
                        xl2 = 95
                    elif transect == 'TA1':
                        xl1 = 110
                        xl2 = 150
                    qlayer1 = plt.quiver(xplot, xl1, vxmoylayer1, vymoylayer1, width=0.005, color='green', scale=scale,
                                         scale_units='dots', zorder=3)
                    qlayer2 = plt.quiver(xplot, xl2, vxmoylayer2, vymoylayer2, width=0.005, color='red', scale=scale,
                                         scale_units='dots', zorder=3)
                    avglen1 = np.around(np.sqrt(np.array(vxmoylayer1) ** 2 + np.array(vymoylayer1) ** 2).mean(), 2)
                    avglen2 = np.around(np.sqrt(np.array(vxmoylayer2) ** 2 + np.array(vymoylayer2) ** 2).mean(), 2)
                    print('avglen', avglen1, avglen2)
                    displen = 250
                    plt.quiverkey(qlayer1, posx[incr], posy[0], -1, '{}'.format(np.around(avglen1 / 1000, 2)),
                                  color='white',
                                  labelcolor='green')
                    plt.quiverkey(qlayer2, posx[incr], posy[1], 0, '{}'.format(np.around(avglen2 / 1000, 2)), color='white',
                                  labelcolor='red')
                # avglen = np.around(np.sqrt(np.array(vx) ** 2 + np.array(vy) ** 2).mean(),2)
                # displen = np.around(avglen, np.int(np.ceil(np.abs(np.log10(avglen)))))
                # displen=np.around(displen/1000,2)
                if incr == 0:
                    displen = 250
                    print('value incr for 2d plot', incr)
                    plt.quiverkey(q, 1.1, 0.05, displen, '{} (m/s)'.format(np.around(displen / 1000, 2)))
                # plt.quiverkey(q, 0.12*n, 0.05, displen, '{} (m/s)'.format(np.around(displen/1000,2)), labelcolor=labelcolor[incr])
                # ax.quiverkey(q, 0.2, 0.05, 0.25, 'velocity: {} [m/s]'.format(0.25))
                # n=n+1
                # pos = [-1, -1]

            elif dim == '3d':
                zplot = np.zeros(np.shape(yplot))
                zmoy = np.zeros(np.shape(vxmoy))  # no velocity on the z level
                # ax.scatter(zplot + (n - 1) * limvel, zplot, yplot, marker='x', s=10, color='k')
                ax.scatter(zplot + x_absis[n], zplot, yplot, marker='x', s=10, color='k')
                if color_arrow_dir:
                    # TODO doenst work like that

                    # test1
                    cNorm = mpl.colors.Normalize(vmin=0, vmax=360)
                    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap2)
                    colorVal = scalarMap.to_rgba(np.arctan2(vxmoy, vymoy))
                    # print('colorVal', colorVal)

                    # test2
                    colors = np.arctan2(vxmoy, vymoy)
                    norm = mpl.colors.Normalize()
                    norm.autoscale(colors)
                    colormap = plt.cm.hsv  # mpl.cm.inferno
                    color = colormap(norm(colors))
                    color = np.concatenate((color, color,
                                            color))  # np.repeat(color, 2))) #len(station_CTD)-1)) #3 elements in one arrow, so to have the same color==> 3*len

                    # test4
                    # Color by azimuthal angle
                    c = np.arctan2(vymoy, vxmoy)
                    print('c1', c, np.shape(c))
                    # Flatten and normalize
                    c = (c.ravel() - c.min()) / c.ptp()
                    print('c2', c)
                    # Repeat for each body line and two head lines
                    c = np.concatenate((c, np.repeat(c, 2)))  # len(station_CTD)-1)))
                    print('c3', np.shape(c))
                    # Colormap
                    c = plt.cm.hsv(c)
                    print('c4', c, np.shape(c))
                    print('min c ', np.min(c), np.max(c))

                    q = ax.quiver(zplot + (n - 1) * limvel, zplot, yplot, vxmoy, vymoy, zmoy, colors=color, cmap='hsv',
                                  # colors=c, #
                                  arrow_length_ratio=0.3,
                                  length=length)  # , alpha=1, zorder=1)  # , scale=9, scale_units='dots')
                    fig.colorbar('hsv', q)

                    # q.set_array(np.linspace(0, 360, 10))
                    # q.set_edgecolor(c)
                    # q.set_facecolor(c)
                    # plt.show()
                    # sys.exit(1)

                if correct_dir_segment:
                    print('station for correct dir',
                          s)  # je veux trouver la station correspondante pour connaitre le bon segment
                    # xplot est le numéro ==> trouver dans df2_station la station correspondant au numéro
                    sta_correctdir = df2['Stations'][
                        df2['numero'] == xplot].values  # find the name of the station by its numero
                    sta_correctdir = sta_correctdir[0]
                    print('sta_correctdir', sta_correctdir)
                    if sta_correctdir in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A26']:
                        angle = 144.1
                    elif sta_correctdir in ['A7', 'A27']:
                        angle = 110.5
                    elif sta_correctdir in ['A28', 'A29', 'A30', 'A8']:
                        angle = 69
                    elif sta_correctdir in ['A9']:
                        angle = 94.7
                    elif sta_correctdir in ['A10']:  # test, difficile car dans l'angle
                        angle = 164.3
                    elif sta_correctdir in ['A10', 'A11', 'A12', 'A31', 'A32']:
                        angle = 170  # 178.38
                    elif sta_correctdir in ['SF1', 'A33']:
                        angle = 126.72
                    elif sta_correctdir in ['A34', 'A35', 'A36']:
                        angle = 58.8
                    elif sta_correctdir in ['A37']:
                        angle = 110.6
                    elif sta_correctdir in ['A38']:
                        angle = 157.1
                    elif sta_correctdir in ['A39', 'A40']:
                        angle = 129.7
                    else:
                        print('PB ANGLE')

                    angle = (angle - 90) % 360
                    print('angle', angle)

                    # print('signe vymoy', np.sign(vymoy))
                    # vy_dir=np.linalg.norm([vxmoy,vymoy], axis=0)*np.sign(vymoy) # On obtient la norme (vx,vy) avec le signe de vxmoy
                    # print('vy_dir', vy_dir)
                    # vx_dir=vymoy*np.tan(angle*np.pi/180)
                    # print('vxdir', vx_dir)

                    print('signe vxmoy', np.sign(vxmoy))
                    vx_dir = np.linalg.norm([vxmoy, vymoy], axis=0) * np.sign(
                        vxmoy)  # On obtient la norme (vx,vy) avec le signe de vxmoy
                    print('vx_dir', vx_dir)
                    vy_dir = vx_dir * np.arctan(np.tan(vymoy / vxmoy) + angle * np.pi / 180)
                    # vy_dir=vxmoy*np.tan(angle*np.pi/180)
                    print('vydir', vy_dir)

                    q = ax.quiver(zplot + (n - 1) * limvel, zplot, yplot, vx_dir, vy_dir, zmoy, color='grey',
                                  arrow_length_ratio=0.3,
                                  length=length, alpha=0.3, zorder=1)  # , scale=9, scale_units='dots')
                    # vx_ex=np.random.randint(-600,600, np.shape(vx_dir))
                    # q = ax.quiver(zplot + (n - 1) * limvel, zplot, yplot, vx_ex,
                    #              600 * np.ones(np.shape(vx_dir)), zmoy, color='grey', arrow_length_ratio=0.3,
                    #              length=length, alpha=0.3, zorder=1)  # , scale=9, scale_units='dots')
                    # q = ax.quiver(zplot+(n-1)*limvel, zplot, yplot,  np.zeros(np.shape(vy_dir)), 600*np.ones(np.shape(vx_dir)), zmoy, color='grey',  arrow_length_ratio=0.3,
                    #              length=length, alpha=0.3, zorder=1)#, scale=9, scale_units='dots')
                else:
                    print('vxmoy', vxmoy)
                    print('vymoy', vymoy)
                    print(np.linalg.norm([vxmoy, vymoy], axis=0))

                    #q = ax.quiver(zplot + (n - 1) * limvel, zplot, yplot, vxmoy, vymoy, zmoy, color='grey',
                    #              arrow_length_ratio=0.3,
                    #              length=length, alpha=0.3, zorder=1)  # , scale=9, scale_units='dots')
                    q = ax.quiver(zplot + x_absis[n], zplot, yplot, vxmoy, vymoy, zmoy, color='grey',
                                  arrow_length_ratio=0.3,
                                  length=length, alpha=0.3, zorder=1)  # , scale=9, scale_units='dots')
                if layer == 2:
                    qlayer1 = ax.quiver(zplot + (n - 1) * limvel, zplot, 4, vxmoylayer1, vymoylayer1, 0, color='green',
                                        arrow_length_ratio=0.3, length=length,
                                        zorder=3)  # , width=0.005,  scale=9, scale_units='dots')
                    qlayer2 = ax.quiver(zplot + (n - 1) * limvel, zplot, 12, vxmoylayer2, vymoylayer2, 0, color='red',
                                        arrow_length_ratio=0.3, length=length,
                                        zorder=3)  # ,width=0.005, , scale=9, scale_units='dots')
                    qlayer3 = ax.quiver(zplot + (n - 1) * limvel, zplot, 24, vxmoylayer3, vymoylayer3, 0, color='blue',
                                        arrow_length_ratio=0.3, length=length, zorder=3,
                                        linestyle='solid')  # , width=0.005, scale=9, scale_units='dots')
                    # ax.scatter(xplot2, zplot, yplot, marker='x', s=10, color='k')
                    # q = ax.quiver(xplot2, zplot, yplot, vxmoy, vymoy, zmoy, color='grey', length=0.1, arrow_length_ratio=0.1 )#, scale=9, scale_units='dots')
                    # qlayer1 = ax.quiver(xplot, zplot, 4,  vxmoylayer1, vymoylayer1, 0, color='green', length=0.1)#, width=0.005,  scale=9, scale_units='dots')
                    # qlayer2 = ax.quiver(xplot, zplot, 12, vxmoylayer2, vymoylayer2, 0, color='red',  length=0.1)#width=0.005, , scale=9, scale_units='dots')
                    # qlayer3 = ax.quiver(xplot, zplot, 24, vxmoylayer3, vymoylayer3, 0, color='blue',  length=0.1)#, width=0.005, scale=9, scale_units='dots')
                    avglen1 = np.around(np.sqrt(np.array(vxmoylayer1) ** 2 + np.array(vymoylayer1) ** 2).mean(), 2)
                    avglen2 = np.around(np.sqrt(np.array(vxmoylayer2) ** 2 + np.array(vymoylayer2) ** 2).mean(), 2)
                    avglen3 = np.around(np.sqrt(np.array(vxmoylayer3) ** 2 + np.array(vymoylayer3) ** 2).mean(), 2)
                    # print('avglen', avglen1, avglen2, avglen3)
                    # avglen = np.around(np.sqrt(np.array(vx) ** 2 + np.array(vy) ** 2).mean(),2)
                    # displen = np.around(avglen, np.int(np.ceil(np.abs(np.log10(avglen)))))
                    # displen = 250
                    # displen=np.around(displen/1000,2)
                    ax.text(-0.5 + limvel * n, -0.6, 1, '{}'.format(np.around(avglen1 / 1000, 2)), color='green',
                            fontsize=fontsize - 4)
                    ax.text(-0.5 + limvel * n, -0.4, 1, '{}'.format(np.around(avglen2 / 1000, 2)), color='red',
                            fontsize=fontsize - 4)
                    ax.text(-0.5 + limvel * n, -0.2, 1, '{}'.format(np.around(avglen3 / 1000, 2)), color='blue',
                            fontsize=fontsize - 4)
                elif layer == 1:
                    qlayer1 = ax.quiver(zplot + (n - 1) * limvel, zplot, 90, vxmoylayer1, vymoylayer1, 0, color='green',
                                        arrow_length_ratio=0.3, length=length, zorder=3,
                                        lw=5)  # , width=0.005,  scale=9, scale_units='dots')
                    qlayer2 = ax.quiver(zplot + (n - 1) * limvel, zplot, int((nb_bin + lim1) / 2), vxmoylayer2, vymoylayer2,
                                        0, color='red',
                                        arrow_length_ratio=0.3, length=length, zorder=3,
                                        lw=5)  # ,width=0.005, , scale=9, scale_units='dots')
                    avglen1 = np.around(np.sqrt(np.array(vxmoylayer1) ** 2 + np.array(vymoylayer1) ** 2).mean(), 2)
                    avglen2 = np.around(np.sqrt(np.array(vxmoylayer2) ** 2 + np.array(vymoylayer2) ** 2).mean(), 2)
                    ax.text(-1.3 + limvel * n, -0.8, 1, '{}'.format(np.around(avglen1 / 1000, 2)), color='green',
                            fontsize=fontsize - 4)  # 0.5+limvel*n
                    ax.text(-1.3 + limvel * n, -0.4, 1, '{}'.format(np.around(avglen2 / 1000, 2)), color='red',
                            fontsize=fontsize - 4)  # -0.5 + limvel * n

                # if sta=='random':
                #     if incr==2 or incr==3 :
                #         ax.text(posx3dgreen[incr], -0.25, 1 , '{}'.format(np.around(avglen1 / 1000, 2)), color='green', fontsize=fontsize-4)
                #         ax.text(posx3dred[incr], 0, 22, '{}'.format(np.around(avglen2 / 1000, 2)), color='red', fontsize=fontsize-4)
                #         ax.text(posx3dblue[incr], 0.2, 31 , '{}'.format(np.around(avglen3 / 1000, 2)), color='blue', fontsize=fontsize-4 )
                #     else :
                #         ax.text(posx3dgreen[incr], -0.25, 1 , '{}'.format(np.around(avglen1 / 1000, 2)), color='green', fontsize=fontsize-4)
                #         ax.text(posx3dred[incr], 0, 20, '{}'.format(np.around(avglen2 / 1000, 2)), color='red', fontsize=fontsize-4)
                #         ax.text(posx3dblue[incr], 0.2, 31 , '{}'.format(np.around(avglen3 / 1000, 2)), color='blue', fontsize=fontsize-4 )

                n = n + 2
                # n = n+1

            else:
                print('ERROR in the dim')

            incr = incr + 1
        # if dim == '3d':
        #    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.9,  wspace=0.4, hspace=0.4)
        # plt.show()

        if No_save:
            print('PAS D ENREGISTREMENT')
            sys.exit(1)

        else:
            outfile = 'ADCP_profiles_velocities_'
            if layer == 1:
                outfile = outfile + '1layer_'
            elif layer == 2:
                outfile = outfile + '2layers_'
            if color_arrow_dir:
                outfile = outfile + 'coloredarrowdir_'
            if correct_dir_segment:
                outfile = outfile + 'correctdirsegment_'
            outfile = outfile + transect + '_' + day + month + '_' + dim + '_' + str(moy) + 'profilesavg.png'
            print('outfile', outfile)

            # fig.savefig('test_1808_filter5_profile.png')
            fig.savefig(outfile, format='png')
print('ok')
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.3, 0.8, 0.8, 1]))
# ax = fig.add_subplot(2, 1, 2, projection=dim )  # , sharex='row')#,sharey = ax1,sharez = ax1)
ax.set_zlabel('Depth (m)', fontsize=fontsize - 3, labelpad=labelpad - 4)
ax.set_zlim(0, 35)
ax.set_zticks(y)
ax.set_zticklabels(depth2, fontsize=fontsize - 4)  # depth[::int(nb_bin/2)]
# ax.set_xlabel('Velocity East (m/s)', fontsize=fontsize-2, labelpad=labelpad)
ax.set_xlabel('Profiles', fontsize=fontsize - 2, labelpad=labelpad)
ax.set_xlim(-limvel, limvel * 2)
# ax.set_xlim(0,len(station2_CTD))
ax.set_xticks(np.linspace(0, len(station2_CTD) * limvel, len(station2_CTD)))
# ax.set_xticks(np.linspace(-limvel, len(station2_CTD)*limvel, len(station2_CTD)))
# ax.set_xticks(np.arange(-limvel, len(station2_CTD) * limvel + 0.1, val))
ax.set_xticklabels(station2_CTD)
# ax.set_ylabel('Velocity North (m/s)', fontsize=fontsize-2, labelpad=labelpad-6)
# ax.set_ylabel('', fontsize=fontsize-2, labelpad=labelpad-6)
ax.set_ylim(-limvel, limvel)
ax.set_yticks(np.arange(-limvel, limvel + 0.1, val))
ax.set_yticklabels([])
ax.tick_params(axis='x', pad=-1)

# Bout de code pour faire figures rapidement :
azim = -90
dist = 7
elev = 25
ax.azim = azim
ax.dist = dist
ax.elev = elev

qNor = ax.quiver3D(-limvel, 0, 0, 0, val, 0, color='grey', arrow_length_ratio=0.3)
qEas = ax.quiver3D(-limvel, 0, 0, val, 0, 0, color='grey', arrow_length_ratio=0.3)
ax.text(-3.2 * val, 0.5 * val, 1, s=str(val) + '\nm/s N', fontsize=fontsize - 5)
ax.text(-3.2 * val, -0.9 * val, 0, s=str(val) + 'm/s E', fontsize=fontsize - 5)  # -1.8*val

for xplot in profilevertical_CTD:  # _CTD:
    xplot2 = np.ones(np.shape(yplot))
    xplot2 = xplot2 * xplot
    # print('xplot2', xplot2)
    # Define vx (east) and vy (north) in our case for each profile (x2)
    vx = (df.iloc[[xplot], 4:4 + nb_bin])  # extract the line of East Eas, mm/s # for mag 65:96])
    vx = vx.to_numpy()

    if deb == 1:
        vxmoy = df.iloc[int(xplot):int(xplot + moy),
                4:4 + nb_bin]  # do the average on the 16seconds after, because we cannot center on the 1st value
    else:  # TODO : see if this condition is enough
        vxmoy = df.iloc[int(xplot - moy / 2):int(xplot + moy / 2),
                4:4 + nb_bin]  # average velocity x over the moy in sec determined
    vxmoy = vxmoy.mean()  # TODO : check si ok avec nan ?
    vxmoy = vxmoy.to_numpy()

    vy = (df.iloc[[xplot], 4 + nb_bin:4 + 2 * (nb_bin)])  # for dir 96:127])
    vy = vy.to_numpy()

    if deb == 1:
        vymoy = df.iloc[int(xplot):int(xplot + moy), 4 + nb_bin:4 + 2 * (
            nb_bin)]  # do the average on the 16seconds after, because we cannot center on the 1st value
    else:  # TODO : see if this condition is enough
        vymoy = df.iloc[int(xplot - moy / 2):int(xplot + moy / 2),
                4 + nb_bin:4 + 2 * (nb_bin)]  # average velocity x over the moy in sec determined

    # vymoy=df.iloc[int(xplot-moy/2):int(xplot+moy/2), 34:65]
    vymoy = vymoy.mean()  # TODO : check si ok avec nan ?
    vymoy = vymoy.to_numpy()
    # print('vxmoy', len(vxmoy), vxmoy)
    # sys.exit(1)
    # last_val=(~np.isnan(vx)).cumsum(1).argmax(1) #last value non nan : to remove. WORKS ONLY FOR VX, shape [[]]
    if vxmoy[-1] == np.nan:  # if the last value is != nan, no need of this loop
        last_val = np.argwhere(np.isnan(vxmoy))[0]  # find the last value different of nan
        print('nb bin ', nb_bin, last_val[0])
        for i in range(last_val[0] - 1, nb_bin - 1):
            if vxmoy[i] != np.nan:
                print('vxmoy diff nan', vxmoy[i])
                vxmoy[i] = np.nan
                vymoy[i] = np.nan
    else:
        last_val = nb_bin - 1
    # last_valy=(~np.isnan(vymoy)).cumsum(0).argmax(1) #last value non nan : to remove
    # print('vxmoy', vxmoy)
    # print('vymoy', vymoy)
    print('vx last before nan ', last_val)  # df[].last_valid_index() )
    if remove_last_val:
        vxmoy[last_val - 1] = np.nan
        vymoy[last_val - 1] = np.nan

    vx = np.flip(vx)  # to have the data of the bottom in first lines
    vxmoy = np.flip(vxmoy)
    vy = np.flip(vy)
    vymoy = np.flip(vymoy)
    # vd=np.flip(vd)
    # print('vx, ' , vx)
    # print('vy, ' , vy)

    vx = np.nan_to_num(vx, nan=0)
    vxmoy = np.nan_to_num(vxmoy, nan=0)
    vy = np.nan_to_num(vy, nan=0)
    vymoy = np.nan_to_num(vymoy, nan=0)
    # vd=np.nan_to_num(vd, nan=0)

    zplot = np.zeros(np.shape(yplot))
    zmoy = np.zeros(np.shape(vxmoy))  # no velocity on the z level
    ax.scatter(zplot + (n - 1) * limvel, zplot, yplot, marker='x', s=10, color='k')
    print('vxmoy', vxmoy)
    print('vymoy', vymoy)
    print(np.linalg.norm([vxmoy, vymoy], axis=0))

    q = ax.quiver(zplot + (n - 1) * limvel, zplot, yplot, vxmoy, vymoy, zmoy, color='grey',
                  arrow_length_ratio=0.3,
                  length=length, alpha=0.3, zorder=1)  # , scale=9, scale_units='dots')
    n = n + 1
    incr = incr + 1
"""
