# 27/07/2022 Treat ADCP DATA
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')
import math
# from adcploader import *
from mpl_toolkits import mplot3d

# LIST OF variables : station, station2 same but with '' inbetween to plot on the 2d graph, profilevertical : the number of the rows of the table

month = '08'  # TO COMPLETE
year = '2022'
No_save = False

if month == '06':
    survey = 'June'
    transect = 'T4'  # T1 to T4 #TO COMPLETE
    depth_bin = 0.3
    blank_1st_bin = 0.6
    if transect == 'T1' or transect == 'T2':
        day = '16'
        T = ''
    elif transect == 'T3' or transect == 'T4':
        day = '17'
        T = '_' + transect
    else:
        print('problem with the transect')
    file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + T + '_alldata_BT.csv'

elif month == '08':
    survey = 'August'
    transect = 'SF1'  # TA1 to TA4, with SF1 and SF_24.1 ou .2 for fixed stations
    depth_bin = 0.1
    blank_1st_bin = 0.66
    if transect == 'TA1':
        type = 'transect'
        tide = 'Mid Flood'
        nfile = 1
        day = '10'
        hour_deb = '10h53'
        hour_end = '12h24'
        shift = 25
    elif transect == 'TA2':
        type = 'transect'
        tide = 'Late Flood'
        nfile = 1
        day = '10'
        hour_deb = '10h53'
        hour_end = '12h24'
        hour_deb = '12h41'
        hour_end = '14h35'
        shift = 10
    elif transect == 'SF1':
        type = 'fixed'
        tide = 'HT-ME'
        nfile = 1
        day = '10'
        hour_deb = '14h36'
        hour_end = '20h14'
        shift = 50
        # First values of current of the fixed station on the previous file
    elif transect == 'TA3':
        type = 'transect'
        tide = 'HT'
        nfile = 1
        day = '11'
        hour_deb = '15h01'
        hour_end = '18h31'
        shift = 65
    elif transect == 'TA4':
        type = 'transect'
        tide = 'Early Flood'
        nfile = 1
        day = '12'
        hour_deb = '06h46'
        hour_end = '07h06'
        shift = 3
        # first minutes on another file
    elif transect == 'SF_24.1':
        type = 'fixed'
        tide = 'cycle MF-EE'
        nfile = 1
        day = '12'
        hour_deb = '11h48'
        hour_end = '19h49'
        shift = 10
    elif transect == 'SF_24.2':
        type = 'fixed'
        tide = 'cycle ME-MF'
        nfile = 1
        day = '12'
        hour_deb = '19h50'
        hour_end = '08h55'
        shift = 10
    else:
        print('PROBLEM T>RANSECT august')
        sys.exit(1)

    if nfile == 1:
        file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_' + hour_deb + '-' + hour_end + '_BT.csv'
    elif nfile == 2:
        file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_' + hour_deb + '-' + hour_end + '_BT.csv'
        # file2 = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_'+hour_deb2+'-'+hour_end2+ '_BT.csv'

elif month == '10':
    survey = 'Octobre'
    transect = 'TO1'  # TO1 to TO4, and FO fixed stations
    depth_bin = 0.1
    blank_1st_bin = 0.66
    if transect == 'TO1':
        type = 'transect'
        tide = 'Late Ebb'
        nfile = 1
        day = '02'
        hour_deb = '17h19'
        hour_end = '18h28'
        shift = 25  # A voir
    elif transect == 'TO2':
        type = 'transect'
        tide = 'Early Ebb'
        nfile = 1
        day = '03'
        hour_deb = '10h20'
        hour_end = '18h00'
        shift = 10  # a voir
    elif transect == 'TO3':  # pareil, voir comment faire ?
        type = 'transect'
        tide = 'Late Ebb'
        nfile = 1
        day = '03'
        hour_deb = '10h20'
        hour_end = '18h00'
        shift = 50  # a voir
        # First values of current of the fixed station on the previous file
    elif transect == 'TO4':
        type = 'transect'
        tide = 'Mid Ebb'
        nfile = 1
        day = '04'
        hour_deb = '07h40'
        hour_end = '08h24'
        shift = 65
    elif transect == 'FO':  # fichier sépareé en 2, voir si tout mettre ne 1 ? en changeant les valeurs des ensembles ?
        type = 'transect'
        tide = '1 cycle'
        nfile = 1
        day = '04'
        shift = 3
    else:
        print('PROBLEM TRANSECT octobre')

    if transect == 'TO2' or transect == 'TO3':
        transect2 = 'TO2_TO3'
    else:
        transect2 = transect

    if nfile == 1:
        file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + month + day + '_' + transect2 + '_' + hour_deb + '_' + hour_end + '.csv'
    elif nfile == 2:
        file = '/home/penicaud/Documents/Data/ADCP/Survey_' + survey + '/' + day + month + '_' + hour_deb + '-' + hour_end + '_BT.csv'

else:
    print("PROBLEM WITH THE MONTH")

file_recap = '/home/penicaud/Documents/Data/Survey_'+survey+'/Recap_all_param_'+survey+'.xlsx'

print(file)

#############################            PARAMETER TO FILL         ###############################"
var = 'profile'  # 'dir or profile #TO COMPLETE
dim = '3d'  # or '3d' #TO COMPLETE
if var == 'dir':
    dim = '2d'

layer = 0  # 0 1 or 2 depending on the number of layer we want to show #Bool to add the averaged velocity of the layers, need to define the lim of the layers
remove_last_val = 1  # remove the last bin value
color_arrow_dir = 0  # 0 or 1, if 1 add a color to the profile 3D arrows to have a better idea of the direction
correct_dir_segment = 1  # 0 or 1, Change the north south orientation of 3D profile into the "along segment" direction, to not take into account the meanders

if month == '06':  # fq=1hz
    nb_bin = 45  # TO COMPLETE

    moy = 16  # EVEN NUMBER #nb of PROFILES (and because 1hz, of secondes also !) we want to average.
    # 16 for the june survey where 1 measure per second, august, 1 measure every 5 seconds
    # Limit of the layers if layer=true
    lim1 = 7  # nb of the limit bin between layer 1 and 2
    lim2 = 17  # nb of the limit bin between layer 2 and 3 #FOR 1806 : 17 contain every intense bins, 15 around 80 % (a vue de nez)
if month == '08':  # fq=0.2Hz
    moy = 6  # WARNING : fq=0.2HZ, it is the nb of PROFILES we average, and moy*5 is the period
    nb_bin = 180
    if transect == 'SF1':
        lim1 = 115
    elif transect == 'TA1':
        lim1 = 120  # WARNING : should be a mooving part, because thin layer not horizontal
if month == '10':  # fq=0.2Hz
    moy = 6  # WARNING : fq=0.2HZ, it is the nb of PROFILES we average, and moy*5 is the period
    nb_bin = 120

CTD_to_observe = 'LOG'  # 'LOG' or 'IMER'

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

data = pd.read_csv(file, skiprows=11, low_memory=False, sep=',')  # , usecols=col_list)
df = pd.DataFrame(data)
hour = df["HH"]
mn = df["MM"]
sec = df["SS"]
ens = df["Ens"]

mag, dir = [], []
for i in range(1, nb_bin):
    string1 = "Mag, mm/s, " + str(i)
    string2 = "Dir, deg, " + str(i)
    mag.append(string1)
    dir.append(string2)

# manip on the data to have the values of the first bins on the top of the figure
magnitude = pd.DataFrame(data, columns=mag)
magnitude2 = magnitude.transpose()
magnitude2 = magnitude2.iloc[::-1]
direction = pd.DataFrame(data, columns=dir)
direction2 = direction.transpose()
direction2 = direction2.iloc[::-1]

if nfile == 2:
    data_file2 = pd.read_csv(file2, skiprows=11, low_memory=False, sep=',', usecols=col_list)
    df_file2 = pd.DataFrame(data_file2)
    hour_file2 = df_file2["HH"]
    mn_file2 = df_file2["MM"]
    sec_file2 = df_file2["SS"]
    ens_file2 = df_file2["Ens"]

    magnitude_file2 = pd.DataFrame(data_file2, columns=mag)
    magnitude2_file2 = magnitude_file2.transpose()
    magnitude2_file2 = magnitude2_file2.iloc[::-1]
    direction_file2 = pd.DataFrame(data_file2, columns=dir)
    direction2_file2 = direction_file2.transpose()
    direction2_file2 = direction2_file2.iloc[::-1]

######################################################################################
# DF of the stations of the transects

if survey == 'June':
    for i in range(1, 40):
        string = "S" + str(i)
        sta.append(string)

    df_sta = pd.DataFrame({'stations': sta,
                           'hour': ['13.09', '13.32', '13.56', '14.23', '14.44', '15.19', '15.42', '16.02', '16.22',
                                    '16.35',
                                    '17.05', '18.00', '18.28', '18.52', '19.20', '19.35', '19.48', '12.22', '12.32',
                                    '12.40',
                                    '13.00', '13.11', '13.23', '13.40', '13.56', '14.14', '14.31', '14.48', '20.31',
                                    '20.47',
                                    '21.05', '21.20', '21.34', '21.42', '21.52', '22.00', '22.22', '22.35', '22.58']
                           # bidouillage for the 1st station of the transect T1 : 13.08.00 doesn't exist, so 13.09
                           # same for T3 : 12.16 is the right hour of station, but only ADCP from 12.22
                           })

    df_sta = pd.read_excel()

    if transect == 'T1':
        df2 = df_sta[0:10].copy()
        label1 = 'River'  # label to localize in space
        label2 = 'Ocean'
    elif transect == 'T2':
        df2 = df_sta[10:17].copy()
        label2 = 'River'
        label1 = 'Ocean'
    elif transect == 'T3':
        df2 = df_sta[17:28].copy()
        label2 = 'River'
        label1 = 'Ocean'
    elif transect == 'T4':
        df2 = df_sta[28:39].copy()
        label2 = 'River'
        label1 = 'Ocean'

elif survey == 'August':
    # transect 1 : A1 to A5
    # transect 2 : A6 to A11
    # Fixed station A12 to A25
    sta.append('A1.a')
    sta.append('A1.b')
    for i in range(3, 48):
        string = "A" + str(i)
        sta.append(string)
    for i in range(1, 39):
        string = "AF" + str(i)
        sta.append(string)

    df_sta = pd.DataFrame({'stations': sta,
                           'hour': ['11.04', '11.32', '11.38', '11.48', '12.01', '12.21', '12.39', '12.56', '13.02',
                                    '13.08',
                                    '13.15', '13.22', '14.21', '14.32', '14.50', '15.03', '15.23', '15.50', '16.37',
                                    '17.03',
                                    '17.36', '18.06', '18.17', '19.00', '19.18', '15.05', '15.17', '15.23', '15.32',
                                    # 19.35 not put
                                    '15.38', '15.45', '15.55', '16.03', '16.08', '16.15', '16.23', '16.30', '16.41',
                                    '16,50',
                                    '17.04', '06.47', '06.56', '07.07', '07.16', '07.25', '07.41', '07.51', '09.30',
                                    '10.05',
                                    '10.35', '11.00', '11.30', '12.00', '12.30', '13.00', '13.30', '14.00', '14.30',
                                    '15.00', '15.05',
                                    '15.10', '15.16', '15.22', '15.27', '15.40', '16.00', '17.00', '17.30', '18.57',
                                    '20.00',
                                    '21.17', '22.50', '23.30', '00.00', '00.30', '01.00', '01.45', '02.30', '03.15',
                                    '04.15',
                                    '05.15', '06.15', '07.05', '08.00', '08.55']
                           })

    if transect == 'TA1':
        df2 = df_sta[0:5].copy()
        label1 = 'River'  # label to localize in space
        label2 = 'Ocean'
    elif transect == 'TA2':
        df2 = df_sta[5:12].copy()
        label2 = 'River'
        label1 = 'Ocean'
    elif transect == 'SF1':
        df2 = df_sta[12:25].copy()
    elif transect == 'TA3':
        df2 = df_sta[25:40].copy()
        label2 = 'River'
        label1 = 'Ocean'
    elif transect == 'TA4':
        df2 = df_sta[40:47].copy()
        label1 = 'River'
        label2 = 'Ocean'
    elif transect == 'SF_24.1' or 'SF_24.2':
        df2 = df_sta[47:86].copy()

elif survey == 'Octobre':
    for i in range(1, 23):
        string = "O" + str(i)
        sta.append(string)
    sta.append("O22.2")
    for i in range(23, 36):
        string = "O" + str(i)
        sta.append(string)
    for i in range(1, 52):
        string = "FO" + str(i)
        sta.append(string)
    print(sta)
    print('len sta', len(sta))

    df_sta = pd.DataFrame({'stations': sta,
                           'hour': ['17.23', '17.40', '17.50', '18.00', '18.08', '18.17', '18.26', '09.50', '10.03',
                                    '10.10',
                                    '10.18', '10.27', '10.37', '10.45', '11.00', '11.06', '11.12', '11.20', '11.27',
                                    '11.37',
                                    '11.50', '15.10', '15.17', '15.27', '15.40', '15.50', '16.05', '16.15', '16.31',
                                    '16.42', '07.42', '07.53', '08.06', '08.18', '08.44', '08.54', '09.18', '09.30',
                                    '10.00:', '10.30', '11.00', '11.30', '12.00', '12.30', '13.00', '13.30',
                                    '14.00', '14.30', '15.00', '15.30', '16.00', '16.30', '17.00', '17.30', '18.00',
                                    '18.30', '19.00', '19.30', '20.00', '20.30', '21.00', '21.30', '22.00', '22.30',
                                    '23.00', '23.30', '00.00', '00.30', '01.00', '01.30', '02.00', '02.30', '03.00',
                                    '03.30', '04.00', '04.30', '05.00', '05.30', '06.00', '06.30', '07.00', '07.30',
                                    '08.00', '08.30', '09.00', '09.10', '09.30']})

    if transect == 'TO1':
        df2 = df_sta[0:7].copy()
        label1 = 'River'  # label to localize in space
        label2 = 'Ocean'
    elif transect == 'TO2':
        df2 = df_sta[8:21].copy()
        label1 = 'Ocean'
        label2 = 'River'
    elif transect == 'TO3':
        df2 = df_sta[22:29].copy()
        label1 = 'River'  # label to localize in space
        label2 = 'Ocean'
    elif transect == 'TO4':
        df2 = df_sta[30:35].copy()
        label1 = 'Ocean'
        label2 = 'River'
    elif transect == 'FO':
        df2 = df_sta[36:51].copy()

print(df2)
station = df2['stations']
hour_station = df2['hour']

# Add the numero of the ensemble of each hour
num = []
# print('hour deb, hour end ', int(hour_deb[0:2]), 'h',int(hour_deb[3:5]), int(hour_end[0:2]), 'h', int(hour_end[3:5]))
# print('hour deb2, hour end ', int(hour_deb2[0:2]), int(hour_deb2[3:5]), int(hour_end2[0:2]), int(hour_end2[3:5]))
for h in hour_station:
    hour_sta = int(h[0:2])
    mn_sta = int(h[3:5])
    print('hour sta', hour_sta, mn_sta)
    # sec_sta = 30 #sec= 30 for file where 1measure/sec, the first measure corresponding to HH and MM
    # if hour_sta>=int(hour_deb[0:2]) and hour_sta<=int(hour_end[0:2]) mn_sta>=int(hour_deb[0:2]):
    condition1 = (hour_sta == int(hour_deb[0:2]) and mn_sta >= int(hour_deb[3:5])) or (
                hour_sta == int(hour_end[0:2]) and mn_sta <= int(hour_end[3:5])) or (
                             hour_sta > int(hour_deb[0:2]) and hour_sta < int(hour_end[0:2]))
    if condition1:  # TODO : attention si jamais sur station 24h
        print('case1')
        condition = ((df.HH == hour_sta) & (df.MM == mn_sta))  # & (df.SS == sec_sta)) #
        l = df.index[
            condition]  # gives the index of the several measures at the time HH MM (for august, one measure every 5s)
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
    station_CTD = df2_station['stations']
    station2_CTD = station_CTD.copy()
    station2_CTD = station2_CTD.values.tolist()  # from df or series to list
    station2_CTD.insert(0, '')
    i = 1
    while i < 2 * len(station_CTD):  # need this trick only for 3D, to plot the vertical profiles on the values of
        station2_CTD.insert(i, '')
        i += 2
    profilevertical_CTD = df2_station['numero']

print('df2 station', df2[df2.numero != -10])

cmap = plt.cm.jet
cmap2 = plt.cm.plasma
cmap2 = plt.cm.twilight
fontsize = 12

# print((df.iloc[-1]))#==24186) #gives a serie
# print(df[df.columns[0]].count()) #Warning : if the file doesnt begin by 0 or 1, doesn't work
# print(ens.iloc[-1]) #THE GOOD SOLUTION

# X axis
# x=np.linspace(1,df2.at[len(df2['stations'])-1,'numero'],len(df2['stations']), dtype=int)
# TODO : what if first or 2d values are -10

incr = 0  # TODO Check if it works everytime or just if it depends on the number of values in the 1st file
if nfile == 1:
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

if transect == 'SF_24.1' or transect == 'SF_24.2':
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
        time.append(h + ':' + m ) # 24/07 comment the part + ':' + s)
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
depth2 = depth[::int(nb_bin / 2)]
print('depth2', depth2)  # better to use this one, the other is not right every time
print('depth', depth[::int(nb_bin / 2)])
yplot = np.arange(1, nb_bin + 1, 1)
# print('yplot', yplot)
# parameter for figures
n = 1  # indicator for quiver plot legend
labelcolor = 'black'
incr = 0
print('deoth2', depth2)

# TODO : attention on peut etre ne 2D avec des stations
if dim == '2d':
    fig, axs = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
    ax = axs[0]
    posx = [0.09, 0.25, 0.45, 0.65, 0.85, 0.97]  # position for the 2d graph for the legend of the averaged layer
    posy = [0.05, 0.28, 0.65]
    profilevertical = df2['numero']
elif dim == '3d':
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)  # TODO : 2 plots différents pour les aligner
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
fig.suptitle(day + '/' + month + ', ' + transect + ' ' + tide)
disp = 'Magnitude (mm/s)'
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
# ax.set_xlim(0, np.max(x) + 1)
# ax.set_xticks(ticks=x)  # , labels=stations)
# ax.set_xticklabels(stations)  # , minor=True)
if transect == 'T4' or transect == 'TA4' or transect == 'SF_24.2':
    vmax = 1750
elif transect == 'TA2' or transect == 'TA3' or transect == 'SF1':
    vmax = 1250
# elif transect=='SF_24.1':
#    vmax=900
else:
    vmax = 1000
p1 = ax.pcolormesh(magnitude2, cmap=cmap, vmin=0, vmax=vmax)
cbar = plt.colorbar(p1, label=disp, ax=ax, extend='max')  # , ticks=1)#ax=ax
cbar.ax.tick_params(labelsize=8)
if layer == 1:
    print('horizontal line on the graph layer=1')
    lineh2 = ax.axhline(lim1, xmin=0, xmax=1, color='k')
elif layer == 2:
    lineh2 = ax.axhline(lim1, xmin=0, xmax=1, color='k')
    lineh3 = ax.axhline(lim2, xmin=0, xmax=1, color='k')
if type == 'transect': # ocean and river sides
    if dim == '2d':
        ax.text(x[0] - 10, -20, label1, fontsize=fontsize - 4, fontweight='bold')
        ax.text(x[-1] - 10, -20, label2, fontsize=fontsize - 4, fontweight='bold')
    elif dim == '3d':
        ax.text(x[0] - 10, -40, label1, fontsize=fontsize - 4, fontweight='bold')
        ax.text(x[0] - 10, -40, label1, fontsize=fontsize - 4, fontweight='bold')
        ax.text(x[-1] - 10, -40, label2, fontsize=fontsize - 4, fontweight='bold')

# TODO : for the 2 files
# Plot the vertical profiles only
xver = 0  # increment only for the list of "stations"
for xvertical in profilevertical:  # I changed here to test
    # s = station[xver]
    s = df2['stations'].iloc[xver]
    #percentage =
    print('station', s)
    if df2['numero'].iloc[xver] == -10:  # skipped because too many station in one place
        print('station skipped', df2['numero'].iloc[xver])
    else:
        ax.axvline(xvertical, color='black')
        if (transect == 'SF_24.1' and s in ['AF12', 'AF13', 'AF14', 'AF15', 'AF16', 'AF17', 'AF18']):
            print('station SF24.1 skipped')
        else:
            ax.text(xvertical - shift, y=np.max(yplot) + 1, s=s, color='k',fontsize=fontsize - 4)
            # WARNING : the xvertical-300 depends on the xaxis size!!! 300 is ok for T4
            ax.text(xvertical - shift, y= 1, s=s, color='k',fontsize=fontsize - 4)

    xver = xver + 1

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
    xver = 0  # increment only for the list of "stations"
    for xvertical in profilevertical:  # I changed here to test
        # s = station[xver]
        s = df2['stations'].iloc[xver]
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

    outfile = 'current_magnitude_direction_stations_cmaptwilight' + transect + '_' + day + month + '_2407.png'
    fig.savefig(outfile, format='png')
    # plt.show()

    sys.exit(1)



elif var == 'profile':
    ###################################################################################################################
    # Parameters for the 2d plot
    if transect == 'T4' or month == '08':
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

        ax = fig.add_subplot(2, 1, 2, projection=dim)  # , sharex='row')#,sharey = ax1,sharez = ax1)
        ax.set_zlabel('Depth (m)', fontsize=fontsize - 3, labelpad=labelpad - 4)
        ax.set_zlim(0, 35)
        ax.set_zticks(y)
        ax.set_zticklabels(tickslabels=depth2, fontsize=fontsize - 4)  # depth[::int(nb_bin/2)]
        # ax.set_xlabel('Velocity East (m/s)', fontsize=fontsize-2, labelpad=labelpad)
        ax.set_xlabel('Profiles', fontsize=fontsize - 2, labelpad=labelpad)
        ax.set_xlim(-limvel, limvel * 2)
        ax.set_xticks(np.arange(-limvel, len(station_CTD) * limvel + 0.1, val))
        # print('station2', station2)
        ax.set_xticklabels(station2_CTD)
        # ax.set_ylabel('Velocity North (m/s)', fontsize=fontsize-2, labelpad=labelpad-6)
        # ax.set_ylabel('', fontsize=fontsize-2, labelpad=labelpad-6)
        ax.set_ylim(-limvel, limvel)
        ax.set_yticks(np.arange(-limvel, limvel + 0.1, val))
        ax.set_yticklabels([])
        ax.tick_params(axis='x', pad=-1)

        # 3D parameters
        azim = -90
        dist = 7
        elev = 25
        ax.azim = azim
        ax.dist = dist
        ax.elev = elev

        if correct_dir_segment:
            qNor = ax.quiver3D(-limvel, 0, 0, 0, val, 0, color='grey', arrow_length_ratio=0.3)
            qEas = ax.quiver3D(-limvel, 0, 0, val, 0, 0, color='grey', arrow_length_ratio=0.3)
            ax.text(-3.2 * val, 0.5 * val, 1, s=str(val) + '\nm/s lateral', fontsize=fontsize - 5)
            ax.text(-3.2 * val, -0.9 * val, 0, s=str(val) + 'm/s to ocean', fontsize=fontsize - 5)  # -1.8*val
        # plot échelle à station 0=-limvel/s
        else:
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
            ax.scatter(zplot + (n - 1) * limvel, zplot, yplot, marker='x', s=10, color='k')
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
                sta_correctdir = df2['stations'][
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

                q = ax.quiver(zplot + (n - 1) * limvel, zplot, yplot, vxmoy, vymoy, zmoy, color='grey',
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

            n = n + 1

        else:
            print('ERROR in the dim')

        incr = incr + 1
    if dim == '3d':
        plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.95,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

    # plt.show()

    if No_save:
        print('PAS D ENREGISTREMENT')
        sys.exit(1)

    else:

        outfile = 'current_magnitude_profiles_velocities_test'
        if layer == 1:
            outfile = outfile + '1layer_'
        elif layer == 2:
            outfile = outfile + '2layers_'
        if color_arrow_dir:
            outfile = outfile + 'coloredarrowdir_'
        if correct_dir_segment:
            outfile = outfile + 'correctdirsegment_'
        outfile = outfile + transect + '_' + day + month + '_' + dim + '_' + str(moy) + 'profilesavg_2407.png'
        print('outfile', outfile)

        fig.savefig(outfile, format='png')
