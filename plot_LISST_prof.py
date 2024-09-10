# 25/05/23 : Je créée les plots pour visualiser les pb de profondeurs avec les données du LISST
# 19/06/23 : I add all stations (even if only CTD or ADCP values)
import pandas as pd
import numpy as np
import sys, os
import datetime
import scipy.signal as signal
import re
from openpyxl import load_workbook
import gsw
import scipy.integrate as integrate
import matplotlib.pyplot as plt

figure = False
Pcorr = True
# VARIABLES DATE REP
year = '2022'
list_month = ['June', 'August', 'Octobre']
i = 2 # 0 1 2 a voir pour faire une boucle si besoin
month = list_month[i]
rep = '/home/penicaud/Documents/Data/LISST/'

# LISST FILE : sert juste pour les stations
rep_LISST = '/home/penicaud/Documents/Data/LISST/'
suffixe = '.xlsx'
file_LISST = rep_LISST + 'LISST_' + month + '_' + year + suffixe
print('file LISST : ', file_LISST)

dict_char = {'June' : 'S',
             'August' : 'A' ,
             'Octobre' : 'O'}

# DECLARATION OF LIST
list_noms_col = []
list_min, list_max, list_sta = [], [], []
list_class = ['1.25-1.48', '1.48-1.74', '1.74-2.05', '2.05-2.42', '2.42-2.86', '2.86-3.38', '3.38-3.98', '3.98-4.70',
              '4.70-5.55', '5.55-6.55', '6.55-7.72', '7.72-9.12', '9.12-10.8', '10.8-12.7', '12.7-15.0', '15.0-17.7',
              '17.7-20.9', '20.9-24.6', '24.6-29.1', '29.1-34.3', '34.3-40.5', '40.5-47.7', '47.7-56.3', '56.3-66.5',
              '66.5-78.4', '78.4-92.6', '92.6-109', '109-129', '129-152', '152-180', '180-212', '212-250']
for i in range(len(list_class)):
    list_noms_col.append('#' + str(i + 1))

list_noms_col = list_noms_col + ['laser trans', 'battery V', 'ext aux input', 'laser ref', 'P', 'T', 'int date',
                                 'int date 2', 'inc 1', 'inc 2']#, 'Date', 'Hour']
print(list_noms_col)
print(len(list_noms_col))

# ICI, je fais une liste a partir des stations contenues dans le LISST
d1 = pd.ExcelFile(file_LISST)  # df du LISST
list_sheet_LISST = d1.sheet_names  # Liste des sheet de fil_LISST
list_sheet_LISST = [col for col in list_sheet_LISST if dict_char[month] in col]
list_sheet_LISST_station = [col for col in list_sheet_LISST if not 'F' in col]
list_sheet_LISST_f = [col for col in list_sheet_LISST if 'F' in col]
indice = range(len(list_sheet_LISST))

dict_month = {'June': {'nrows': 87},
              'August': {'nrows': 86},
              'Octobre': {'nrows': 111}}
rep = '/home/penicaud/Documents/Data/Survey_' + month
file_station = rep + '/Stations_' + month + '.xlsx'
df_global = pd.read_excel(file_station, sheet_name=0, nrows=dict_month[month]['nrows'])  # read the stations name
df_global = df_global.dropna(subset=['Stations'])
list_station = df_global['Stations'].values  # 07/06/23


def day_to_date(day_from, year):
    date_format = '%Y-%m-%d'
    # create a datetime object for January 1st of the given year
    start_date = datetime.datetime(year, 1, 1)
    # add the number of days to the start date
    result_list = []
    for d in range(len(day_from)) :
        result_date = start_date + datetime.timedelta(days=int(day_from[d] - 1))
        result_list.append(result_date)
    # format the date string using the specified format
    return result_list

def calcul_date(date_1,date_2,year) :
    day_from = (date_1/100).astype(int)
    start_date = datetime.datetime(int(year), 1, 1)
    # add the number of days to the start date
    result_list = []
    for d in (day_from.index) :
        result_date = start_date + datetime.timedelta(days=int(day_from[d] - 1))
        result_list.append(result_date)
    day = result_list
    #day = day_to_date(day_from,int(year))
    hour = ((date_1/100 - (date_1/100).astype(int) ) *100) # WARNING : +1 necessary to have right value if astype(int) at the end
    mn = (date_2/100)
    sec = (((date_2/100)-(date_2/100).astype(int))*100)#.astype(int)
    date_list = []
    for d in range(len(day)):
        date = datetime.datetime(int(year), day[d].month , day[d].day, int(hour[d]), int(mn[d]), int(sec[d]))
        date_list.append(date)
    return date_list

#for i in indice:  # len(indice)] : #indice #45:46
for i in list_station :
    #station = str(list_sheet_LISST[i])
    station = i
    print('station', station)

    data_LISST = pd.read_excel(file_LISST, usecols='A:AP', sheet_name=station, skiprows=2)
    data_LISST.columns = list_noms_col
    data_LISST = data_LISST[data_LISST.first_valid_index():data_LISST.last_valid_index()] # or dropna would have been enough
    # in order to avoid nan value everywhere, because nan values cannot be understood in next calculations
    data_LISST = data_LISST.reset_index()
    # FORMAT of the date with the excel file : int date 1 : ex 16713 = 167e jour 13h int date 2 : 2930 : 29m30s
    date_1 = data_LISST['int date']
    date_2 = data_LISST['int date 2']
    date_list = calcul_date(date_1,date_2,int(year))
    date = date_list[0]
    print( 'date OK')
    P = data_LISST['P']
    P = P[2:] # supprime les 2 1eres lignes qui contiennent nan ou mauvaises val
    #je refais le traitement fait dans LISST.py
    data_month = data_LISST.copy()
    data_month = data_month.dropna() # apparently not used, but in case

    # 1e CRITERION : skip if file does not have more than X different depth
    if np.shape(data_month['P'].drop_duplicates().index.values)[0] < 10:
        continue

    max = data_month['P'].max()
    #idxmax = data_month['P'].drop_duplicates(keep='last').idxmax() # permet de trouver index de la dernière rep de cette valeur max
    idxmax = data_month['P'].idxmax() # keep only the 1st val of the maxval to avoid invalid data of resuspension
    min1 = data_month['P'].loc[0:idxmax].min() #min on the down profile
    idxmin = data_month['P'].idxmin()
    min2 = data_month['P'].loc[idxmax:].min() # min on the up profile
    idxmin2 = data_month['P'].loc[idxmax:].idxmin()
    list_min.append([min1,min2])
    list_max.append(max)
    print(min1, 'index_min', idxmin, '\n', max, idxmax)
    list_sta.append(station) # only after the last check to have final values

    data_month2 = data_month.loc[idxmin:idxmax].copy()
    data_month2 = data_month2[(data_month2 != 0).all(axis=1)]  # Delete all 0 only after the determination of idxmin max
    # ID ; plutôt que de supprimer partout ou c'est à 0 on garde les valeurs des profs, et on met à -1000 par ex, pour pouvoir comparer avec les autres profs
    # data_month2.where(cond = data_month2 == 0, inplace = -1000, axis=1)

    # CRITERE 2
    if np.shape(data_month2['P'].drop_duplicates().index.values)[
        0] < 10:  # Skip if no more than X values in the file i.e. not enough depth
        continue


    if (min1 > 1.5):  # TODO A VOIR SI ON FAIT CA QUE POUR AOUT OU POUR TOUTES LES CAMPAGNES
        print('Recalculation of the min')
        data_month2['Pcorr'] = data_month2['P'].values - (
                min1 - 0.5)  # on va considérer que première mesure est à 50cm sous surface
    else:
        print('Pcorr = P')
        data_month2['Pcorr'] = data_month2['P']


    #TODO : criterion if the shape of the bottom is flat : chose only the beginning to avoid resuspension
    print('shape data 1 : ', np.shape(data_month))
    print('shape data 2 : ', np.shape(data_month2), 'diff', np.shape(data_month)[0] - np.shape(data_month2)[0])
    P2 = data_month2['P']
    P3 = data_LISST[data_LISST['#1'] == 0]
    P3 = P3['P']

    P4 = data_month2['Pcorr']

    if figure :
        fontsize = 10
        fig, ax = plt.subplots()
        fig.suptitle(month+' '+station)
        ax.set_xlabel('Measure points', fontsize=fontsize)
        ax.set_ylabel('Depth (m)', fontsize=fontsize)
        ax.grid(True, which='major')
        ax.plot(P.index, -P.values,  marker='+', color='blue', label='raw data')
        ax.plot(P2.index, -P2.values,  color='red', label='selected data')
        ax.scatter(P3.index, -P3.values, color='black', label='0 data')
        outfile = rep_LISST + 'Survey_' + month + '/figure/Pb_depth/Pb_depth_'+month+'_'+station
        if Pcorr :
            ax.plot(P4.index , -P4.values, color='green', label='Pcorr')
            outfile = outfile + '_withPcorr'
        ax.legend()
        fig.savefig(outfile, format='png')

print('list depth min down profile and up ')
df_depth = pd.DataFrame(list_min)
df_depth = df_depth.rename(columns={0: 'min down', 1 : 'min up'})
df_depth['max'] = list_max
df_depth['height down'] = df_depth['max'] - df_depth['min down']
df_depth['height up'] = df_depth['max'] - df_depth['min up']
df_depth['index'] = list_sta
df_depth = df_depth.set_index('index')

rep2 = '/home/penicaud/Documents/Data/'
outfile = rep2 + 'Survey_' + month + '/diff_depth_surface_LISST_'+month + '.xlsx'
df_depth.to_excel(outfile, header=True)

#file_LISST = rep + 'Survey_' + month + '/' + station + '_SPM' # + '_#' + str(classe_first + 1) + '-#' + str(classe_last)