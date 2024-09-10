# 11/07/2022 Comparer T,S,D des différentes CTD (petite et Imer dans 1er temps, puis MIO)
# 08/06/2023 : Je modifie le code, la structure. Renamed with stations names in Data/CTD/Survey_month/CTD_LOG/Stations/
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression, TheilSenRegressor
# from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import sys, os, glob
# import xarray as xr
# import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import gsw
import random
import pickle
from itertools import zip_longest

# Small CTD, all different files
init = 'CC1238004_'
year = '2022'
list_month = ['June', 'August', 'Octobre']
i = 2 # 0 1 2 loop if needed ?
month = list_month[i]# day= ['16','17','18']
suffixe = '.csv'

# IMER CTD, one file different sheets
rep = '/home/penicaud/Documents/Data/CTD/'
f_imer = rep + 'Survey_' + month + '/CTD_imer_' + month + '.xlsx'
survey = 'Survey_' +month
d_imer = pd.ExcelFile(f_imer)
list_sheet = d_imer.sheet_names
# LOG CTD
rep_LOG = rep + survey + '/CTD_LOG/Stations/'
#liste = glob.glob(rep_LOG + '*' + suffixe)  # Go and find all the files
#liste = sorted(liste)

fontsize = 10
# maxdepth=[] #list of all the max depth of each profil
# ssalinity=[] #surface salinity (choose the 1st or 2d point of each profil
# date=[] #record the date of the CTD profil
# tempmax, tempmin, sal, density, turb, errorT, errorS =[],[],[],[],[],[],[]
cmap = plt.cm.binary  # grey
# cmap=plt.cm.hsv_r
# extracting all colors
color = [cmap(i) for i in range(cmap.N)]
color = color[15::10]  # good for the fig with all the profiles of T and S of the 18/06 in shades of grey
# because the colors go from white to black choose only from a certain phase and with a step ==> see some differences
# color = color[30::25] #good for hsv_r from purple - green -red on 6 curves

for l in list_sheet :
    station = l
    print('station ', station)
    f_LOG = glob.glob(rep_LOG + station + '_' + '*' + suffixe)
    if len(f_LOG) == 0 :
        print('No LOG file corresponding')
        continue
    f_LOG = f_LOG[0]
    # LOAD LOG DATA
    col_list_LOG = ["Depth (Meter)", "Temperature (Celsius)", "Conductivity (MicroSiemens per Centimeter)",
                    "Salinity (Practical Salinity Scale)", "Density (Kilograms per Cubic Meter)"]
    df_LOG = pd.read_csv(f_LOG, skiprows=28, usecols=col_list_LOG) # lambda x : x > 0 and x <= 27 )#, usecols=col_list)
    # df_LOG = pd.DataFrame(data_LOG, columns=['Depth (Meter)', "Temperature (Celsius)",
    #                                         "Salinity (Practical Salinity Scale)",
    #                                         "Density (Kilograms per Cubic Meter)"])
    depth_LOG = df_LOG['Depth (Meter)']
    T_LOG = df_LOG["Temperature (Celsius)"]
    S_LOG = df_LOG["Salinity (Practical Salinity Scale)"]
    D_LOG = df_LOG["Density (Kilograms per Cubic Meter)"]

    hour_LOG = str(int(f_LOG[-10:-8]) + 7)  # To have the local time UTC+7
    mn_LOG = f_LOG[-8:-6]  # minutes
    sec_LOG = f_LOG[-6:-4]
    if int(hour_LOG) >= 24:
        day = str(int(f_LOG[-13:-11]) + 1)
        hour_LOG = str(int(hour_LOG) % 24)
        print('CASE hour>24, new hour ', hour_LOG)
    else:
        day = f_LOG[-13:-11]
    print("hour LOG", hour_LOG, mn_LOG)
    # SA_LOG = gsw.SA_from_SP(S_LOG, p=0, lon=106.5, lat=20.5)
    # CT_LOG = gsw.CT_from_t(SA_LOG, T_LOG, p=0)

    # LOAD IMER DATA :
    col_list_imer = ["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"]
    data_imer = pd.read_excel(f_imer, station, skiprows=23,
                              usecols=col_list_imer)  # lambda x : x > 0 and x <= 27 )#, usecols=col_list)
    df_imer = pd.DataFrame(data_imer, columns=["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"])
    depth_imer = df_imer['Depth']
    T_imer = df_imer["Temp"]
    S_imer = df_imer["Salinity"]
    D_imer = df_imer["Density"]
    Chl = df_imer["Chl"]
    Turbidity = df_imer["Turbidity"]
    print('ok IMER')
    date_imer = pd.read_excel(f_imer, sheet_name=station, skiprows=12, usecols="A", nrows=2, header=None)
    date_imer = date_imer.to_numpy()
    imer_date = date_imer[0][0][10::]
    month_imer = imer_date[5:7]
    day_imer = imer_date[8:]
    imer_hour = date_imer[1][0][10::]
    # print('date imer ', type(date_imer2), print(date_imer2))
    hour_imer = imer_hour[0:2]
    mn_imer = imer_hour[3:5]  # minutes
    sec_imer = imer_hour[6::]
    print('hour imer : ', hour_imer, mn_imer, sec_imer)

    ##################    FIGURE     ##########################"
    fig, axs = plt.subplots(ncols=2)
    # , ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
    fig.suptitle(day_imer + '/' + month_imer + '/' + year + ', ' + station)
    ax = axs[0]
    ax.set_xlim(26, 32.1)  # (25,31)
    ax.set_ylim(-16.5, 0)
    # t1 = ax.scatter(T_LOG, -depth_LOG, alpha=0.8, marker='x', color='red', label='CTD LOG')
    t1 = ax.plot(T_LOG, -depth_LOG, color='red',
                 label='CTD LOG ' + hour_LOG + 'h' + mn_LOG)  # color=color[i], label=hour+'h'+mn)
    # t2 = ax.scatter(T_imer, -depth_imer, alpha=0.8, marker='x', color='gold', label='CTD IMER')
    t2 = ax.plot(T_imer, -depth_imer, color='gold', label='CTD IMER ' + hour_imer + 'h' + mn_imer)
    ax.set_xlabel('Temperature (°C)', fontsize=fontsize)  # ('Conservative Temperature (°C)', fontsize=fontsize)
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='lower right', fontsize=fontsize - 3)

    ax = axs[1]
    ax.set_xlim(-2, 32)
    ax.set_ylim(-16.5, 0)
    # s1 = ax.scatter(S_LOG, -depth_LOG, alpha=0.8, marker='x', color='blue' , label='CTD LOG')
    s1 = ax.plot(S_LOG, -depth_LOG, color='blue', label='CTD LOG ' + hour_LOG + 'h' + mn_LOG)
    # s2 = ax.scatter(S_imer, -depth_imer, alpha=0.8, marker='x', color='lightgreen', label='CTD IMER')
    s2 = ax.plot(S_imer, -depth_imer, color='lightgreen', label='CTD IMER ' + hour_imer + 'h' + mn_imer)
    # color=color[i] , label=hour+'h'+mn)#, label='GFF')
    ax.set_xlabel('Salinity (PSU)')  # ('Absolute Salinity (g/kg)', fontsize=fontsize)
    # ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.legend(loc='lower right', fontsize=fontsize - 3)
    plt.subplots_adjust(left=0.11,
                        bottom=0.1,
                        right=0.95,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    # outfile='TS_comp_CTD_LOG_IMER_' + day + month + year + '_' + hour_LOG + mn_LOG + '.png'
    outfile = rep + survey + '/Comp_2CTD/' + station + 'TS_comp_CTD_LOG_IMER_' + month + year + '.png'
    fig.savefig(outfile, format='png')


sys.exit(1)
# LOG CTD
rep_LOG = rep + survey + '/CTD_LOG/colocalise_CTD/'
print('f_imer', f_imer)
liste = glob.glob(rep_LOG + '*' + suffixe)  # Go and find all the files
liste = sorted(liste)
print('len CTD LOG files', len(liste))


dict_month = {'June': {'T1' : {'deb':1 , 'end' : 12, 'i' : 0}, 'T2': {'deb':12 , 'end' : 18, 'i' : 11},
                       'T3' : {'deb':18 , 'end' : 29, 'i' : 16}, 'T4' : {'deb':29 , 'end' : 40, 'i' : 27},
                       'fixe' : {'deb': 1, 'end': 26, 'i': 37}},
              'August': {'T1' : {'deb':1 , 'end' : 6, 'i' : 0}, 'T2': {'deb':6 , 'end' : 13, 'i' : 5},
                       'T3' : {'deb':26 , 'end' : 41, 'i' : 25}, 'T4' : {'deb':41 , 'end': 48, 'i' : 40},
                       'small fixe' : {'deb': 13, 'end': 26, 'i': 12}, 'fixe': {'deb': 1, 'end': 39, 'i': 47}},
              'Octobre': {'T1' : {'deb':1 , 'end' : 6, 'i' : 0}, 'T2': {'deb':6 , 'end' : 13, 'i' : 5},
                       'T3' : {'deb':26 , 'end' : 41, 'i' : 25}, 'T4' : {'deb':41 , 'end': 48, 'i' : 40},
                       'small fixe' : {'deb': 13, 'end': 26, 'i': 12}, 'fixe': {'deb': 1, 'end': 39, 'i': 47}}
            }

transect = 'T1'
# June
# T1 : S1 à S11
# T2 : S12 à S17
# T3 : S18 S28
# T4 : S29 S39
# TA1 : A1 à A5
# TA2 : A6 à A12
# TA3 : A26 A40
# TA4 : A41 A47
# SF1 : A13 A25
# SF_24 : AF1 AF38

for s in range(dict_month[month][transect]['deb'], dict_month[month][transect]['end']):
    i = dict_month[month][transect]['i']
    print('i ', i)
    if month == 'June':
        print('CASE June, 16-20 ')
        if transect == 'fixed':
            station = 'Fixed S.' + str(s)
        else:
            if s == 12:
                station = 'S12-1'
            elif s == 13:
                print('break 13')
                continue
            elif s == 36:
                print('break 36')
                continue
            else:
                station = 'S' + str(s)

    elif month == 'August':
        if transect == 'SF_24':
            if s in [5, 8, 10, 14, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:  # 14 exists but not on <ctd log
                print('break AF')
                continue
            else:
                station = 'AF' + str(s)
        else:
            if s == 17:
                print('break 17')
                continue
            if s == 25:
                station = 'A025-1'
                print('STATION 25')
            else:
                if s < 10:
                    station = 'A00' + str(s)
                else:
                    station = 'A0' + str(s)
    elif month == 'Octobre':
        print('Octobre')
    else:
        print('PB with the file IMER and the transect HERE')
    print('station', station)

    # open the Imer CTD
    col_list_imer = ["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"]
    data_imer = pd.read_excel(f_imer, station, skiprows=23,
                              usecols=col_list_imer)  # lambda x : x > 0 and x <= 27 )#, usecols=col_list)
    df_imer = pd.DataFrame(data_imer, columns=["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"])
    depth_imer = df_imer['Depth']
    T_imer = df_imer["Temp"]
    S_imer = df_imer["Salinity"]
    D_imer = df_imer["Density"]
    Chl = df_imer["Chl"]
    Turbidity = df_imer["Turbidity"]
    print('ok IMER')

    date_imer = pd.read_excel(f_imer, sheet_name=station, skiprows=13, usecols="A", nrows=1,
                              header=None)  # , names=["Start"].iloc[0]["Start"])
    date_imer = pd.DataFrame(date_imer)
    date_imer2 = date_imer.to_numpy()
    date_imer2 = date_imer2[0][0][10::]
    # print('date imer ', type(date_imer2), print(date_imer2))
    hour_imer = date_imer2[0:2]
    mn_imer = date_imer2[3:5]  # minutes
    sec_imer = date_imer2[6::]
    print('hour imer : ', hour_imer, mn_imer, sec_imer)

    # TODO : tableau de pression selon profondeur
    # p_imer=np.zeros(depth_imer, dtype=float)
    # p_imer=
    SA_imer = gsw.SA_from_SP(S_imer, p=0, lon=106.5, lat=20.5)
    CT_imer = gsw.CT_from_t(SA_imer, T_imer, p=0)
    # print('data_imer', data_imer)

    f_LOG = liste[i]
    # print('f_log', f_LOG)
    col_list_LOG = ["Depth (Meter)", "Temperature (Celsius)", "Conductivity (MicroSiemens per Centimeter)",
                    "Salinity (Practical Salinity Scale)", "Density (Kilograms per Cubic Meter)"]
    data_LOG = pd.read_csv(f_LOG, skiprows=28,
                           usecols=col_list_LOG)  # lambda x : x > 0 and x <= 27 )#, usecols=col_list)
    df_LOG = pd.DataFrame(data_LOG, columns=['Depth (Meter)', "Temperature (Celsius)",
                                             "Salinity (Practical Salinity Scale)",
                                             "Density (Kilograms per Cubic Meter)"])
    depth_LOG = df_LOG['Depth (Meter)']
    T_LOG = df_LOG["Temperature (Celsius)"]
    S_LOG = df_LOG["Salinity (Practical Salinity Scale)"]
    D_LOG = df_LOG["Density (Kilograms per Cubic Meter)"]
    # print('data_LOG', data_LOG)

    hour_LOG = str(int(f_LOG[-10:-8]) + 7)  # To have the local time UTC+7
    mn_LOG = f_LOG[-8:-6]  # minutes
    sec_LOG = f_LOG[-6:-4]
    if int(hour_LOG) >= 24:
        day = str(int(f_LOG[-13:-11]) + 1)
        hour_LOG = str(int(hour_LOG) % 24)
        print('CASE hour>24, new hour ', hour_LOG)
    else:
        day = f_LOG[-13:-11]

    print("hour LOG", hour_LOG, mn_LOG)
    SA_LOG = gsw.SA_from_SP(S_LOG, p=0, lon=106.5, lat=20.5)
    CT_LOG = gsw.CT_from_t(SA_LOG, T_LOG, p=0)
    i = i + 1

    print('max diff T', np.nanmax(abs(T_imer - T_LOG)), 'max diff sal', np.nanmax(abs(S_imer - S_LOG)),
          np.nanmax(D_imer - D_LOG))

    # MSE_S=mean_squared_error(S_imer,S_LOG)
    # RMSE_S=np.sqrt((MSE_S))
    # print('RMSE_S', RMSE_S)
    # MSE_T=mean_squared_error(T_imer,T_LOG)
    # RMSE_T=np.sqrt(MSE_T)
    # print('RMSE_T', RMSE_T)

    # print('test normal distribution, if 2d value> 0.05, not concluding and so the sitribution is normal',
    #       stats.shapiro(S_imer - S_LOG))
    # print('test equal variance, if 2d value> 0.05, not concluding and so the variance are equals',
    #       stats.levene(S_imer,S_LOG, center='mean'))
    #
    # stats.ttest_ind(S_imer,S_LOG)
    #
    # print('T test  ', stats.ttest_ind(S_imer,S_LOG) )
    # sys.exit(1)

    # # print(depth.max())
    # maxdepth.append(depth_imer.max())  # table of the deepest depth of each measurement
    # ssalinity.append(S_imer[1])  # choice of the 2d point not 1st (0) : to defend
    # sal.append(S_imer.max())
    # tempmax.append(T_imer.max())
    # tempmin.append(T_imer.min())
    # density.append(D_imer.min())
    # density.append(D_imer.max())
    # turb.append(Turbidity.max())
    # errorT.append(np.nanmax((T_imer-T_LOG)/T_LOG*100)) #PAS OK car ne tient pas compte de la profondeur et fait terme à terme, mais pas meme pas
    # errorS.append(np.nanmax((S_imer-S_LOG)/S_LOG*100))
    # #date.append(hour + ':' + mn + ":" + sec)

    # print(len(S_imer), len(S_LOG), len((S_imer-S_LOG)/S_LOG*100))
    # print((S_imer-S_LOG)/S_LOG*100)

    # Figure of 3 subplots
    fig, axs = plt.subplots(
        ncols=2)  # , ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
    fig.suptitle(day + '/' + month + '/' + year + '. ' + transect + ', ' + station)
    ax = axs[0]
    ax.set_xlim(26, 32.1)  # (25,31)
    ax.set_ylim(-16.5, 0)
    # t1 = ax.scatter(T_LOG, -depth_LOG, alpha=0.8, marker='x', color='red', label='CTD LOG')
    t1 = ax.plot(T_LOG, -depth_LOG, color='red',
                 label='CTD LOG ' + hour_LOG + 'h' + mn_LOG)  # color=color[i], label=hour+'h'+mn)
    # t2 = ax.scatter(T_imer, -depth_imer, alpha=0.8, marker='x', color='gold', label='CTD IMER')
    t2 = ax.plot(T_imer, -depth_imer, color='gold', label='CTD IMER ' + hour_imer + 'h' + mn_imer)
    ax.set_xlabel('Temperature (°C)', fontsize=fontsize)  # ('Conservative Temperature (°C)', fontsize=fontsize)
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='lower right', fontsize=fontsize - 3)

    ax = axs[1]
    ax.set_xlim(-2, 32)
    ax.set_ylim(-16.5, 0)
    # s1 = ax.scatter(S_LOG, -depth_LOG, alpha=0.8, marker='x', color='blue' , label='CTD LOG')
    s1 = ax.plot(S_LOG, -depth_LOG, color='blue', label='CTD LOG ' + hour_LOG + 'h' + mn_LOG)
    # s2 = ax.scatter(S_imer, -depth_imer, alpha=0.8, marker='x', color='lightgreen', label='CTD IMER')
    s2 = ax.plot(S_imer, -depth_imer, color='lightgreen', label='CTD IMER ' + hour_imer + 'h' + mn_imer)
    # color=color[i] , label=hour+'h'+mn)#, label='GFF')
    ax.set_xlabel('Salinity (PSU)')  # ('Absolute Salinity (g/kg)', fontsize=fontsize)
    # ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.legend(loc='lower right', fontsize=fontsize - 3)

    # ax = axs[1,0]
    # #ax.set_xlim(26, 32.1)  # (25,31)
    # ax.set_ylim(-16.5, 0)
    # #t1 = ax.scatter(T_LOG, -depth_LOG, alpha=0.8, marker='x', color='red', label='CTD LOG')
    # t1=ax.plot(T_LOG, -depth_LOG, color='red', label='CTD LOG' )  # color=color[i], label=hour+'h'+mn)
    # #t2 = ax.scatter(T_imer, -depth_imer, alpha=0.8, marker='x', color='gold', label='CTD IMER')
    # t2=ax.plot(T_imer, -depth_imer, color='gold', label='CTD IMER')
    # ax.set_xlabel('Temperature (°C)', fontsize=fontsize)#('Conservative Temperature (°C)', fontsize=fontsize)
    # ax.set_ylabel('Depth (m)', fontsize=fontsize)
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.legend(loc='lower right', fontsize=fontsize-3)
    #
    # ax = axs[1,1]
    # #ax.set_xlim(-2, 32)
    # ax.set_ylim(-16.5, 0)
    # #s1 = ax.scatter(S_LOG, -depth_LOG, alpha=0.8, marker='x', color='blue' , label='CTD LOG')
    # s1= ax.plot(S_LOG, -depth_LOG, color='blue', label='CTD LOG')
    # #s2 = ax.scatter(S_imer, -depth_imer, alpha=0.8, marker='x', color='lightgreen', label='CTD IMER')
    # s2= ax.plot(S_imer, -depth_imer, color='lightgreen', label='CTD IMER')
    # # color=color[i] , label=hour+'h'+mn)#, label='GFF')
    # ax.set_xlabel('Salinity (PS)')#('Absolute Salinity (g/kg)', fontsize=fontsize)
    # # ax.set_ylabel('Depth (m)', fontsize=fontsize)
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(5))
    # ax.legend(loc='lower right', fontsize=fontsize-3)

    # ax = axs[2]
    # ax.set_xlim(990, 1020)
    # ax.set_ylim(-16.5, 0)
    # #d1 = ax.scatter(D_LOG, -depth_LOG, alpha=0.8, marker='x', color='k', label='CTD LOG')
    # d1=ax.plot(D_LOG, -depth_LOG, color='k', label='CTD LOG')
    # #d2 = ax.scatter(D_imer, -depth_imer, alpha=0.8, marker='x', color='grey', label='CTD IMER')
    # d2=ax.plot(D_imer, -depth_imer, color='grey', label='CTD IMER')
    # # color=color[i],label=hour + 'h' + mn)  # , label='GFF')
    # ax.set_xlabel('Density (kg/m\u00b3)', fontsize=fontsize)
    # # ax.set_ylabel('Depth (m)', fontsize=fontsize)
    # ax.xaxis.set_major_locator(MultipleLocator(10))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(5))
    # ax.legend(loc='lower right', fontsize=fontsize-3)

    # ax = axs[3]
    # ax.set_xlim(0, 200)
    # ax.set_ylim(-16.5, 0)
    # #d1 = ax.scatter(D_LOG, -depth_LOG, alpha=0.8, marker='x', color='k', label='CTD LOG')
    # turb=ax.plot(Turbidity, -depth_imer, color='brown', label='CTD IMER')
    # #d2 = ax.scatter(D_imer, -depth_imer, alpha=0.8, marker='x', color='grey', label='CTD IMER')
    # #d2=ax.plot(D_imer, -depth_imer, color='grey', label='CTD IMER')
    # # color=color[i],label=hour + 'h' + mn)  # , label='GFF')
    # ax.set_xlabel('Turbidity (FTU)', fontsize=fontsize)
    # # ax.set_ylabel('Depth (m)', fontsize=fontsize)
    # ax.xaxis.set_major_locator(MultipleLocator(50))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(25))
    # ax.legend(loc='lower right', fontsize=fontsize-3)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.11,
                        bottom=0.1,
                        right=0.95,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    # outfile='TS_comp_CTD_LOG_IMER_' + day + month + year + '_' + hour_LOG + mn_LOG + '.png'
    outfile = 'TS_comp_CTD_LOG_IMER_' + day + month + year + '_' + station + '.png'
    fig.savefig(outfile, format='png')

    # plt.legend(ncol=3, fontsize='xx-small')
    # for single plots
    # plt.show()
    # sys.exit(1)
    # outfile='CT_SA_D_' + d + month + year + '_' + hour + mn + sec + '.png' #for CT SA figures
    # outfile2='TS_adjust_' + d + month + year + '_' + hour + mn + sec + '.png' #
    # fig.savefig(outfil) #for single plots, station by station

#
#         # Extraire données de surface : moyennes des premiers points sur les 2m ? Juste les 2 premiers points ? Le min des 2/3 premiers points ?
#         #print('salinity', type(ssalinity), len(ssalinity), ssalinity)
#         #record all the salinity of the 2d point of salinity profiles
#         #file='Surface_salinity_' + d + month + year
#         #print('f', file)
#         #l=[date,ssalinity]
#         #export_data = zip_longest(*l, fillvalue='')
#         #with open(file+'.csv', 'w', encoding="ISO-8859-1", newline='') as f:
#         #    pickle.dump(ssalinity, f)
#         #    write = csv.writer(f)
#         #    write.writerow(('Time', 'Surface Salinity'))
#         #    write.writerows(export_data)
#
plt.show()
# print('max depth' , np.nanmax(maxdepth)) #give the maximum depth over the whole period
# print('min depth' , np.nanmin(maxdepth)) #give the min depth over the whole period
# print('T' , np.nanmin(tempmin), np.nanmax(tempmax))
# print('S' , np.nanmin(sal), np.nanmax(sal))
# print('density', np.nanmin(density), np.nanmax(density))
# print('turbidity', np.nanmin(turb), np.nanmax(turb))
# print('% error T', np.nanmin(errorT), np.nanmax(errorT))
# print('% error S', np.nanmin(errorS), np.nanmax(errorS))
