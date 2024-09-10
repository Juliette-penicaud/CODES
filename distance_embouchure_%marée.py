#25/07/2022 faire profils des TS a partir des fichiers excels /csv de la petite CTD
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression, TheilSenRegressor
#from sklearn.linear_model import RANSACRegressor
#import csv
import sys, os, glob
#import xarray as xr
#import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import gsw
import random
import pickle
from itertools import zip_longest
from mpl_toolkits import mplot3d

#2 versions : day by day or all data

#CC1238004_20220604_050128
rep='/home/penicaud/Documents/Data/CTD/'
init= 'CC1238004_'
year='2022'
month= '08' #TO CHANGE
day= ['16','17','18'] #USED FOR THE NOT ALL DATA
suffixe = '.csv'

data_sta='transect' #transect or all data, all data includes the fixed stations
var='Temperature' #Temperature, Salinity or Turbidity #TO CHANGE the variable to observe


if month=='06':
    survey='Survey_June'

    df = pd.DataFrame({'Stations': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12-1', 'S13',
                                    'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25',
                                    'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37',
                                    'S38', 'S39', ],
                       'Percentage': [59., 62.78, 66.11, 69.86, 72.78, 77.64, 80.83, 83.61, 86.39, 88.19, 92.36, 100.0,
                                      -3.89,
                                      -7.22, -11.11, -13.19, -15.0, 52.22, 54.44, 55.56, 58.33, 59.86, 61.53, 63.89,
                                      66.11, 68.61,
                                      70.97, 73.33, -20.97, -23.19, -25.69, -27.78, -29.72, -30.83, -32.22, -33.33,
                                      -36.39,
                                      -38.19, -41.39],
                       'Distance': [-3815, -2200, -352, 1180.45, 2059.71, 3154.4, 4638.59, 6800.46, 8042.91, 8941.99,
                                    6075.93,
                                    3123.59, 2543.97, 877.78, -570, -1642, -2923, 12581.53, 12596.93, 11368.68,
                                    10364.06,
                                    9126.97, 8553.47, 7555.29, 5667.21, 3660.98, 1332.25, -785, -1508, -1864, -3548,
                                    -3974, -4276,
                                    -4213, -4130, -4058, -5522, -6166, -6941, ]})


elif month=='08':
    survey='Survey_August'
    station_file='/home/penicaud/Documents/Data/survey_august/Stations_10-13_august.xlsx'
    col_list=['Stations', 'Percentage', 'Distance']
    df=pd.read_excel(station_file, usecols=col_list, nrows=87)

else :
    print('PB SURVEY')
    sys.exit(1)

rep=rep+survey+'/'



fontsize=10

cmap=plt.cm.binary #grey
#cmap=plt.cm.hsv_r
#15 ==> 90
# extracting all colors
color = [cmap(i) for i in range(cmap.N)]
color=color[20::20] #good for the fig with all the profiles of T and S of the 18/06 in shades of grey
# #because the colors go from white to black, choose only from a certain phase and with a step so that we can see some differences
#color = color[30::25] #good for hsv_r from purple - green -red on 6 curves


cmap=plt.cm.jet
#Dataframe of distance to mouth and % tide

# #diagram 3D
fig= plt.figure()
#fig, ax = plt.subplots()  # fonctionne avec plot : share=1,left=3 , right=5, bottom=5 , top=7, wspace=10, hspace=5)
fig.suptitle(var + ', '+survey)
ax = plt.axes(projection='3d')
ax.set_xlabel('Distance to the river mouth (km)', fontsize=fontsize)
ax.set_ylabel('% of tide', fontsize=fontsize)
ax.set_zlabel('depth (m)', fontsize=fontsize)
if month=='06':
    ax.set_zlim(-14,0)
    ax.set_xticks(np.arange(-7500,15000,2500))
    ax.set_xticklabels(np.arange(-7.5,15,2.5))
elif month=='08':
    ax.set_zlim(-18,0)
    ax.set_xticks(np.arange(-15000,15000,5000))
    ax.set_xticklabels(np.arange(-15,15,5))
#ax.set_xlim()
#ax.set_ylim(0,32.1)#(0, 31)
#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#ax.yaxis.set_major_locator(MultipleLocator(5))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax.yaxis.set_minor_locator(MultipleLocator(1))

#IMER CTD, one file different sheets
rep2='/home/penicaud/Documents/Data/CTD/'

if month=='06':
    f_imer=rep2+survey+'/CTD_VU_16-20_06.xlsx'
    #sheets de S1 à S39 puis Fixed S.1 à 25 vérifeir correspondances puis G1 à G8 sauf G7
    if data_sta=='transect':
        transect = ['T1', 'T2', 'T3', 'T4']
    elif data_sta=='All_data':
        transect=['T1', 'T2', 'T3', 'T4', 'fixed']
    else :
        print('ERROR TRANSECT')

elif month == '08':
    f_imer = rep2 + survey+'/CTD_imer_aug2022.xlsx'
    if data_sta=='transect':
        transect = ['TA1', 'TA2', 'TA3', 'TA4']
    elif data_sta=='All_data':
        transect = ['TA1', 'TA2', 'TA3', 'TA4', 'SF1', 'SF_24']
    else :
        print('ERROR TRANSECT')

print('file imer', f_imer)
for t in transect :
    print('transect', t)
    if t=='T1':
        sta=range(1,12)
        i=0
    elif t=='T2':
        sta=range(12,18)
        i=11
    elif t=='T3':
        sta=range(18,29)
        i=16
    elif t=='T4':
        sta=range(29,40)
        i=27
    elif t=='fixed':
        sta=range(1,26)
        i=37
    ###############   AUGUST  ##################
    elif t == 'TA1':
        deb = 1
        end = 6  # num station +1 because range(deb,end), end is excluded
        sta = range(deb, end)
        i = deb - 1  # different indice for CTD LOG to scroll the list of files and not beeing affect by the brek 13 36 ect
    elif t == 'TA2':
        deb = 6
        end = 13
        sta = range(deb, end)
        i = deb - 1
    elif t == 'TA3':
        deb = 26
        end = 41
        sta = range(deb, end)
        i = deb - 1  # because we only use 25.2
    elif t == 'TA4':
        deb = 41
        end = 48
        sta = range(deb, end)
        i = deb - 1
    elif t == 'SF1':
        deb = 13
        end = 26
        sta = range(deb, end)
        i = deb - 1
    elif t == 'SF_24':
        deb = 1
        end = 39
        sta = range(deb, end)
        i = 47
    else:
        print('ERROR, not good transect or fixed')
        sys.exit(1)


    for s in sta:
        if month=='06':
            if transect == 'fixed':
                station = 'Fixed S.' + str(s)
            else :
                if s==12 :
                    station='S12-1'
                elif s==13 :
                    print('break 13')
                    continue
                elif s==36 :
                    print('break 36')
                    continue
                else :
                    station='S'+str(s)

        elif month == '08':
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
        else:
            print('PB with the file IMER and the transect HERE')
        print('station', station)

        #open the Imer CTD
        col_list_imer=["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"]
        data_imer = pd.read_excel(f_imer, station, skiprows=23, usecols=col_list_imer)  # lambda x : x > 0 and x <= 27 )#, usecols=col_list)
        df_imer= pd.DataFrame(data_imer, columns=["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"])
        depth_imer = df_imer['Depth']
        #depth_imer=np.flip(depth_imer)
        depth_imer=-depth_imer
        T_imer = df_imer["Temp"]
        S_imer = df_imer["Salinity"]
        D_imer = df_imer["Density"]
        Chl=df_imer["Chl"]
        Turbidity=df_imer["Turbidity"]
        print('ok IMER')

        if var=='Temperature':
            var_imer=T_imer
            vmin=28.5
            vmax=31.5
        elif var=='Salinity':
            var_imer=S_imer
            vmin=0
            vmax=30
        elif var=='Turbidity':
            var_imer=Turbidity
            vmin=0
            vmax=200
        else :
            print('PB with the name of the variable')

        date_imer = pd.read_excel(f_imer, sheet_name=station, skiprows=13,  usecols="A", nrows=1, header=None)#, names=["Start"].iloc[0]["Start"])
        date_imer=pd.DataFrame(date_imer)
        date_imer2=date_imer.to_numpy()
        date_imer2=date_imer2[0][0][10::]
        #print('date imer ', type(date_imer2), print(date_imer2))
        hour_imer = date_imer2[0:2]
        mn_imer = date_imer2[3:5]  # minutes
        sec_imer = date_imer2[6::]
        print('hour imer : ', hour_imer, mn_imer, sec_imer)

        df2 = df.loc[df['Stations'] == station]
        print(df2)
        #print(df2['Distance'])
        p=ax.scatter3D( df2['Distance'], df2['Percentage'], depth_imer, c=var_imer, cmap=cmap, vmin=vmin, vmax=vmax )
        #plt.show()

c=plt.colorbar(p, ax=ax)
outfile='3D_dist_%tide_'+survey+'_'+var+'_'+data_sta+'.png'
fig.savefig(outfile, format='png')
print('Saved fig', outfile)
plt.show()
sys.exit(1)

for d in day:
    for t in transect:
        y = np.linspace(0, 140, 5)
        prof = ['-14', '-10.5', '-7', '-3.5', '0']
        if t == 'T1':
            day = '16'
            stations = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
            x = np.arange(0, 11, 1)
        elif t == 'T2':
            day = '16'
            stations = ['S12', 'S13', 'S14', 'S15', 'S16', 'S17']
            x = np.arange(0, 6, 1)
        elif t == 'T3':
            day = '17'
            stations = ['S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28']
            x = np.arange(0, 11, 1)
        elif t == 'T4':
            day = '17'
            stations = ['S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S37', 'S38', 'S39']
            x = np.arange(0, 10, 1)
        elif t == 'fixed':
            day = '18'
            # stations=['1']
            x = np.arange(0, 25, 1)
            y = np.linspace(0, 10, 5)
            prof = ['-10', '-5', '0']
        else:
            print('PROBLEM OF DAY')
        i = 0

    i = 0  # to select the color of the table
    liste=glob.glob(rep+init+year+month+d+'*'+suffixe)     #Go and find all the files
    for f in sorted(liste):

        col_list = ["Depth (Meter)", "Temperature (Celsius)", "Conductivity (MicroSiemens per Centimeter)",
        "Salinity (Practical Salinity Scale)", "Density (Kilograms per Cubic Meter)"]
        data = pd.read_csv(f, skiprows=28, usecols=col_list)#lambda x : x > 0 and x <= 27 )#, usecols=col_list)
        df=pd.DataFrame(data, columns=['Depth (Meter)', "Temperature (Celsius)",
                                       "Salinity (Practical Salinity Scale)",
                                       "Density (Kilograms per Cubic Meter)"])
        depth=df['Depth (Meter)']
        T=df["Temperature (Celsius)"]
        S=df["Salinity (Practical Salinity Scale)"]
        D=df["Density (Kilograms per Cubic Meter)"]
        Depth=df["Depth (Meter)"]

        p=Depth#1013.25
        SA=gsw.SA_from_SP(S,p, 106.5, 20.5)
        CT=gsw.CT_from_t(SA,T,p)
        PT=gsw.pt0_from_t(SA, T, p)
        [N2, p_mid]=gsw.Nsquared(SA,CT,p, 20.5)

        print('p', p)
        print('N2', N2, p_mid)

        ax.scatter3D()

    fig.savefig(''+d+month+year+'.png')
    plt.show()
