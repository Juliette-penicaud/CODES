#18/07/2022 Graph évolution des T, S, turbidity focntion des stations / temps
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression, TheilSenRegressor
#from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
#import csv
import sys, os, glob
#import xarray as xr
#import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import gsw
import random
import pickle
from itertools import zip_longest


#Small CTD, all different files
#CC1238004_20220604_050128
init= 'CC1238004_'
suffixe = '.csv'
year='2022'

month= '08' #TO CHANGE
#day= ['16','17','18']
if month=='06':
    survey='Survey_June'
elif month=='08':
    survey='Survey_August'


comp_LOG=1 #necessary to obtain the day
rep_LOG='/home/penicaud/Documents/Data/CTD/'+survey+'/CTD_LOG/colocalise_CTD/'



#IMER CTD, one file different sheets
rep2='/home/penicaud/Documents/Data/CTD/'
if month=='06':
    f_imer=rep2+survey+'/CTD_VU_16-20_06.xlsx'
elif month=='08':
    f_imer=rep2+survey+'/CTD_imer_aug2022.xlsx'

#sheets de S1 à S39 puis Fixed S.1 à 25 vérifeir correspondances puis G1 à G8 sauf G7

fontsize=10
maxdepth=[] #list of all the max depth of each profil
ssalinity=[] #surface salinity (choose the 1st or 2d point of each profil
date=[] #record the date of the CTD profil
tempmax, tempmin, sal, density, turb, errorT, errorS =[],[],[],[],[],[],[]

cmap=plt.cm.binary #grey
#cmap=plt.cm.hsv_r
# extracting all colors
color = [cmap(i) for i in range(cmap.N)]
color=color[15::10] #good for the fig with all the profiles of T and S of the 18/06 in shades of grey
# #because the colors go from white to black, choose only from a certain phase and with a step so that we can see some differences
#color = color[30::25] #good for hsv_r from purple - green -red on 6 curves
cmap=plt.cm.jet

liste=glob.glob(rep_LOG+'*'+suffixe)     #Go and find all the files
liste=sorted(liste)


print('WE ARE IN THE SURVEY OF ', month,len(liste), liste)

if month=='06':
    transect='T1' #TRANSECT TO CHANGE #T1 T2 T3 T4 or fixed
    # T1 : S1 à S11
    # T2 : S12 à S17
    # T3 : S18 S28
    # T4 : S29 S39
    if transect=='T1':
        sta=range(1,12)
        i=0 #different indice for CTD LOG to scroll the list of files and not beeing affect by the brek 13 36 ect
        tab_turb = np.empty((11, 137))  # 25 station with the IMER CTD, 9.8 meters max ==> around 100 points
    elif transect=='T2':
        sta=range(12,18)
        i=11
        tab_turb = np.empty((7, 137))  # 25 station with the IMER CTD, 9.8 meters max ==> around 100 points
    elif transect=='T3':
        sta=range(18,29)
        i=16
        tab_turb = np.empty((11, 137))  # 25 station with the IMER CTD, 9.8 meters max ==> around 100 points
    elif transect=='T4':
        sta=range(29,40)
        i=27
        tab_turb = np.empty((11, 137))  # 25 station with the IMER CTD, 9.8 meters max ==> around 100 points
    elif transect=='fixed':
        sta=range(1,26)
        i=37
        tab_turb = np.empty((25, 100))  # 25 station with the IMER CTD, 9.8 meters max ==> around 100 points
    else :
        print('ERROR, not good transect or fixed')
        sys.exit(1)


    tab_turb[:, :] = np.nan
    tab_S = np.copy(tab_turb)
    tab_T = np.copy(tab_turb)
    c=0
    for s in sta:
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
        print('transect ', transect)
        print('station', station)

        #open the Imer CTD
        col_list_imer=["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"]
        data_imer = pd.read_excel(f_imer, station, skiprows=23, usecols=col_list_imer)  # lambda x : x > 0 and x <= 27 )#, usecols=col_list)
        df_imer= pd.DataFrame(data_imer, columns=["Depth", "Temp", "Salinity", "Density", "Chl", "Turbidity"])
        depth_imer = df_imer['Depth']
        T_imer = df_imer["Temp"]
        S_imer = df_imer["Salinity"]
        D_imer = df_imer["Density"]
        Chl=df_imer["Chl"]
        Turbidity=df_imer["Turbidity"]
        print('ok IMER')

        date_imer = pd.read_excel(f_imer, sheet_name=station, skiprows=13,  usecols="A", nrows=1, header=None)#, names=["Start"].iloc[0]["Start"])
        date_imer=pd.DataFrame(date_imer)
        date_imer2=date_imer.to_numpy()
        date_imer2=date_imer2[0][0][10::]
        #print('date imer ', type(date_imer2), print(date_imer2))
        hour_imer = date_imer2[0:2]
        mn_imer = date_imer2[3:5]  # minutes
        sec_imer = date_imer2[6::]
        print('hour imer : ', hour_imer, mn_imer, sec_imer)
        #sys.exit(1)


        #TODO : tableau de pression selon profondeur
        #p_imer=np.zeros(depth_imer, dtype=float)
        #p_imer=
        SA_imer=gsw.SA_from_SP(S_imer, p=0, lon=106.5, lat=20.5)
        CT_imer=gsw.CT_from_t(SA_imer, T_imer, p=0)
        #print('data_imer', data_imer)



        if comp_LOG :
            f_LOG=liste[i]
            #print('f_log', f_LOG)
            col_list_LOG = ["Depth (Meter)", "Temperature (Celsius)", "Conductivity (MicroSiemens per Centimeter)",
                "Salinity (Practical Salinity Scale)", "Density (Kilograms per Cubic Meter)"]
            data_LOG = pd.read_csv(f_LOG, skiprows=28, usecols=col_list_LOG)#lambda x : x > 0 and x <= 27 )#, usecols=col_list)
            df_LOG=pd.DataFrame(data_LOG, columns=['Depth (Meter)', "Temperature (Celsius)",
                                           "Salinity (Practical Salinity Scale)",
                                           "Density (Kilograms per Cubic Meter)"])
            depth_LOG=df_LOG['Depth (Meter)']
            T_LOG=df_LOG["Temperature (Celsius)"]
            S_LOG=df_LOG["Salinity (Practical Salinity Scale)"]
            D_LOG=df_LOG["Density (Kilograms per Cubic Meter)"]
            #print('data_LOG', data_LOG)


            hour_LOG = str(int(f_LOG[-10:-8]) + 7)  # To have the local time UTC+7
            mn_LOG = f_LOG[-8:-6]  # minutes
            sec_LOG = f_LOG[-6:-4]
            day=f_LOG[-13:-11]


            print("hour LOG", hour_LOG, mn_LOG)
            SA_LOG=gsw.SA_from_SP(S_LOG, p=0, lon=106.5, lat=20.5)
            CT_LOG=gsw.CT_from_t(SA_LOG, T_LOG, p=0)
            i=i+1
            #print('max diff T', np.nanmax(abs(T_imer-T_LOG)), 'max diff sal', np.nanmax(abs(S_imer-S_LOG)), np.nanmax(D_imer-D_LOG))

        maxdepth.append(depth_imer.max())  # table of the deepest depth of each measurement
        ssalinity.append(S_imer[1])  # choice of the 2d point not 1st (0) : to defend
        sal.append(S_imer.max())
        tempmax.append(T_imer.max())
        tempmin.append(T_imer.min())
        density.append(D_imer.min())
        density.append(D_imer.max())
        turb.append(Turbidity.max())
        #errorT.append(np.nanmax((T_imer-T_LOG)/T_LOG*100)) #PAS OK car ne tient pas compte de la profondeur et fait terme à terme, mais pas meme pas
        #errorS.append(np.nanmax((S_imer-S_LOG)/S_LOG*100))

        #Idee : creer tableaux pour chaque param a pcolor (T,S,turb,..) de la taille en colonne du nb de station par transect, et en colonne du max de prof (136 pour T4)
        print('number of values at this station', len(depth_imer)) #give the max of the
        tab_S[c][0:len(depth_imer)]=S_imer
        tab_T[c][0:len(depth_imer)]=T_imer
        tab_turb[c][0:len(depth_imer)]=Turbidity
        #print('tab_S', tab_S[c])
        c=c+1

    print('T ', tab_T)
    print('S ', tab_S)
    print('no file written')
    sys.exit(1)


    #         # Extraire données de surface : moyennes des premiers points sur les 2m ? Juste les 2 premiers points ? Le min des 2/3 premiers points ?
    file='Tab_S_' + day + month + year + '_'+transect
    print('f', file)
    #l=[date,ssalinity]
    #export_data = zip_longest(*l, fillvalue='')
    with open(file+'.csv', 'w', encoding="ISO-8859-1", newline='') as f:
       #pickle.dump(ssalinity, f)
       write = csv.writer(f)
       #write.writerow(('Time', 'Surface Salinity'))
       #write.writerows(export_data)
       write.writerows(tab_S)

    file='Tab_T_' + day + month + year + '_'+transect
    print('f', file)
    #l=[date,ssalinity]
    #export_data = zip_longest(*l, fillvalue='')
    with open(file+'.csv', 'w', encoding="ISO-8859-1", newline='') as f:
       #pickle.dump(ssalinity, f)
       write = csv.writer(f)
       #write.writerow(('Time', 'Surface Salinity'))
       #write.writerows(export_data)
       write.writerows(tab_T)

    file='Tab_turb_' + day + month + year + '_'+transect
    print('f', file)
    with open(file+'.csv', 'w', encoding="ISO-8859-1", newline='') as f:
       #pickle.dump(ssalinity, f)
       write = csv.writer(f)
       #write.writerow(('Time', 'Surface Salinity'))
       #write.writerows(export_data)
       write.writerows(tab_turb)


##########################         AUGUST             ####################################

elif month=='08':
    nb_bin = int(19/0.1)
    nb_bin=200 #useful to have 20meters and so ticks refulary spaced of 5m
    transect = 'SF1'  # TRANSECT TO CHANGE #TA1 TA2 TA3 TA4 or SF1 or SF_24
    # TA1 : A1 à A5
    # TA2 : A6 à A12
    # TA3 : A26 A40
    # TA4 : A41 A47
    # SF1 : A13 A25 plus A25-1
    # SF_24 : AF1 AF38
    if transect == 'TA1':
        deb=1
        end=6 #num station +1 because range(deb,end), end is excluded
        sta = range(deb, end)
        i = deb-1  # different indice for CTD LOG to scroll the list of files and not beeing affect by the brek 13 36 ect
        tab_turb = np.empty((end-deb, nb_bin))
    elif transect == 'TA2':
        deb=6
        end=13
        sta = range(deb, end)
        i = deb-1
        tab_turb = np.empty((end-deb, nb_bin))
    elif transect == 'TA3':
        deb=26
        end=41
        sta = range(deb, end)
        i = deb-1 #because we only use 25.2
        tab_turb = np.empty((end-deb, nb_bin))
    elif transect == 'TA4':
        deb=41
        end=48
        sta = range(deb,end)
        i = deb-1
        tab_turb = np.empty((end-deb, nb_bin))
    elif transect == 'SF1':
        deb=13
        end=26
        sta = range(deb, end)
        i = deb-1
        tab_turb = np.empty((end-deb, nb_bin))  #  +1 BECAUSE station 25-1
    elif transect == 'SF_24':
        deb = 1
        end = 39
        sta = range(deb, end)
        i = 47
        tab_turb = np.empty((end - deb, nb_bin))
    else:
        print('ERROR, not good transect or fixed')
        sys.exit(1)


    tab_turb[:, :] = np.nan
    tab_S = np.copy(tab_turb)
    tab_T = np.copy(tab_turb)
    c = 0

    for s in sta:
        if transect == 'SF_24':
            print('SF24 case')
            if s in [5,8,10,21,24,25,26,27,28,29,30,31,32,33,34,35]:
                print('break AF')
                continue
            else:
                station = 'AF' + str(s)
        else:
            if s==17:
                print('break 17')
                continue
            #if s==25 :
            #    station='A025-1'
            #    print('STATION 25')
            else :
                if s<10:
                    print('S<10')
                    station = 'A00' + str(s)
                else :
                    print('S>10')
                    station = 'A0' + str(s)
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
        print(df_imer)


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


        if comp_LOG :
            if station=='AF13':
                continue
            else :
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
                day = f_LOG[-13:-11]

                print("hour LOG", hour_LOG, mn_LOG)
                SA_LOG = gsw.SA_from_SP(S_LOG, p=0, lon=106.5, lat=20.5)
                CT_LOG = gsw.CT_from_t(SA_LOG, T_LOG, p=0)

                i=i+1


                # print('max diff T', np.nanmax(abs(T_imer-T_LOG)), 'max diff sal', np.nanmax(abs(S_imer-S_LOG)), np.nanmax(D_imer-D_LOG))

        maxdepth.append(depth_imer.max())  # table of the deepest depth of each measurement
        if len(S_imer)>2:
            ssalinity.append(S_imer[1])  # choice of the 2d point not 1st (0) : to defend
        else :
            ssalinity.append(S_imer[0])  # choice of the 2d point not 1st (0) : to defend

        sal.append(S_imer.max())
        tempmax.append(T_imer.max())
        tempmin.append(T_imer.min())
        density.append(D_imer.min())
        density.append(D_imer.max())
        turb.append(Turbidity.max())
        # errorT.append(np.nanmax((T_imer-T_LOG)/T_LOG*100)) #PAS OK car ne tient pas compte de la profondeur et fait terme à terme, mais pas meme pas
        # errorS.append(np.nanmax((S_imer-S_LOG)/S_LOG*100))

        # Idee : creer tableaux pour chaque param a pcolor (T,S,turb,..) de la taille en colonne du nb de station par transect, et en colonne du max de prof (136 pour T4)
        print('number of values at this station', len(depth_imer))  # give the max of the
        tab_S[c][0:len(depth_imer)] = S_imer
        tab_T[c][0:len(depth_imer)] = T_imer
        tab_turb[c][0:len(depth_imer)] = Turbidity
        # print('tab_S', tab_S[c])
        c = c + 1
        print('c val', c)
        print('i val', i)
        print('len depth imer', len(depth_imer))

    print("end of transect")
    #print('T ', tab_T)
    #print('S ', tab_S)

    #         # Extraire données de surface : moyennes des premiers points sur les 2m ? Juste les 2 premiers points ? Le min des 2/3 premiers points ?
    file = 'Tab_S_' + day + month + year + '_' + transect
    if nb_bin==200:
        file=file+'_20m'
    print('f', file)
    # l=[date,ssalinity]
    # export_data = zip_longest(*l, fillvalue='')
    with open(file + '.csv', 'w', encoding="ISO-8859-1", newline='') as f:
        # pickle.dump(ssalinity, f)
        write = csv.writer(f)
        # write.writerow(('Time', 'Surface Salinity'))
        # write.writerows(export_data)
        write.writerows(tab_S)

    file = 'Tab_T_' + day + month + year + '_' + transect
    if nb_bin==200:
        file=file+'_20m'
    print('f', file)
    # l=[date,ssalinity]
    # export_data = zip_longest(*l, fillvalue='')
    with open(file + '.csv', 'w', encoding="ISO-8859-1", newline='') as f:
        # pickle.dump(ssalinity, f)
        write = csv.writer(f)
        # write.writerow(('Time', 'Surface Salinity'))
        # write.writerows(export_data)
        write.writerows(tab_T)

    file = 'Tab_turb_' + day + month + year + '_' + transect
    if nb_bin==200:
        file=file+'_20m'
    print('f', file)
    with open(file + '.csv', 'w', encoding="ISO-8859-1", newline='') as f:
        # pickle.dump(ssalinity, f)
        write = csv.writer(f)
        # write.writerow(('Time', 'Surface Salinity'))
        # write.writerows(export_data)
        write.writerows(tab_turb)

sys.exit(1)

#Figure of 3 subplots : hovmoller, T S and Turb, il faudra faire N2
fig, axs = plt.subplots(nrows=3)#, ncols=2)#fonctionne avec plot:share=1,left=3 , right=5,bottom=5,top=7,wspace=10, hspace=5)
fig.suptitle(day+'/'+month+'/'+year+', '+station)


#temperature
ax = axs[0]
#ax.set_xlim(26, 32.1)  # (25,31)
ax.set_ylim(-10, 0)
t1=ax.pcolor(tab_S, cmap=cmap, vmin=27, vmax=30, label='Temeprature °C')
ax.set_xlabel('stations ', fontsize=fontsize)#('Conservative Temperature (°C)', fontsize=fontsize)
ax.set_ylabel('Depth (m)', fontsize=fontsize)
#ax.xaxis.set_major_locator(MultipleLocator(2))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax.xaxis.set_minor_locator(MultipleLocator(1))
#ax.legend(loc='lower right', fontsize=fontsize-3)

plt.show()
sys.exit(1)

ax = axs[1]
ax.set_xlim(-2, 32)
ax.set_ylim(-16.5, 0)
#s1 = ax.scatter(S_LOG, -depth_LOG, alpha=0.8, marker='x', color='blue' , label='CTD LOG')
s1= ax.plot(S_LOG, -depth_LOG, color='blue', label='CTD LOG '+hour_LOG+'h'+mn_LOG )
#s2 = ax.scatter(S_imer, -depth_imer, alpha=0.8, marker='x', color='lightgreen', label='CTD IMER')
s2= ax.plot(S_imer, -depth_imer, color='lightgreen', label='CTD IMER '+hour_imer+'h'+mn_imer)
# color=color[i] , label=hour+'h'+mn)#, label='GFF')
ax.set_xlabel('Salinity (PSU)')#('Absolute Salinity (g/kg)', fontsize=fontsize)
# ax.set_ylabel('Depth (m)', fontsize=fontsize)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.legend(loc='lower right', fontsize=fontsize-3)

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

plt.show()
outfile='Hovmoller_T_S_turb_' + day + month + year + '_' +transect+'.png'
fig.savefig(outfile, format='png')
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


print('Depth', np.nanmin(maxdepth), np.nanmax(maxdepth))  # give the maximum depth over the whole period
print('T', np.nanmin(tempmin), np.nanmax(tempmax))
print('S', np.nanmin(sal), np.nanmax(sal))
print('density', np.nanmin(density), np.nanmax(density))
print('turbidity', np.nanmin(turb), np.nanmax(turb))
#print('% error T', np.nanmin(errorT), np.nanmax(errorT))
#print('% error S', np.nanmin(errorS), np.nanmax(errorS))